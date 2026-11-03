"""Multi-episode decision memory with zero or more exact tool dependencies.

Host-owned evidence and behavior checks reuse decision_v2's admission gate.
No model-supplied test scores or automatic compatibility assumptions are accepted.
"""
from datetime import datetime, timezone
import re
from uuid import uuid4

from rsagent.evolution.decision_skill import DecisionSkillRegistry, LIMIT
from rsagent.evolution.tool_registry import canonical, digest, identifier, safe_bytes, strict_json


def bindings_by_name(bindings):
    if not isinstance(bindings, list):
        raise ValueError("Tool bindings must be a list")
    result = {}
    for item in bindings:
        if not isinstance(item, dict) or set(item) != {"name", "revision", "sha256"}:
            raise ValueError("Need exact tool identities")
        name = identifier(item["name"])
        if name in result or not isinstance(item["revision"], str) or not re.fullmatch(r"[a-f0-9]{32}", item["revision"]):
            raise ValueError("Duplicate tool or invalid revision")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise ValueError("Invalid tool hash")
        result[name] = dict(item)
    return result


class ContinualSkillRegistry(DecisionSkillRegistry):
    document_fields = {"title", "rules", "unknowns", "requires_tools"}

    def set_learning_contract(self, evidence, tool_bindings, behavior_policy):
        """Host-only append of immutable evidence; older revisions keep their context."""
        available = bindings_by_name(tool_bindings)
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError("Need evidence")
        for name, value in evidence.items():
            if (not isinstance(name, str) or not isinstance(value, dict)
                    or set(value) != {"text", "source", "sha256"}
                    or not isinstance(value["text"], str) or not isinstance(value["source"], str)
                    or digest(value["text"].encode()) != value["sha256"]):
                raise ValueError("Invalid evidence identity/hash")
        if (not isinstance(behavior_policy, dict) or not behavior_policy
                or not all(isinstance(k, str) and isinstance(v, str)
                           and re.fullmatch(r"[a-f0-9]{64}", v) for k, v in behavior_policy.items())):
            raise ValueError("Need behavior case fingerprints")
        contract = {"schema": "decision_skill_v3", "evidence": evidence,
                    "available_tools": available, "behavior_policy": behavior_policy}
        encoded = canonical(contract).encode()
        if len(encoded) > 512 * 1024:
            raise ValueError("Contract too large")
        contract_hash = digest(encoded)
        with self.locked():
            self.write(f"contracts/{contract_hash}.json", contract)
            self.write("learning-context.json", {"contract_hash": contract_hash})
            self.event("skill_learning_context", contract_hash=contract_hash)
        return contract_hash

    def set_decision_contract(self, *args, **kwargs):
        raise ValueError("decision_v3 uses set_learning_contract")

    def propose(self, name, source, parent=None):
        identifier(name)
        data = safe_bytes(source, LIMIT)
        doc = strict_json(data.decode())
        required = doc.get("requires_tools") if isinstance(doc, dict) else None
        if (not isinstance(required, list) or len(required) > 8
                or not all(isinstance(x, str) for x in required) or len(set(required)) != len(required)):
            raise ValueError("requires_tools must list 0..8 distinct tool names")
        with self.locked():
            context = self.read("learning-context.json")
            if not context:
                raise ValueError("Host learning context missing")
            contract = self.read(f"contracts/{context['contract_hash']}.json")
            if any(n not in contract["available_tools"] for n in required):
                raise ValueError("Unknown tool dependency")
            if self.index().get(name, {}).get("active") != parent:
                raise ValueError("Parent must match active revision")
            revision = uuid4().hex
            base = self.root / "versions" / name / revision
            base.mkdir(parents=True)
            (base / "skill.json").write_bytes(data)
            manifest = {"name": name, "revision": revision, "parent": parent, "sha256": digest(data),
                        "contract_hash": context["contract_hash"],
                        "tool_binding": {"tools": [contract["available_tools"][n] for n in sorted(required)]},
                        "created_at": datetime.now(timezone.utc).isoformat()}
            self.write(str(base.relative_to(self.root) / "manifest.json"), manifest)
            self.event("skill_proposed", **manifest)
        return manifest

    def checked(self, name, revision):
        manifest, doc = self.revision(name, revision)
        contract = self.read(f"contracts/{manifest['contract_hash']}.json")
        if not contract or digest(canonical(contract).encode()) != manifest["contract_hash"]:
            raise ValueError("Evidence contract changed")
        required = doc.get("requires_tools")
        if (not isinstance(required, list) or not all(isinstance(n, str) and n in contract["available_tools"] for n in required)
                or manifest["tool_binding"] != {"tools": [contract["available_tools"][n] for n in sorted(required)]}):
            raise ValueError("Changed dependencies")
        return manifest, doc, contract

    def activate(self, name, revision):
        with self.locked():
            manifest, _, _ = self.behavior_report(name, revision)
            index = self.index()
            entry = index.get(name, {"active": None, "accepted": []})
            if entry["active"] != manifest["parent"]:
                raise ValueError("Stale skill parent")
            if revision not in entry["accepted"]:
                entry["accepted"].append(revision)
            entry.update(active=revision, status="development_checked")
            index[name] = entry
            self.write("index.json", index)
            self.event("skill_activated", **manifest, status="development_checked")
        return {**manifest, "status": "development_checked", "effectiveness_verified": False}

    def retire(self, name):
        """Host-only; remove from retrieval, preserve source and accepted history."""
        identifier(name)
        with self.locked():
            index = self.index()
            if name not in index:
                raise ValueError("Unknown skill")
            previous = index[name]["active"]
            index[name].update(active=None, status="retired")
            self.write("index.json", index)
            self.event("skill_retired", name=name, previous=previous)

    def rollback(self, name, revision, available_tools):
        available = bindings_by_name(available_tools)
        with self.locked():
            index = self.index()
            if revision not in index.get(name, {}).get("accepted", []):
                raise ValueError("Rollback target was not accepted")
            manifest, _, _ = self.behavior_report(name, revision)
            if not self.compatible(manifest, available):
                raise ValueError("Rollback tool dependencies differ")
            previous = index[name]["active"]
            index[name].update(active=revision, status="development_checked")
            self.write("index.json", index)
            self.event("skill_rolled_back", name=name, revision=revision, previous=previous)

    @staticmethod
    def compatible(manifest, available):
        return all(available.get(b["name"]) == b for b in manifest["tool_binding"]["tools"])

    def search(self, query, tool_binding=None):
        # Legacy callers must not accidentally bypass dependency filtering.
        bindings = [] if tool_binding is None else tool_binding.get("tools", [tool_binding])
        return self.search_all(query, bindings)

    def search_all(self, query, available_tools):
        if not isinstance(query, str) or len(query) > 1000:
            raise ValueError("Invalid query")
        available = bindings_by_name(available_tools)
        matches, incompatible = [], []
        terms = set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", query.lower()))
        with self.locked():
            for name, entry in self.index().items():
                if entry.get("active") is None:
                    continue
                manifest, doc, _ = self.behavior_report(name, entry["active"])
                if not self.compatible(manifest, available):
                    incompatible.append({"name": name, "revision": manifest["revision"]})
                    continue
                score = sum(term in canonical(doc).lower() for term in terms)
                if score:
                    matches.append({**manifest, "skill": doc, "score": score, "status": "development_checked"})
            matches.sort(key=lambda x: (-x["score"], x["name"]))
            selected = matches[:3]
            self.event("skill_retrieved", query=query, revisions=[x["revision"] for x in selected], incompatible=incompatible)
        return {"matches": selected, "needs_revalidation": incompatible,
                "warning": "Development-checked advice; verify current task conditions. Benefit remains unproven."}
