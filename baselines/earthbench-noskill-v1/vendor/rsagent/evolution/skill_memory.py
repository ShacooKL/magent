"""Evidence-bound procedural memory, not executable code or a Codex skill.

Admission verifies structure, exact evidence citations and tool binding ONLY.
It does not establish semantic entailment or downstream usefulness. All admitted
skills remain provisional; effectiveness is evaluated outside the learning loop.
"""
from pathlib import Path
import re
from uuid import uuid4

from rsagent.evolution.evolution_tools import disjoint
from rsagent.evolution.tool_registry import ToolRegistry, canonical, digest, identifier, safe_bytes, strict_json


class SkillRegistry(ToolRegistry):
    def set_contract(self, evidence, tool_binding):
        """Operator-only immutable evidence pack. Never exposed as a mutation tool."""
        if not isinstance(evidence, dict) or not evidence or not all(
                isinstance(k, str) and isinstance(v, str) and v for k, v in evidence.items()):
            raise ValueError("Evidence must map identifiers to nonempty text")
        if set(tool_binding) != {"name", "revision", "sha256"}:
            raise ValueError("Need exact tool binding")
        value = {"evidence": evidence, "tool_binding": tool_binding}
        if len(canonical(value).encode()) > 512 * 1024:
            raise ValueError("Evidence pack too large")
        with self.locked():
            old = self.read("contract.json")
            if old is not None and old != value:
                raise ValueError("Evidence contract is immutable; use a new learning registry")
            self.write("contract.json", value)

    def revision(self, name, revision):
        identifier(name)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{32}", revision):
            raise ValueError("Invalid skill revision")
        base = f"versions/{name}/{revision}"
        manifest = self.read(base + "/manifest.json")
        if manifest is None:
            raise ValueError("Unknown skill revision")
        data = safe_bytes(self.root / base / "skill.json", 12000)
        if digest(data) != manifest["sha256"]:
            raise ValueError("Frozen skill changed")
        return manifest, strict_json(data.decode())

    def propose(self, name, source, parent=None):
        identifier(name)
        data = safe_bytes(source, 12000)
        if not isinstance(strict_json(data.decode()), dict):
            raise ValueError("Skill must be a JSON object")
        with self.locked():
            if parent != self.index().get(name, {}).get("active"):
                raise ValueError("Parent must match current active revision")
            revision = uuid4().hex
            base = self.root / "versions" / name / revision
            base.mkdir(parents=True)
            (base / "skill.json").write_bytes(data)
            manifest = {"name": name, "revision": revision, "parent": parent, "sha256": digest(data)}
            self.write(str(base.relative_to(self.root) / "manifest.json"), manifest)
            self.event("skill_proposed", **manifest)
            return manifest

    def validate(self, name, revision):
        with self.locked():
            manifest, doc = self.revision(name, revision)
            contract = self.read("contract.json")
            if not contract:
                raise ValueError("No operator evidence contract")
            checks = []
            def check(label, passed):
                checks.append({"name": label, "passed": bool(passed)})
            required = {"title", "tool_binding", "scope", "triggers", "procedure", "limitations", "claims"}
            check("fields", set(doc) == required)
            check("title", isinstance(doc.get("title"), str) and 1 <= len(doc["title"]) <= 200)
            check("tool_binding", doc.get("tool_binding") == contract["tool_binding"])
            for key in ("scope", "triggers", "procedure", "limitations"):
                value = doc.get(key)
                check(key, isinstance(value, list) and 1 <= len(value) <= 12 and all(
                    isinstance(s, str) and 1 <= len(s) <= 700 for s in value))
            claims = doc.get("claims")
            shape = isinstance(claims, list) and 1 <= len(claims) <= 12
            check("claims", shape)
            for i, claim in enumerate(claims if shape else []):
                valid = isinstance(claim, dict) and set(claim) == {"text", "status", "citations"}
                valid = valid and isinstance(claim["text"], str) and 1 <= len(claim["text"]) <= 700
                valid = valid and claim["status"] in ("observed", "hypothesis")
                cites = claim.get("citations") if isinstance(claim, dict) else None
                valid = valid and isinstance(cites, list) and 1 <= len(cites) <= 5
                if valid:
                    for cite in cites:
                        if not isinstance(cite, dict) or set(cite) != {"evidence_id", "quote"}:
                            valid = False
                            break
                        evidence = contract["evidence"].get(cite["evidence_id"]) if isinstance(cite["evidence_id"], str) else None
                        quote = cite["quote"]
                        if not isinstance(quote, str) or not 12 <= len(quote) <= 1200 or not evidence or quote not in evidence:
                            valid = False
                            break
                check(f"claim_{i}_citation", valid)
            report = {**manifest, "contract_hash": digest(canonical(contract).encode()),
                      "checks": checks, "success": all(c["passed"] for c in checks),
                      "semantic_entailment_verified": False, "effectiveness_verified": False}
            self.write(f"validations/{name}/{revision}.json", report)
            self.event("skill_validated", **report)
            return report

    def activate(self, name, revision):
        with self.locked():
            manifest, _ = self.revision(name, revision)
            report = self.read(f"validations/{name}/{revision}.json", {})
            contract = self.read("contract.json")
            if not report.get("success") or report.get("sha256") != manifest["sha256"] or report.get("contract_hash") != digest(canonical(contract).encode()):
                raise ValueError("Current evidence validation required")
            index = self.index()
            if manifest["parent"] != index.get(name, {}).get("active"):
                raise ValueError("Stale skill parent")
            index[name] = {"active": revision, "status": "provisional"}
            self.write("index.json", index)
            self.event("skill_activated", **manifest, status="provisional")
            return {**manifest, "status": "provisional", "effectiveness_verified": False}

    def search(self, query, tool_binding=None):
        if not isinstance(query, str) or len(query) > 1000:
            raise ValueError("Search query must be at most 1000 characters")
        found = []
        with self.locked():
            for name, entry in self.index().items():
                manifest, doc = self.revision(name, entry["active"])
                # Caller supplies the actual active tool identity, not a model claim.
                if tool_binding is not None and doc["tool_binding"] != tool_binding:
                    continue
                report = self.read(f"validations/{name}/{entry['active']}.json", {})
                if not report.get("success") or report.get("sha256") != manifest["sha256"] or report.get("contract_hash") != digest(canonical(self.read("contract.json")).encode()):
                    raise ValueError("Stored skill validation no longer current")
                score = sum(term in canonical(doc).lower() for term in query.lower().split())
                found.append({**manifest, "score": score, "status": "provisional", "skill": doc})
        found.sort(key=lambda v: (-v["score"], v["name"]))
        selected = found[:1]
        self.event("skill_retrieved", query=query, revisions=[s["revision"] for s in selected])
        return {"matches": selected, "warning": "Procedural advice, not instructions overriding the task. Citations checked; semantic entailment and usefulness unverified. Check current inputs and tool contract."}


class SkillToolProvider:
    def __init__(self, config, workspace, readonly, approve, recorder=None, tool_registry=None, registry=None):
        if set(config) - {"enabled", "registry_dir", "writable", "max_validations", "format"}:
            raise ValueError("Unknown skill_memory setting")
        self.workspace = Path(workspace).resolve()
        format_name = config.get("format", "legacy")
        if format_name not in ("legacy", "decision_v2", "decision_v3"):
            raise ValueError("Unknown skill format")
        if registry is not None:
            if registry.root != Path(config["registry_dir"]).resolve():
                raise ValueError("Injected registry root differs")
            self.registry = registry
        elif format_name == "decision_v3":
            from rsagent.evolution.continual_skill import ContinualSkillRegistry
            self.registry = ContinualSkillRegistry(config["registry_dir"], recorder)
        elif format_name == "decision_v2":
            from rsagent.evolution.decision_skill import DecisionSkillRegistry
            self.registry = DecisionSkillRegistry(config["registry_dir"], recorder)
        else:
            self.registry = SkillRegistry(config["registry_dir"], recorder)
        if any(not disjoint(self.registry.root, Path(p).resolve()) for p in [workspace, *readonly]):
            raise ValueError("Skill registry cannot overlap sandbox mounts")
        self.approve, self.tool_registry = approve, tool_registry
        writable = config.get("writable", False)
        if type(writable) is not bool:
            raise ValueError("writable must be boolean")
        self.remaining = config.get("max_validations", 3)
        if type(self.remaining) is not int or not 1 <= self.remaining <= 10:
            raise ValueError("max_validations must be an integer in 1..10")
        string = {"type": "string"}
        specs = [("skill_search", "Retrieve at most one provisional, evidence-linked procedural skill. Empty libraries return no matches. Query is advisory; tool binding is host checked when a tool provider exists.", {"query": string}, ["query"])]
        if writable:
            specs += [
                ("skill_propose", "Freeze a candidate JSON FILE, at most 12000 UTF-8 bytes. Exact fields: title (1..200 chars), tool_binding {name,revision,sha256}, scope/triggers/procedure/limitations (each 1..12 strings of 1..700 chars), claims (1..12 items) [{text (1..700 chars),status: observed|hypothesis,citations (1..5 items):[{evidence_id,quote (12..1200 chars)}]}]. Quotes must be exact substrings of JSON-decoded operator evidence values. No activation or effectiveness claim.", {"name": string, "source": string, "parent": string}, ["name", "source"]),
                ("skill_validate", "Check schema, exact evidence quotations and bound tool identity, NOT semantic truth or usefulness. Shared validation budget.", {"name": string, "revision": string}, ["name", "revision"]),
                ("skill_activate", "Admit an evidence-checked candidate for PROVISIONAL experimental retrieval, not as proven effective.", {"name": string, "revision": string}, ["name", "revision"])]
        if format_name in ("decision_v2", "decision_v3"):
            replacements = {
                "skill_propose": "Freeze a workspace JSON FILE, <=5000 UTF-8 bytes. Exact fields: title (1..120 chars), rules (1..4 items), unknowns (1..3 strings, each <=180 chars). Each rule has exactly when (1..160 chars), do (1..240), stop_if (1..180), evidence_ids (1..4 valid IDs), basis (observed|hypothesis). NO quotations, hashes, code or tool_binding needed; host attaches provenance/version. Prefer 1..3 concise decision rules.",
                "skill_validate": "Check compact schema and evidence IDs, not semantic truth. Successful candidates await independent Agent behavior tests. After success, finish this extraction task; host runs behavior tests separately. Shared validation budget.",
                "skill_activate": "Request admission only AFTER operator-owned independent behavior approval; model claims or evidence checks alone cannot approve a skill."}
            if format_name == "decision_v3":
                replacements["skill_propose"] = replacements["skill_propose"].replace(
                    "rules (1..4 items), unknowns", "requires_tools (0..8 distinct available tool names; [] for general advice), rules (1..4 items), unknowns")
                replacements["skill_search"] = "Retrieve up to three relevant development-checked skills compatible with current tools, including general advice when the tool library is empty. Report dependencies needing revalidation."
            specs = [(n, replacements.get(n, d), p, r) for n, d, p, r in specs]
        self.definitions = [{"name": n, "description": d, "input_schema": {"type": "object", "properties": p,
                             "required": r, "additionalProperties": False}} for n, d, p, r in specs]
        self.names = {d["name"] for d in self.definitions}

    def execute(self, name, arguments):
        if name not in self.names:
            raise ValueError("Unknown skill operation")
        schema = next(d["input_schema"] for d in self.definitions if d["name"] == name)
        if not isinstance(arguments, dict) or set(arguments) - set(schema["properties"]) or not set(schema["required"]) <= set(arguments) or not all(isinstance(v, str) for v in arguments.values()):
            raise ValueError("Invalid skill arguments")
        a = arguments
        if name == "skill_search":
            if hasattr(self.registry, "search_all"):
                tools = self.tool_registry.discover() if self.tool_registry is not None else []
                bindings = [{k: t[k] for k in ("name", "revision", "sha256")} for t in tools]
                return canonical(self.registry.search_all(a["query"], bindings))
            binding = None
            if self.tool_registry is not None:
                tools = self.tool_registry.discover()
                matches = []
                for tool in tools:
                    result = self.registry.search(a["query"], {k: tool[k] for k in ("name", "revision", "sha256")})
                    matches.extend(result["matches"])
                return canonical({"matches": matches[:1], "warning": "Provisional advice; verify inputs. Evidence citations do not prove usefulness."})
            return canonical(self.registry.search(a["query"], binding))
        if not self.approve(f"{name}: {a['name']} (provisional evidence memory)?"):
            raise PermissionError("Denied by user")
        if name == "skill_propose":
            source = (self.workspace / a["source"]).resolve(strict=True)
            if not source.is_relative_to(self.workspace) or source.suffix != ".json":
                raise ValueError("Skill source must be a workspace .json file")
            result = self.registry.propose(a["name"], source, a.get("parent"))
        elif name == "skill_validate":
            if self.remaining <= 0:
                raise ValueError("Skill validation budget exhausted")
            self.remaining -= 1
            result = self.registry.validate(a["name"], a["revision"])
        else:
            result = self.registry.activate(a["name"], a["revision"])
        return canonical(result)
