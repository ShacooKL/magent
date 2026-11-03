"""Compact decision memory with a trusted, independently recorded behavior gate.

Evidence IDs establish provenance, not entailment. Behavior approval establishes
only that candidate-assisted agents passed the registered cases, not causality.
Legacy SkillRegistry remains available for its historical schema.
"""
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

from rsagent.evolution.skill_memory import SkillRegistry
from rsagent.evolution.tool_registry import canonical, digest, identifier, safe_bytes, strict_json

LIMIT = 5000


class DecisionSkillRegistry(SkillRegistry):
    document_fields = {"title", "rules", "unknowns"}

    def set_decision_contract(self, evidence, tool_binding, behavior_policy):
        if not isinstance(evidence, dict) or not evidence or any(
                not isinstance(k, str) or not isinstance(v, dict) or set(v) != {"text", "source", "sha256"}
                or not isinstance(v["text"], str) or digest(v["text"].encode()) != v["sha256"]
                for k, v in evidence.items()):
            raise ValueError("Need named, hashed evidence texts and source links")
        if set(tool_binding) != {"name", "revision", "sha256"}:
            raise ValueError("Need exact tool binding")
        if not isinstance(behavior_policy, dict) or not behavior_policy or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in behavior_policy.items()):
            raise ValueError("Need nonempty behavior case fingerprint map")
        contract = {"schema": "decision_skill_v2", "evidence": evidence, "tool_binding": tool_binding,
                    "behavior_policy": behavior_policy}
        if len(canonical(contract).encode()) > 512 * 1024:
            raise ValueError("Contract too large")
        with self.locked():
            old = self.read("contract.json")
            if old is not None and old != contract:
                raise ValueError("Contract is immutable")
            self.write("contract.json", contract)

    def revision(self, name, revision):
        identifier(name)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{32}", revision):
            raise ValueError("Invalid revision")
        base = f"versions/{name}/{revision}"
        manifest = self.read(base + "/manifest.json")
        data = safe_bytes(self.root / base / "skill.json", LIMIT)
        if not manifest or digest(data) != manifest["sha256"]:
            raise ValueError("Unknown or changed skill")
        return manifest, strict_json(data.decode())

    def propose(self, name, source, parent=None):
        identifier(name)
        data = safe_bytes(source, LIMIT)
        doc = strict_json(data.decode())
        if not isinstance(doc, dict):
            raise ValueError("Need JSON object")
        with self.locked():
            contract = self.read("contract.json")
            if not contract or contract.get("schema") != "decision_skill_v2":
                raise ValueError("Operator contract missing")
            if self.index().get(name, {}).get("active") != parent:
                raise ValueError("Parent must match active revision")
            revision = uuid4().hex
            base = self.root / "versions" / name / revision
            base.mkdir(parents=True)
            (base / "skill.json").write_bytes(data)
            manifest = {"name": name, "revision": revision, "parent": parent, "sha256": digest(data),
                        "tool_binding": contract["tool_binding"], "created_at": datetime.now(timezone.utc).isoformat(),
                        "contract_hash": digest(canonical(contract).encode())}
            self.write(str(base.relative_to(self.root) / "manifest.json"), manifest)
            self.event("skill_proposed", **manifest)
            return manifest

    def checked(self, name, revision):
        manifest, doc = self.revision(name, revision)
        contract = self.read("contract.json")
        if manifest["contract_hash"] != digest(canonical(contract).encode()):
            raise ValueError("Evidence or behavior contract changed")
        return manifest, doc, contract

    def validate(self, name, revision):
        with self.locked():
            manifest, doc, contract = self.checked(name, revision)
            checks = []
            def check(name, ok):
                checks.append({"name": name, "passed": bool(ok)})
            def string(value, size):
                return isinstance(value, str) and 1 <= len(value) <= size
            check("fields", set(doc) == self.document_fields)
            check("title_1_to_120_chars", string(doc.get("title"), 120))
            unknowns = doc.get("unknowns")
            check("unknowns_1_to_3_strings_180_chars", isinstance(unknowns, list) and 1 <= len(unknowns) <= 3 and all(string(s, 180) for s in unknowns))
            rules = doc.get("rules")
            shape = isinstance(rules, list) and 1 <= len(rules) <= 4
            check("rules_1_to_4", shape)
            refs = set()
            for i, rule in enumerate(rules if shape else []):
                valid = isinstance(rule, dict) and set(rule) == {"when", "do", "stop_if", "evidence_ids", "basis"}
                valid = valid and all(string(rule[k], n) for k, n in (("when", 160), ("do", 240), ("stop_if", 180)))
                valid = valid and rule["basis"] in ("observed", "hypothesis")
                ids = rule.get("evidence_ids") if isinstance(rule, dict) else None
                valid = valid and isinstance(ids, list) and 1 <= len(ids) <= 4 and all(isinstance(k, str) and k in contract["evidence"] for k in ids)
                check(f"rule_{i}_fields_limits_evidence_ids", valid)
                if valid:
                    refs.update(ids)
            report = {**manifest, "checks": checks, "success": all(c["passed"] for c in checks),
                      "evidence_links": {k: {"source": contract["evidence"][k]["source"], "sha256": contract["evidence"][k]["sha256"]} for k in sorted(refs)},
                      "status": "awaiting_independent_behavior", "semantic_entailment_verified": False,
                      "effectiveness_verified": False}
            self.write(f"validations/{name}/{revision}.json", report)
            self.event("skill_validated", **report)
            return report

    def evidence_report(self, name, revision):
        manifest, doc, contract = self.checked(name, revision)
        report = self.read(f"validations/{name}/{revision}.json", {})
        if not report.get("success") or any(report.get(k) != manifest[k] for k in ("sha256", "contract_hash")):
            raise ValueError("Current evidence validation required")
        return manifest, doc, contract

    def record_behavior(self, name, revision, cases):
        """Trusted orchestration ONLY. This method is never exposed to model tools.

Each record links a host-scored report and fingerprint to this exact candidate.
The operator must score actual outputs and keep these attestations private.
"""
        with self.locked():
            manifest, _, contract = self.evidence_report(name, revision)
            if not isinstance(cases, dict) or set(cases) != set(contract["behavior_policy"]):
                raise ValueError("Every registered behavior case is required")
            for key, case in cases.items():
                if (not isinstance(case, dict) or case.get("fingerprint") != contract["behavior_policy"][key] or
                    case.get("skill_sha256") != manifest["sha256"] or case.get("retrieved_revision") != revision or
                    case.get("tool_binding") != manifest["tool_binding"] or type(case.get("passed")) is not bool or
                    not isinstance(case.get("report_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", case["report_sha256"])):
                    raise ValueError("Behavior attestation does not match contract/candidate")
            report = {"revision": revision, "sha256": manifest["sha256"], "contract_hash": manifest["contract_hash"],
                      "cases": cases, "success": all(c["passed"] for c in cases.values()),
                      "scope": "candidate-assisted Agent behavior on registered development cases only; not causal benefit"}
            self.write(f"behavior/{name}/{revision}.json", report)
            self.event("skill_behavior_checked", **report)
            return report

    def behavior_report(self, name, revision):
        manifest, doc, contract = self.evidence_report(name, revision)
        report = self.read(f"behavior/{name}/{revision}.json", {})
        if not report.get("success") or any(report.get(k) != manifest[k] for k in ("sha256", "contract_hash")):
            raise ValueError("Independent behavior approval required")
        return manifest, doc, contract

    def activate(self, name, revision):
        with self.locked():
            manifest, _, _ = self.behavior_report(name, revision)
            index = self.index()
            if index.get(name, {}).get("active") != manifest["parent"]:
                raise ValueError("Stale parent")
            index[name] = {"active": revision, "status": "development_checked"}
            self.write("index.json", index)
            self.event("skill_activated", **manifest, status="development_checked")
            return {**manifest, "status": "development_checked", "effectiveness_verified": False}

    def preview(self, name, revision, tool_binding):
        """Trusted behavior sessions only; never an Agent-selected revision."""
        manifest, doc, _ = self.evidence_report(name, revision)
        if tool_binding != manifest["tool_binding"]:
            return {"matches": []}
        self.event("skill_retrieved", query="behavior-preview", revisions=[revision], preview=True)
        return {"matches": [{**manifest, "skill": doc, "status": "unadmitted_behavior_candidate"}],
                "warning": "Candidate advice under test, not authority. Follow the task and inspect inputs."}

    def search(self, query, tool_binding=None):
        if not isinstance(query, str) or len(query) > 1000:
            raise ValueError("Invalid query")
        found = []
        with self.locked():
            for name, entry in self.index().items():
                manifest, doc, _ = self.behavior_report(name, entry["active"])
                if tool_binding is not None and tool_binding != manifest["tool_binding"]:
                    continue
                found.append({**manifest, "skill": doc, "status": "development_checked",
                              "score": sum(t in canonical(doc).lower() for t in query.lower().split())})
        found.sort(key=lambda x: (-x["score"], x["name"]))
        self.event("skill_retrieved", query=query, revisions=[x["revision"] for x in found[:1]])
        return {"matches": found[:1], "warning": "Development-checked advice, not a guarantee or a task override. Causal benefit unverified."}
