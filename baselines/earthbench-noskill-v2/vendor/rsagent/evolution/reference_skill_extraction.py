"""Lightweight two-category reference Skills. Structure/provenance checks, not truth certification."""
from pathlib import Path

from rsagent.runtime.jsonio import dump
from rsagent.evolution.reference_skill_context import check_plan
from rsagent.evolution.tool_registry import canonical, safe_bytes, strict_json

CATEGORIES = ("remote_sensing_knowledge", "tool_generation")
FIELDS = {"title", "when", "guidance", "rationale", "limitations", "evidence_ids"}
MAX_DRAFT = 32768


def library_ref(item):
    return f"library:{item['id']}:{item['revision']}"


def check_bundle(doc, evidence, catalog):
    errors = []
    decisions = doc.get("decisions") if isinstance(doc, dict) else None
    if not isinstance(decisions, list) or len(decisions) != 2:
        return {"success": False, "errors": ["decisions must contain exactly one decision per category (2 total); none is allowed"]}
    seen = set()
    available = {library_ref(v): v for v in catalog.values()}
    for i, decision in enumerate(decisions):
        prefix = f"decisions[{i}]"
        if not isinstance(decision, dict) or set(decision) != {"category", "action", "target_id", "reason", "skill"}:
            errors.append(prefix + ": fields must be category, action, target_id, reason, skill")
            continue
        category, action, target = decision["category"], decision["action"], decision["target_id"]
        if not isinstance(category, str) or category not in CATEGORIES or category in seen:
            errors.append(prefix + ".category: use each declared category exactly once")
        else:
            seen.add(category)
        if not isinstance(action, str) or action not in {"add", "update", "none"}:
            errors.append(prefix + ".action: add, update or none")
        if not isinstance(decision["reason"], str) or not decision["reason"].strip():
            errors.append(prefix + ".reason: explain value, change, or why nothing is worth retaining")
        if action == "none":
            if target is not None or decision["skill"] is not None:
                errors.append(prefix + ": none requires target_id=null and skill=null")
            continue
        if action == "update":
            if not isinstance(target, str) or target not in catalog or catalog[target]["category"] != category:
                errors.append(prefix + ".target_id: update must reference a current same-category catalog ID")
        elif target is not None:
            errors.append(prefix + ".target_id: add requires null; host assigns IDs")
        skill = decision["skill"]
        if not isinstance(skill, dict) or set(skill) != FIELDS:
            errors.append(prefix + ".skill: fields must be title, when, guidance, rationale, limitations, evidence_ids")
            continue
        for key in ("title", "when", "rationale"):
            if not isinstance(skill[key], str) or not skill[key].strip():
                errors.append(prefix + f".skill.{key}: nonempty string required")
        for key in ("guidance", "limitations", "evidence_ids"):
            value = skill[key]
            if not isinstance(value, list) or not 1 <= len(value) <= 12 or not all(isinstance(v, str) and v.strip() for v in value):
                errors.append(prefix + f".skill.{key}: 1..12 nonempty strings required")
        ids = skill["evidence_ids"]
        if isinstance(ids, list) and all(isinstance(v, str) for v in ids):
            for eid in ids:
                if eid not in evidence and eid not in available:
                    errors.append(prefix + ".skill.evidence_ids: unknown ID " + eid)
            if not any(eid in evidence for eid in ids):
                errors.append(prefix + ".skill.evidence_ids: cite at least one current authorized observation")
        if action == "add" and any(v["category"] == category and v["skill"] == skill for v in catalog.values()):
            errors.append(prefix + ": exact duplicate; choose none or a justified update")
    return {"success": not errors, "errors": errors,
            "scope": "Schema and authorized evidence identity only; applicability, semantic support and utility require review, not certified."}


class ReferenceExtractionProvider:
    def __init__(self, workspace, output, evidence, packet_id, catalog, recorder=None, *, require_plan=False):
        self.workspace, self.output = Path(workspace).resolve(), Path(output).resolve()
        self.evidence, self.packet_id, self.catalog, self.recorder = evidence, packet_id, catalog, recorder
        self.require_plan = require_plan
        if self.output.is_relative_to(self.workspace) or self.workspace.is_relative_to(self.output):
            raise ValueError("Protected extraction output must be outside workspace")
        path = {"type": "string", "description": "Workspace JSON FILE path, not inline JSON; at most 32768 UTF-8 bytes."}
        specifications = [
            ("reference_skill_catalog", "Read current experimental reference candidates. They are advice, not truth or mandatory instructions. IDs are host-assigned; use exact IDs only for updates.", {}, []),
            ("reference_skill_check", "Preflight a draft; returns precise field/ID errors, never certifies truth. Does not publish. Each category may choose none. Read the packet's schema.json for the full format.", {"source": path}, ["source"]),
            ("reference_skill_submit", "Check and save ONE final two-category extraction bundle outside workspace. Candidates remain unproven references, not production skills. You may submit both none. On success, finish; repeat submissions are rejected.", {"source": path}, ["source"]),
        ]
        if require_plan:
            path["description"] = ("Path to the candidate bundle (normally draft.json), NOT extraction_plan.json. "
                                   "Both operations automatically read extraction_plan.json alongside it; do not check that file separately.")
            specifications = [(n, d + (" This checks BOTH the candidate and companion extraction_plan.json automatically; "
                              "source must be the candidate bundle, never the plan." if n != "reference_skill_catalog" else ""), p, r)
                              for n, d, p, r in specifications]
        self.definitions = [{"name": n, "description": d, "input_schema": {"type": "object", "properties": p,
                            "required": r, "additionalProperties": False}} for n, d, p, r in specifications]
        self.names = {n for n, *_ in specifications}

    def execute(self, name, arguments):
        if name not in self.names or not isinstance(arguments, dict):
            raise ValueError("Unknown extraction operation or invalid arguments")
        if name == "reference_skill_catalog":
            if arguments:
                raise ValueError("Catalog takes no arguments")
            return canonical({"status": "experimental_reference_candidates", "skills": [dict(v, evidence_id=library_ref(v)) for v in self.catalog.values()]})
        if set(arguments) != {"source"} or not isinstance(arguments["source"], str):
            raise ValueError("source must be a workspace JSON file path")
        path = (self.workspace / arguments["source"]).resolve(strict=True)
        if not path.is_relative_to(self.workspace):
            raise ValueError("source must stay inside workspace")
        doc = strict_json(safe_bytes(path, MAX_DRAFT).decode())
        if self.require_plan and (path.name == "extraction_plan.json" or
                                  (isinstance(doc, dict) and "candidates" in doc and "decisions" not in doc)):
            result = {"success": False, "errors": [
                "source must point to the candidate bundle, normally draft.json. extraction_plan.json is read and checked automatically alongside it; do not submit/check the plan separately."]}
            if self.recorder:
                self.recorder.log("reference_skill_checked" if name.endswith("check") else "reference_skill_submitted",
                                  success=False, errors=result["errors"], packet=self.packet_id, document=doc)
            return canonical(result)
        result = check_bundle(doc, self.evidence, self.catalog)
        plan = None
        if self.require_plan and result["success"]:
            try:
                plan_path = (self.workspace / "extraction_plan.json").resolve(strict=True)
                if not plan_path.is_relative_to(self.workspace):
                    raise ValueError("plan must stay inside workspace")
                plan = strict_json(safe_bytes(plan_path, MAX_DRAFT).decode())
                errors = check_plan(plan, doc, self.evidence)
            except (OSError, ValueError) as error:
                errors = ["plan: " + str(error)]
            result["errors"].extend(errors)
            result["success"] = not result["errors"]
        if name == "reference_skill_submit" and result["success"]:
            if (self.output / "submission.json").exists():
                raise ValueError("Final extraction already submitted; no overwrite")
            links = {}
            prior = {library_ref(v): v for v in self.catalog.values()}
            for decision in doc["decisions"]:
                if decision["skill"] is None:
                    continue
                for eid in decision["skill"]["evidence_ids"]:
                    if eid in self.evidence:
                        entry = self.evidence[eid]
                        links[eid] = {"source": entry["source"], "packet": self.packet_id, "evidence_id": eid}
                    else:
                        links[eid] = {"kind": "prior_reference_candidate_not_truth", "revision": prior[eid]["revision"]}
            dump(self.output / "submission.json", {"schema": "reference_skill_extraction_v2", "document": doc,
                 "packet": self.packet_id, "evidence_links": links,
                 **({"extraction_plan": plan} if self.require_plan else {}),
                 "check": result, "status": "reference_candidates_pending_review"})
        if self.recorder:
            self.recorder.log("reference_skill_checked" if name.endswith("check") else "reference_skill_submitted",
                              success=result["success"], errors=result["errors"], packet=self.packet_id, document=doc,
                              **({"extraction_plan": plan} if self.require_plan else {}))
        return canonical(result)


def apply_submission(catalog, submitted, packet):
    updated = dict(catalog)
    for decision in submitted["document"]["decisions"]:
        if decision["action"] == "none":
            continue
        if decision["action"] == "update":
            identity = decision["target_id"]
            revision = catalog[identity]["revision"] + 1
        else:
            number = 1
            while f"skill_{number:04d}" in updated:
                number += 1
            identity, revision = f"skill_{number:04d}", 1
        updated[identity] = {"id": identity, "revision": revision, "parent": catalog[identity]["revision"] if identity in catalog else None,
            "category": decision["category"], "skill": decision["skill"], "packet": packet,
            "status": "reference_candidate", "provenance": submitted["evidence_links"], "effectiveness_verified": False}
    return updated
