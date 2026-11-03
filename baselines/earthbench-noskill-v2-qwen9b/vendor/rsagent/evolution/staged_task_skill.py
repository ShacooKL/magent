"""Persist diagnosis before advice; checks are structural, never scientific verdicts."""
import json
import re

from rsagent.evolution.failure_task_skill import FailureTaskSkillExtractionProvider


DIAGNOSIS_FIELDS = {"id", "episode_id", "task_requirement", "definition_evidence_id", "definition_clause",
    "call_evidence_id", "actual_operation", "alignment", "alignment_explanation", "final_link", "alternative", "evidence_ids"}
CHECK_FIELDS = {"skill_index", "diagnosis_ids", "proposed_change", "consistency", "explanation"}


class DiagnosisProvider(FailureTaskSkillExtractionProvider):
    """Use the existing check/submit transport for a separate diagnosis-only session."""
    submission_schema = "task_diagnoses_v1"
    def __init__(self, *args, max_records=3, **kwargs):
        if type(max_records) is not int or not 1 <= max_records <= 3:
            raise ValueError("max_records must be an integer in 1..3")
        self.max_records = max_records
        super().__init__(*args, **kwargs)
        for definition in self.definitions:
            definition["description"] = (
                "Check/publish a DIAGNOSIS document, not Skills in this stage. Use {diagnoses:[...]}, "
                f"0..{max_records}. "
                "Fields and enums are in the stage prompt. Require one complete task/call/result, its ending "
                "and matching original tool definition fully read via trajectory tools per diagnosis. "
                "Checks structure, references and copied clause only; never checks scientific truth. "
                "Successful task_skill_submit persists this stage and ends this session.")

    def _check(self, document):
        errors = []
        rows = document.get("diagnoses") if isinstance(document, dict) else None
        if not isinstance(document, dict) or set(document) != {"diagnoses"} or not isinstance(rows, list) or len(rows) > self.max_records:
            return {"success": False, "errors": ["Use exactly {diagnoses: [...]} with at most " + str(self.max_records) + " records."]}
        ids = set()
        cards = []
        card_indices = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != DIAGNOSIS_FIELDS:
                errors.append(f"diagnoses[{i}]: exact fields required: {sorted(DIAGNOSIS_FIELDS)}")
                continue
            if any(not isinstance(row[k], str) or not row[k].strip() for k in DIAGNOSIS_FIELDS - {"evidence_ids"}):
                errors.append(f"diagnoses[{i}]: text fields must be nonempty strings")
                continue
            if row["id"] in ids:
                errors.append("Diagnosis IDs must be unique")
            ids.add(row["id"])
            if row["alignment"] not in {"consistent", "inconsistent", "unresolved"}:
                errors.append("alignment must be consistent/inconsistent/unresolved")
            ep = self._episodes.get(row["episode_id"])
            actions = [a for n in self.trajectory["nodes"] if n["episode_id"] == row["episode_id"]
                       for a in n["actions"] if a["call_evidence_id"] == row["call_evidence_id"]]
            if ep is None or row["episode_id"] not in self.failure_episode_ids or not actions:
                errors.append(f"diagnoses[{i}]: choose an authorized failure and an action belonging to it")
                continue
            action = actions[0]
            definition_id = row["definition_evidence_id"]
            if definition_id != "tool_definition:" + action["name"] or definition_id not in self.evidence:
                errors.append(f"diagnoses[{i}]: definition must match the chosen action's tool")
                continue
            definition = json.loads(self.evidence[definition_id]["text"])
            if row["definition_clause"] not in definition.get("description", ""):
                errors.append(f"diagnoses[{i}]: copy a short exact clause from the original description; do not paraphrase it")
            refs = row["evidence_ids"]
            required = [ep["task_evidence_id"], ep["ending_evidence_id"], action["call_evidence_id"],
                        action["result_evidence_id"], definition_id]
            if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
                errors.append(f"diagnoses[{i}]: evidence_ids must be a list of strings")
                continue
            for eid in required:
                if eid not in refs or eid not in self.evidence or not self._fully_read(eid):
                    errors.append(f"diagnoses[{i}]: fully read via trajectory tools and cite {eid}")
            cards.append({"title": row["id"], "when": row["task_requirement"], "guidance": [row["alternative"]],
                "rationale": row["alignment_explanation"], "limitations": ["Diagnosis is provisional"], "evidence_ids": refs})
            card_indices.append(i)
        grounded = super()._check({"skills": cards})
        # Some structurally invalid records never become pseudo-cards. Preserve
        # original diagnosis indices in repair feedback instead of compacted indices.
        errors.extend(re.sub(r"^skills\[(\d+)\]", lambda m: f"diagnoses[{card_indices[int(m[1])]}]", message)
                      for message in grounded["errors"])
        for card_audit in grounded["audit"]["cards"]:
            card_audit["index"] = card_indices[card_audit["index"]]
        return {"success": not errors, "errors": errors, "audit": grounded["audit"],
                "artifact_kind": "diagnosis", "scope": "Structure/read coverage/quoted text only, not correct diagnosis."}


class ReflectiveSkillProvider(FailureTaskSkillExtractionProvider):
    submission_schema = "reflective_task_skills_v1"
    def __init__(self, *args, diagnoses, max_cards=3, **kwargs):
        if type(max_cards) is not int or not 1 <= max_cards <= 3:
            raise ValueError("max_cards must be an integer in 1..3")
        self.max_cards = max_cards
        super().__init__(*args, **kwargs)
        self.diagnosis_ids = {d["id"] for d in diagnoses["diagnoses"]}
        for definition in self.definitions:
            definition["description"] = definition["description"].replace("0..3", f"0..{max_cards}").replace(
                "At most three cards.", f"At most {max_cards} cards.")
            definition["description"] += (
                " This stage requires exactly {skills:[...],self_checks:[...]}. One self_check per Skill: "
                "skill_index (zero-based), diagnosis_ids, proposed_change, consistency (consistent/unresolved), "
                "explanation. Diagnose any contradiction yourself, revise or abstain; host checks only fields/references.")

    def _check(self, document):
        if not isinstance(document, dict) or set(document) != {"skills", "self_checks"}:
            return {"success": False, "errors": ["Use exactly {skills:[...],self_checks:[...]}."]}
        result = super()._check({"skills": document["skills"]})
        if isinstance(document["skills"], list) and len(document["skills"]) > self.max_cards:
            result["errors"].append(f"Keep at most {self.max_cards} Skill cards; empty is valid")
        checks = document["self_checks"]
        if not isinstance(checks, list) or not isinstance(document["skills"], list) or len(checks) != len(document["skills"]):
            result["errors"].append("Require exactly one self_check per Skill; empty skills uses empty self_checks")
        else:
            seen = set()
            for check in checks:
                if not isinstance(check, dict) or set(check) != CHECK_FIELDS:
                    result["errors"].append("self_check fields must be: " + ", ".join(sorted(CHECK_FIELDS)))
                    continue
                index = check["skill_index"]
                if type(index) is not int or not 0 <= index < len(checks) or index in seen:
                    result["errors"].append("skill_index must cover each card exactly once")
                else:
                    seen.add(index)
                refs = check["diagnosis_ids"]
                if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or x not in self.diagnosis_ids for x in refs):
                    result["errors"].append("Each self_check must reference saved diagnosis IDs")
                if check["consistency"] not in ("consistent", "unresolved"):
                    result["errors"].append("consistency must be consistent/unresolved; this is your assessment, not certification")
                for field in ("proposed_change", "explanation"):
                    if not isinstance(check[field], str) or not check[field].strip():
                        result["errors"].append(field + " must be a nonempty string")
        result["success"] = not result["errors"]
        result["artifact_kind"] = "skills_with_self_checks"
        return result
