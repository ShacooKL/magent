"""Task-only advisory Skills; mechanical validation, never a semantic truth gate."""
from pathlib import Path

from rsagent.runtime.jsonio import dump
from rsagent.evolution.tool_registry import canonical, safe_bytes, strict_json

FIELDS = {"title", "when", "guidance", "rationale", "limitations", "evidence_ids"}
MAX_SKILLS = 8
MAX_BYTES = 49152


def check_task_skills(document, evidence):
    errors = []
    if not isinstance(document, dict) or set(document) != {"skills"}:
        return {"success": False, "errors": ["Use exactly {skills: [...]}."]}
    cards = document["skills"]
    if not isinstance(cards, list) or len(cards) > MAX_SKILLS:
        return {"success": False, "errors": ["skills must contain 0..8 task Skill cards; no category quota."]}
    for i, card in enumerate(cards):
        prefix = f"skills[{i}]"
        if not isinstance(card, dict) or set(card) != FIELDS:
            errors.append(prefix + ": require exactly " + ", ".join(sorted(FIELDS)))
            continue
        for key in ("title", "when", "rationale"):
            if not isinstance(card[key], str) or not card[key].strip():
                errors.append(prefix + "." + key + ": nonempty string required")
        for key, maximum in (("guidance", 6), ("limitations", 4), ("evidence_ids", 12)):
            values = card[key]
            if not isinstance(values, list) or not 1 <= len(values) <= maximum or not all(
                    isinstance(v, str) and v.strip() for v in values):
                errors.append(prefix + f".{key}: require 1..{maximum} nonempty strings")
            elif key == "evidence_ids":
                errors.extend(prefix + ": unauthorized evidence " + eid for eid in values if eid not in evidence)
    return {"success": not errors, "errors": errors,
            "scope": "Format and evidence authorization only; utility is measured by downstream trials."}


def advisory_cards(document):
    """Separate reusable advice from historical observations and question-specific provenance."""
    return [{"id": f"task_skill_{i+1:03d}", **{key: card[key] for key in
             ("title", "when", "guidance", "limitations")}} for i, card in enumerate(document["skills"])]


class TaskSkillExtractionProvider:
    def __init__(self, workspace, output, evidence, recorder=None):
        self.workspace, self.output = Path(workspace).resolve(), Path(output).resolve()
        if self.output.is_relative_to(self.workspace) or self.workspace.is_relative_to(self.output):
            raise ValueError("Submission store must be outside workspace")
        self.evidence, self.recorder = evidence, recorder
        self.names = {"task_skill_check", "task_skill_submit"}
        self.definitions = [{"name": name, "description": description, "input_schema": {
            "type": "object", "properties": {"source": {"type": "string",
            "description": "Relative workspace JSON file, normally draft.json; not inline JSON."}},
            "required": ["source"], "additionalProperties": False}} for name, description in (
            ("task_skill_check", "Check 0..8 remote-sensing TASK Skills. Checks format/references, not truth. No tool-generation category or mandatory quota."),
            ("task_skill_submit", "Check and publish one task-only candidate bundle. Empty skills is valid. No automatic activation; future evaluation decides utility. Finish after success."))]

    def execute(self, name, arguments):
        if name not in self.names or not isinstance(arguments, dict) or set(arguments) != {"source"} or not isinstance(arguments["source"], str):
            raise ValueError("Expected task_skill_check/submit with source file")
        path = (self.workspace / arguments["source"]).resolve(strict=True)
        if not path.is_relative_to(self.workspace):
            raise ValueError("Source must remain inside workspace")
        document = strict_json(safe_bytes(path, MAX_BYTES).decode())
        result = check_task_skills(document, self.evidence)
        if name == "task_skill_submit" and result["success"]:
            if (self.output / "submission.json").exists():
                raise ValueError("Already submitted; refusing overwrite")
            dump(self.output / "submission.json", {"schema": "remote_sensing_task_skills_v1",
                 "document": document, "check": result, "status": "candidate_pending_behavior_evaluation",
                 "effectiveness_verified": False})
        if self.recorder:
            self.recorder.log(name, document=document, **result)
        return canonical(result)
