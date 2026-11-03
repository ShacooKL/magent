"""Persist a diagnosis, then generate advisory Skills without resetting evidence/context."""
import json

from rsagent.evolution.staged_task_skill import DiagnosisProvider, ReflectiveSkillProvider
from rsagent.evolution.tool_registry import canonical


class ContinuousTaskSkillProvider:
    def __init__(self, workspace, diagnosis_output, skill_output, trajectory, failure_episode_ids,
                 skill_prompt, recorder=None):
        self.workspace, self.skill_output = workspace, skill_output
        self.trajectory, self.recorder = trajectory, recorder
        self.skill_prompt = skill_prompt
        self.stage, self.complete = "diagnosis", False
        self.read_ranges = []
        self.provider = DiagnosisProvider(workspace, diagnosis_output, trajectory, failure_episode_ids,
                                          recorder, max_records=1)
        self.names = set(self.provider.names)
        # Stable catalog across the checkpoint: phase-specific instructions arrive as a tool result.
        self.definitions = [{**d, "description": (
            "Check/save the current phase document from a workspace JSON file. First phase: diagnoses, "
            "maximum one record. After successful nonempty diagnosis submission the tool returns "
            "session_complete=false and Skill-phase instructions; keep the same conversation. "
            "Then submit skills and self_checks, maximum one card. Only final Skill submission or "
            "empty diagnosis ends the session. Format/evidence checks do not certify correctness.")}
            for d in self.provider.definitions]

    def note_evidence_read(self, evidence_id, offset=0, length=4000):
        self.provider.note_evidence_read(evidence_id, offset, length)
        self.read_ranges.append((evidence_id, offset, length))

    def inspection_progress(self):
        return self.provider.inspection_progress()

    def execute(self, name, arguments):
        if self.complete:
            raise ValueError("Final submission already saved")
        output = self.provider.execute(name, arguments)
        result = json.loads(output)
        if name != "task_skill_submit" or not result["success"]:
            return output
        if self.stage == "skills":
            self.complete = True
            return canonical({**result, "session_complete": True, "stage": "skills"})
        saved = json.loads((self.provider.output / "submission.json").read_text())
        diagnosis = saved["document"]
        if self.recorder:
            self.recorder.log("diagnosis_checkpoint", document=diagnosis,
                              read_ranges_inherited=len(self.read_ranges))
        if not diagnosis["diagnoses"]:
            self.complete = True
            return canonical({**result, "session_complete": True, "stage": "diagnosis",
                              "reason": "Empty diagnosis; no Skill generation"})
        selected = sorted({d["episode_id"] for d in diagnosis["diagnoses"]})
        self.provider = ReflectiveSkillProvider(self.workspace, self.skill_output, self.trajectory,
            selected, self.recorder, diagnoses=diagnosis, max_cards=1)
        for eid, offset, length in self.read_ranges:
            self.provider.note_evidence_read(eid, offset, length)
        self.stage = "skills"
        return canonical({**result, "session_complete": False, "stage": "skills",
            "saved_diagnosis": diagnosis, "next_instructions": self.skill_prompt,
            "evidence": "Previously retrieved evidence remains in this conversation and its read coverage is preserved."})
