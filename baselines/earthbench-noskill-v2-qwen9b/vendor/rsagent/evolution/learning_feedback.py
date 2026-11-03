"""Return mechanical candidate rejection to the same bounded learning session."""
from copy import deepcopy
from pathlib import Path

from rsagent.evolution.joint_update import JointUpdateProvider
from rsagent.evolution.tool_registry import canonical


class LearningUpdateProvider:
    def __init__(self, workspace, output, pool, registry, parent, recorder=None, *,
                 allowed_updates=("tool", "skill"), max_operations=2, readonly=()):
        self.workspace, self.output = Path(workspace), Path(output)
        self.pool, self.registry, self.parent, self.recorder = pool, registry, parent, recorder
        self.allowed_updates, self.max_operations, self.readonly = allowed_updates, max_operations, readonly
        self.contrasts, self.attempts = [], []
        self.complete, self.publication, self.candidate_dir = False, None, None
        self.definitions = deepcopy(self._provider().definitions)
        for definition in self.definitions:
            if definition["name"] == "submit_update":
                definition["description"] = (
                    "Submit a candidate bundle after save_contrast; at most "
                    f"{max_operations} tool operations and {max_operations} Skill operations. "
                    "Tool upserts need name, source file path and input/output description. "
                    "First use write_file to save a workspace .py file defining main(arguments) -> JSON; "
                    "submit its path as source (e.g. tools/candidate.py), not code. "
                    "Skill upserts need name, when and guidance. operation defaults to upsert; "
                    "retire only needs name and operation. The host checks interfaces in a sandbox. "
                    "Rejection preserves the draft and lets you revise in THIS session's remaining "
                    "budget. Success ends learning and retains a trial, not verified task benefit.")
                if registry.require_source_validation:
                    definition["description"] += (
                        " Skill updates remain pending, unavailable to ordinary solving, until separate "
                        "host-scored source-task replays pass. Submission alone does not activate them.")
        self.names = {definition["name"] for definition in self.definitions}

    def _provider(self):
        provider = JointUpdateProvider(self.workspace, self.output / f"attempt-{len(self.attempts) + 1}",
                                       self.pool, self.recorder, allowed_updates=self.allowed_updates)
        provider.contrasts = self.contrasts
        return provider

    def execute(self, name, arguments):
        if name not in self.names or self.complete:
            raise ValueError("Unknown operation or learning session already finished")
        if name == "save_contrast":
            return self._provider().execute(name, arguments)
        for kind in ("tools", "skills"):
            if isinstance(arguments.get(kind), list) and len(arguments[kind]) > self.max_operations:
                return canonical({"success": False, "session_complete": False,
                                  "error": f"Retain at most {self.max_operations} {kind} operations"})
        provider = self._provider()
        result = provider.execute(name, arguments)
        if not provider.complete:
            return result
        candidate = provider.output / "candidate"
        # Publication and all rejection artifacts stay private. Only concise
        # execution feedback is returned; no independent expected values leak.
        publication = self.registry.publish(candidate, parent=self.parent, readonly=self.readonly)
        self.attempts.append({"candidate": str(candidate), "publication": publication})
        if self.recorder:
            self.recorder.log("learning_candidate_checked", attempt=len(self.attempts), publication=publication)
        if publication["status"] == "rejected":
            feedback = {key: {"interface_ok": value["interface"]["success"],
                               **{field: value["interface"].get(field)
                                  for field in ("stdout", "error", "exit_code", "timed_out")}}
                        for key, value in publication["tools"].items()}
            return canonical({"success": False, "session_complete": False,
                              "errors": publication["errors"], "tool_feedback": feedback,
                              "next": "Revise code, dependencies or comparison within the remaining request budget."})
        self.complete, self.publication, self.candidate_dir = True, publication, candidate
        return canonical({"success": True, "session_complete": True,
                          "status": publication["status"], "retained": publication.get("retained", False),
                          "effectiveness_verified": False})
