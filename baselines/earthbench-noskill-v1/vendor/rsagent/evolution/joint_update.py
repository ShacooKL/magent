"""Continuous, evidence-backed Tool/Skill proposals; no admission or utility verdict.

The pool owns authorized evidence, the runtime owns the request budget, and this
provider records the learner's choices. Candidate code is copied, never imported.
"""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from rsagent.evolution.tool_registry import canonical, identifier, safe_bytes


TOOL_SOURCE_CONTRACT = (
    "source must be an existing .py file path inside the task workspace, not code. "
    "First use write_file to save e.g. tools/candidate.py defining main(arguments) -> JSON, "
    "then submit source='tools/candidate.py'."
)


class JointUpdateProvider:
    def __init__(self, workspace, output, pool, recorder=None, *,
                 allowed_updates=("tool", "skill")):
        self.workspace = Path(workspace).resolve()
        self.output = Path(output).resolve()
        if self.output.is_relative_to(self.workspace) or self.workspace.is_relative_to(self.output):
            raise ValueError("Candidate storage must be outside the task workspace")
        self.pool, self.recorder = pool, recorder
        self.allowed_updates = set(allowed_updates)
        if not self.allowed_updates <= {"tool", "skill"}:
            raise ValueError("allowed_updates may contain tool and skill")
        self.contrasts, self.complete = [], False
        self.definitions = self._definitions()
        self.names = {d["name"] for d in self.definitions}

    @staticmethod
    def _definitions():
        text = {"type": "string"}
        refs = {"type": "array", "items": text,
                "description": "evidence_ref values returned by rollout_read; partial reads are allowed."}
        tool = {"type": "object", "properties": {
            "name": {"type": "string", "description": "Trial tool identifier. Initial tools stay frozen; "
                     "use a distinct name for a new or modified capability. Existing trial tools may be updated."},
            "operation": {"type": "string", "enum": ["upsert", "retire"]},
            "source": {"type": "string", "description": TOOL_SOURCE_CONTRACT}, "description": text,
        }, "required": ["name"], "additionalProperties": False}
        skill = {"type": "object", "properties": {
            "name": text, "operation": {"type": "string", "enum": ["upsert", "retire"]},
            "when": text, "guidance": text, "limitations": text,
            "requires_tools": {"type": "array", "items": text},
        }, "required": ["name"], "additionalProperties": False}
        specs = [
            ("save_contrast",
             "Save or revise your provisional comparison. Use read evidence from any authorized "
             "rollouts of this task. No exact quotes, full-trace reading or success/failure quota. "
             "This checkpoint does not end the session; you may explore code and revise it again.",
             {"evidence_refs": refs, "observation": text, "explanation": text,
              "uncertainty": text}, ["evidence_refs", "observation", "explanation"]),
            ("submit_update",
             "Submit ONE final candidate bundle after save_contrast. tools/skills may be empty. "
             "Operation defaults to upsert; first write_file a workspace .py file defining "
             "main(arguments) -> JSON, then submit its path as source (not code) and an "
             "input/output description; skill upserts need when and "
             "guidance. retire needs only a name and operation. Latest comparison evidence is "
             "inherited. Code is archived and syntax checked, NOT executed or activated. "
             "A successful submission ends this learning session, not a correctness assessment.",
             {"reason": text, "tools": {"type": "array", "items": tool},
              "skills": {"type": "array", "items": skill}, "evidence_refs": refs}, ["reason"]),
            ("finish_without_update", "Finish with no retained candidate; explain why. "
             "Valid even without a saved comparison.", {"reason": text}, ["reason"]),
        ]
        return [{"name": name, "description": description, "input_schema": {
            "type": "object", "properties": properties, "required": required,
            "additionalProperties": False,
        }} for name, description, properties, required in specs]

    @staticmethod
    def _text(value, field):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} needs nonempty text")
        return value

    def _record(self, event, **fields):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical({"type": event, **fields}) + "\n")
        if self.recorder:
            self.recorder.log(event, **fields)

    def save_contrast(self, evidence_refs, observation, explanation, uncertainty=""):
        if not evidence_refs:
            raise ValueError("Read relevant evidence with rollout_read before saving a comparison")
        evidence = self.pool.resolve_refs(evidence_refs)
        record = {"revision": len(self.contrasts) + 1, "task_id": self.pool.task_id,
                  "observation": self._text(observation, "observation"),
                  "explanation": self._text(explanation, "explanation"),
                  "uncertainty": uncertainty,
                  "evidence_refs": list(dict.fromkeys(evidence_refs)), "evidence": evidence}
        self._record("contrast_saved", comparison=record)
        self.contrasts.append(record)
        return {"success": True, "session_complete": False, "revision": record["revision"],
                "next": "Continue exploring, revise this comparison, submit an update, or finish without one."}

    def _prepare_changes(self, changes, kind):
        if not isinstance(changes, list):
            raise ValueError(f"{kind} changes must be a list")
        if changes and kind not in self.allowed_updates:
            raise ValueError(f"This learning session cannot retain {kind} updates")
        prepared, sources, seen = [], {}, set()
        for change in changes:
            name = identifier(change["name"])
            if name in seen:
                raise ValueError(f"Submit one final {kind} change for {name}")
            seen.add(name)
            operation = change.get("operation", "upsert")
            if operation not in {"upsert", "retire"}:
                raise ValueError("operation must be upsert or retire")
            item = {"name": name, "operation": operation}
            if operation == "upsert" and kind == "tool":
                supplied = change.get("source")
                if not isinstance(supplied, str) or "\n" in supplied or "\r" in supplied:
                    raise ValueError(TOOL_SOURCE_CONTRACT)
                source = Path(supplied)
                if source.suffix != ".py":
                    raise ValueError(TOOL_SOURCE_CONTRACT)
                if source.is_relative_to("/workspace"):
                    source = source.relative_to("/workspace")
                try:
                    source = (self.workspace / source).resolve(strict=True)
                    if not source.is_relative_to(self.workspace):
                        raise ValueError(TOOL_SOURCE_CONTRACT)
                    code = safe_bytes(source)
                except OSError:
                    raise ValueError(TOOL_SOURCE_CONTRACT) from None
                compile(code, str(source.relative_to(self.workspace)), "exec")
                item.update(source=f"tools/{name}.py",
                            workspace_source=str(source.relative_to(self.workspace)),
                            description=self._text(change["description"], "tool description"))
                sources[name] = code
            elif operation == "upsert":
                dependencies = change.get("requires_tools", [])
                if not isinstance(dependencies, list) or not all(isinstance(n, str) for n in dependencies):
                    raise ValueError("requires_tools must be a list of tool names")
                item.update(when=self._text(change["when"], "skill when"),
                            guidance=self._text(change["guidance"], "skill guidance"),
                            limitations=change.get("limitations", ""),
                            requires_tools=list(dict.fromkeys(dependencies)))
            prepared.append(item)
        return prepared, sources

    def submit_update(self, reason, tools=None, skills=None, evidence_refs=None):
        reason = self._text(reason, "reason")
        tools, sources = self._prepare_changes([] if tools is None else tools, "tool")
        skills, _ = self._prepare_changes([] if skills is None else skills, "skill")
        if (tools or skills) and not self.contrasts:
            raise ValueError("Save a comparison before retaining an update")
        inherited = self.contrasts[-1]["evidence_refs"] if self.contrasts else []
        refs = list(dict.fromkeys([*inherited, *(evidence_refs or [])]))
        evidence = self.pool.resolve_refs(refs)
        action = "both" if tools and skills else "tool" if tools else "skill" if skills else "none"
        proposal = {"schema": "joint_update_v1", "task_id": self.pool.task_id,
                    "base_snapshot": self.pool.capability_snapshot,
                    "sampling_config_id": self.pool.sampling_config_id,
                    "action": action, "reason": reason, "tools": tools, "skills": skills,
                    "comparisons": deepcopy(self.contrasts), "evidence_refs": refs, "evidence": evidence,
                    "status": "no_update" if action == "none" else "candidate_pending_validation",
                    "effectiveness_verified": False}
        # Validate all inputs before publishing any candidate. One directory rename
        # keeps code and its proposal together; this does not touch the live library.
        self.output.mkdir(parents=True, exist_ok=True)
        destination = self.output / "candidate"
        if destination.exists():
            raise ValueError("A candidate is already saved here; use a new learning session")
        with TemporaryDirectory(prefix=".candidate-", dir=self.output) as temporary:
            staged = Path(temporary)
            for name, code in sources.items():
                (staged / "tools").mkdir(exist_ok=True)
                (staged / "tools" / f"{name}.py").write_bytes(code)
            (staged / "proposal.json").write_text(canonical(proposal) + "\n", encoding="utf-8")
            staged.rename(destination)
        self.complete = True
        self._record("joint_update_submitted", action=action, status=proposal["status"])
        return {"success": True, "session_complete": True, "action": action,
                "status": proposal["status"], "effectiveness_verified": False}

    def finish_without_update(self, reason):
        return self.submit_update(reason)

    def execute(self, name, arguments):
        if name not in self.names:
            raise ValueError("Unknown joint update operation")
        if self.complete:
            raise ValueError("This learning session has already submitted its final decision")
        try:
            result = getattr(self, name)(**arguments)
        except (ValueError, TypeError, KeyError, SyntaxError) as error:
            result = {"success": False, "session_complete": False, "error": str(error)}
            self._record("joint_update_error", operation=name, error=str(error))
        return canonical(result)
