"""Read-only, snapshot-pinned discovery and reuse for a task Agent."""
from pathlib import Path

from rsagent.evolution.evolution_tools import disjoint
from rsagent.evolution.tool_registry import canonical, safe_bytes


class TrialCapabilityProvider:
    def __init__(self, registry, snapshot, workspace, readonly=(), recorder=None, *,
                 allow_pending_validation=False):
        self.registry, self.snapshot_id = registry, snapshot
        self.workspace = Path(workspace).resolve()
        self.readonly = [Path(p).resolve() for p in readonly]
        if any(not disjoint(registry.root, path) for path in (self.workspace, *self.readonly)):
            raise ValueError("Task mounts must not expose the private trial library")
        self.payload, self.recorder = registry.snapshot(snapshot), recorder
        if type(allow_pending_validation) is not bool:
            raise ValueError("allow_pending_validation must be a boolean")
        self.allow_pending_validation = allow_pending_validation
        self._check_skill_validation()
        string = {"type": "string"}
        specs = [
            ("trial_catalog", "List trial tools and advisory Skills in this frozen snapshot. "
             "They are available for use, not proven to improve task results.", {}, []),
            ("trial_tool_read", "Read a trial tool's source and input/output description, in pages.",
             {"name": string, "offset": {"type": "integer", "minimum": 0},
              "length": {"type": "integer", "minimum": 1, "maximum": 8000}}, ["name"]),
            ("trial_tool_checkout", "Copy this snapshot's tool into the workspace for local inspection or editing. "
             "Does not modify the snapshot; destination must not exist.",
             {"name": string, "destination": string}, ["name", "destination"]),
            ("trial_skill_read", "Read optional decision guidance. You may adapt or reject it; "
             "a listed dependency does not certify scientific correctness.", {"name": string}, ["name"]),
            ("trial_tool_run", "Run a trial tool's main(arguments) in the isolated task workspace. "
             "Input data stay read-only and outputs stay in /workspace. Uses the pinned version, not a live library.",
             {"name": string, "arguments": {"type": "object"}}, ["name", "arguments"]),
        ]
        self.definitions = [{"name": name, "description": description, "input_schema": {
            "type": "object", "properties": fields, "required": required, "additionalProperties": False}}
            for name, description, fields, required in specs]
        self.names = {definition["name"] for definition in self.definitions}

    def _check_skill_validation(self):
        if self.payload["skills"] and (getattr(self.registry, "require_source_validation", False) is True
                                       or self.payload.get("source_validation_required")):
            self.registry.assert_skill_reuse(self.snapshot_id,
                                            allow_pending_validation=self.allow_pending_validation)

    def catalog(self):
        """Public discovery metadata, without reading Skill bodies or running tools."""
        self._check_skill_validation()
        return {"snapshot": self.snapshot_id, "status": "trial", "effectiveness_verified": False,
                "tools": [{"name": key, "description": tool["description"],
                           "validation_scope": tool["validation"]["scope"]}
                          for key, tool in self.payload["tools"].items()],
                "skills": [{"name": key, "when": skill["when"],
                            "requires_tools": list(skill.get("requires_tools", []))}
                           for key, skill in self.payload["skills"].items()]}

    def execute(self, name, arguments):
        if name not in self.names:
            raise ValueError("Unknown trial operation; this provider cannot publish updates")
        self._check_skill_validation()
        if name == "trial_catalog":
            result = self.catalog()
        elif name == "trial_skill_read":
            skill = self.payload["skills"][arguments["name"]]
            result = {key: skill.get(key, [] if key == "requires_tools" else "")
                      for key in ("name", "when", "guidance", "limitations", "requires_tools")}
            result["advisory"] = True
        elif name == "trial_tool_run":
            result = self.registry.run(self.snapshot_id, arguments["name"], arguments["arguments"],
                                       workspace=self.workspace, readonly=self.readonly)
        else:
            tool = self.payload["tools"][arguments["name"]]
            source = self.registry._directory(self.snapshot_id) / tool["source"]
            code = safe_bytes(source).decode()
            if name == "trial_tool_read":
                offset, length = arguments.get("offset", 0), arguments.get("length", 4000)
                if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 8000:
                    raise ValueError("Use a nonnegative offset and length in 1..8000")
                result = {"name": tool["name"], "description": tool["description"],
                          "source": code[offset:offset + length], "offset": offset, "total_chars": len(code),
                          "next_offset": offset + length if offset + length < len(code) else None}
            else:
                destination = Path(arguments["destination"])
                if destination.is_relative_to("/workspace"):
                    destination = destination.relative_to("/workspace")
                destination = (self.workspace / destination).resolve()
                if not destination.is_relative_to(self.workspace):
                    raise ValueError("Checkout destination must be inside the task workspace")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("x", encoding="utf-8") as stream:
                    stream.write(code)
                result = {"name": tool["name"], "destination": str(destination.relative_to(self.workspace))}
        if self.recorder:
            self.recorder.log("trial_asset_used", operation=name, name=arguments.get("name"),
                              snapshot=self.snapshot_id,
                              success=result["success"] if name == "trial_tool_run" else True)
        return canonical(result)
