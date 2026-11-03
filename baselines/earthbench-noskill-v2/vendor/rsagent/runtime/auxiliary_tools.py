"""Session-only code exploration with full command output and bounded code archives."""
from pathlib import Path
from dataclasses import replace

from rsagent.runtime.agent import ToolBox
from rsagent.runtime.jsonio import dump
from rsagent.runtime.sandbox import NamespaceRunner


class AuditedAuxiliaryRunner(NamespaceRunner):
    def __init__(self, recorder, archive, **kwargs):
        super().__init__(**kwargs)
        self.recorder, self.archive, self.sequence = recorder, Path(archive), 0

    def _snapshot(self, workspace, moment):
        workspace = Path(workspace).resolve()
        files, omitted = [], []
        for path in sorted(workspace.rglob("*")):
            if path.suffix.lower() not in {".py", ".sh", ".r"} or not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                continue
            relative = str(path.relative_to(workspace))
            if len(files) >= 64 or path.stat().st_size > 262144:
                omitted.append(relative)
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                omitted.append(relative)
                continue
            files.append({"path": relative, "content": content})
        target = self.archive / f"command-{self.sequence:04d}-{moment}.json"
        dump(target, {"files": files, "omitted": omitted,
                      "scope": "Snapshot around a command, not proof each file executed; audit only, not a tool library"})
        self.recorder.log("auxiliary_code_snapshot", command_index=self.sequence, moment=moment,
                          file=str(target), paths=[f["path"] for f in files], omitted=omitted)

    def run(self, workspace, command, env, timeout):
        self.sequence += 1
        self._snapshot(workspace, "before")
        try:
            result = super().run(workspace, command, env, timeout)
            self.recorder.log("auxiliary_command_result", command_index=self.sequence, command=command,
                output=result.stdout, exit_code=result.exit_code, timed_out=result.timed_out,
                scope="Full auxiliary output; not official final-answer evidence")
            return result
        finally:
            self._snapshot(workspace, "after")


class AuxiliaryToolBox(ToolBox):
    """Only supplied official provider plus ordinary local tools; no tool registry."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed = {d["name"] for d in self.definitions}

    def execute_detailed(self, name, arguments):
        if name not in self.allowed:
            raise ValueError("Unavailable tool; temporary scripts cannot register new tools: " + name)
        return super().execute_detailed(name, arguments)


class ReturningAuxiliaryToolBox(AuxiliaryToolBox):
    """Append a visible workflow notice without changing command output or status."""
    def __init__(self, *args, workflow, **kwargs):
        super().__init__(*args, **kwargs)
        self.workflow = workflow

    def execute_detailed(self, name, arguments):
        result = super().execute_detailed(name, arguments)
        if name == "run_command":
            result = replace(result, output=result.output + self.workflow.note_auxiliary())
        return result
