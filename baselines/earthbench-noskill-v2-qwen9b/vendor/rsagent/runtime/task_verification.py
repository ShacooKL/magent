"""Opt-in artifact verification protocol, not a correctness oracle.

The model supplies requirements and checks. The host enforces coverage, execution,
freshness and an immutable submission copy. No benchmark answers or domain fixes.
One provider per task; no restart/resume support in this experimental protocol.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
from tempfile import TemporaryDirectory

from rsagent.runtime.agent import ToolBox, _tool
from rsagent.runtime.clock import now_beijing
from rsagent.runtime.jsonio import dump
from rsagent.runtime.sandbox import NamespaceRunner


PROTOCOL = """Experimental task verification:
Read the public task and inputs, then call record_requirements BEFORE writing code
or running commands. Include all deliverables and constraints, their public basis,
and a check plan. Do not use private reference answers. Resolve ambiguity by reading
inputs/documentation, not by assuming two libraries share a representation.
Solve normally. Write a Python checker that REOPENS the final files, derives expected
properties independently from public requirements/inputs, and compares actual data.
Do not merely repeat the solver's constants or report file existence/exit code.
Call run_task_checks(script_path). Checks run in an isolated workspace copy, with
no network; do not modify that copy or create files (python bytecode is disabled).
Print exactly one JSON object to stdout: {"checks": [{"id": requirement_id,
"status": "pass|fail|unverified", "expected": any_JSON, "actual": any_JSON,
"basis": "public source/derivation", "method": "what was compared",
"artifacts": ["relative/final-file"]}]}. One row per recorded requirement.
Fix from returned feedback and rerun within the SAME request/step budget. Editing
any file, including checker code, invalidates the last check. A zero exit code alone
does not pass. When covered checks pass, call finish_task(status="checked", summary=...).
If unable to finish, call finish_task(status="incomplete", summary=reason/remaining work).
Plain final text is not submission. 'checked' only means your checks passed; it does
not certify scientific correctness or replace independent benchmark scoring.
"""


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON number: " + value)

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    # Also rejects finite-looking literals such as 1e999 that overflow to infinity.
    json.dumps(value, allow_nan=False)
    return value


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 8000


class TaskVerificationProvider:
    def __init__(self, workspace, audit_root, *, runner=None, timeout=30,
                 max_bytes=512 * 1024 * 1024, max_files=2000):
        self.workspace = Path(workspace).resolve(strict=True)
        self.audit_root = Path(audit_root).resolve()
        if (self.audit_root.is_relative_to(self.workspace)
                or self.workspace.is_relative_to(self.audit_root)):
            raise ValueError("Audit store must be outside the model workspace")
        self.runner = runner or NamespaceRunner()
        # Host callers must supply immutable public mounts only, never private scorers.
        for path in getattr(self.runner, "read_only_paths", []):
            mount = Path(path).resolve()
            if self.audit_root.is_relative_to(mount) or mount.is_relative_to(self.audit_root):
                raise ValueError("Audit store must not be visible through a read-only mount")
        self.audit_root.mkdir(parents=True, exist_ok=False)
        self.timeout, self.max_bytes, self.max_files = timeout, max_bytes, max_files
        self.requirements = None
        self.last_check = None
        self.finished = None
        self.attempts = 0
        text = {"type": "string"}
        req = {"type": "object", "additionalProperties": False,
               "properties": {k: text for k in ("id", "requirement", "basis", "check_plan")},
               "required": ["id", "requirement", "basis", "check_plan"]}
        self.definitions = [
            _tool("record_requirements", "Record public task requirements once before solving; cannot delete failed requirements later.",
                  {"requirements": {"type": "array", "items": req, "minItems": 1, "maxItems": 50}}, ["requirements"]),
            _tool("run_task_checks", "Run a Python checker on a workspace copy; returns per-requirement evidence and errors. Follow the verification protocol JSON format.",
                  {"script_path": text}, ["script_path"]),
            _tool("finish_task", "Freeze checked artifacts, or explicitly finish incomplete. This is self-check evidence, not independent grading.",
                  {"status": {"type": "string", "enum": ["checked", "incomplete"]}, "summary": text},
                  ["status", "summary"]),
        ]
        self.names = {d["name"] for d in self.definitions}

    def _path(self, name):
        if not _text(name):
            raise ValueError("Use a nonempty relative workspace file path")
        path = PurePosixPath(name)
        if path.is_relative_to("/workspace"):
            path = path.relative_to("/workspace")
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Path must stay inside workspace")
        current = self.workspace
        for part in path.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Symlinks are not supported")
        if not current.is_file():
            raise ValueError("Missing workspace file: " + str(path))
        return str(path)

    def manifest(self, root=None):
        """Conservative all-file binding, streamed to avoid loading rasters in memory."""
        root = root or self.workspace
        result, total = {}, 0
        def fail(error):
            raise error
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
            for name in sorted(dirs + files):
                path = Path(directory) / name
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError("Only regular files/directories supported: " + name)
                if stat.S_ISDIR(mode):
                    continue
                total += path.stat().st_size
                if total > self.max_bytes or len(result) >= self.max_files:
                    raise ValueError("Workspace snapshot exceeds configured size/file limit")
                hashed = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        hashed.update(block)
                result[str(path.relative_to(root))] = hashed.hexdigest()
        return result

    def record(self, args):
        if self.requirements is not None:
            raise ValueError("Requirements already recorded; failed items cannot be removed")
        rows = args["requirements"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 50:
            raise ValueError("Provide 1..50 requirements")
        ids = set()
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"id", "requirement", "basis", "check_plan"}
                    or not all(_text(value) for value in row.values())
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", row["id"])
                    or row["id"] in ids):
                raise ValueError("Requirements need unique short IDs and nonempty requirement/basis/check_plan")
            ids.add(row["id"])
        self.requirements = deepcopy(rows)
        dump(self.audit_root / "requirements.json", self.requirements)
        return {"success": True, "requirement_ids": [r["id"] for r in self.requirements]}

    def _validate_report(self, report, manifest):
        if not isinstance(report, dict) or set(report) != {"checks"} or not isinstance(report["checks"], list):
            raise ValueError("Checker must print one JSON object with a checks array")
        ids = set()
        for row in report["checks"]:
            fields = {"id", "status", "expected", "actual", "basis", "method", "artifacts"}
            if (not isinstance(row, dict) or set(row) != fields
                    or not _text(row["id"]) or row["id"] in ids
                    or row["status"] not in ("pass", "fail", "unverified")
                    or not _text(row["basis"]) or not _text(row["method"])
                    or not isinstance(row["artifacts"], list) or not row["artifacts"]):
                raise ValueError("Invalid check row: require unique ID, status, expected/actual, basis/method and artifacts")
            ids.add(row["id"])
            if any(self._path(p) not in manifest for p in row["artifacts"]):
                raise ValueError("Evidence artifact not in checked snapshot")
        if ids != {r["id"] for r in self.requirements}:
            raise ValueError("Check coverage must match ALL recorded requirement IDs exactly")
        return report["checks"]

    def check(self, args):
        if self.requirements is None:
            raise ValueError("Record requirements first")
        # Any attempted new check invalidates a previous pass, including malformed runs.
        self.last_check = None
        name = self._path(args["script_path"])
        if not name.endswith(".py"):
            raise ValueError("Checker must be a Python script")
        before = self.manifest()
        self.attempts += 1
        folder = self.audit_root / f"check-{self.attempts:03d}"
        folder.mkdir()
        archive = folder / "snapshot"
        shutil.copytree(self.workspace, archive, symlinks=True)
        record = {"success": False, "check_id": folder.name, "checks": [],
                  "manifest": before, "errors": [], "independently_verified": False}
        try:
            if self.manifest(archive) != before:
                raise ValueError("Workspace changed while taking snapshot; retry checking")
            with TemporaryDirectory(prefix="verify-") as temporary:
                work = Path(temporary) / "workspace"
                shutil.copytree(archive, work, symlinks=True)
                result = self.runner.run(work, "python -B " + shlex.quote(name), {}, self.timeout)
                record.update(exit_code=result.exit_code, timed_out=result.timed_out, output=result.stdout[:64000])
                if result.exit_code != 0 or result.timed_out:
                    raise ValueError("Checker failed/timed out; inspect output and repair")
                if self.manifest(work) != before:
                    raise ValueError("Checker modified its input snapshot; use read-only checks")
                if len(result.stdout) > 64000:
                    raise ValueError("Checker report exceeds 64000 characters")
                record["checks"] = self._validate_report(_json(result.stdout), before)
            if self.manifest() != before:
                raise ValueError("Workspace changed during checking; rerun")
            record["success"] = all(row["status"] == "pass" for row in record["checks"])
            if not record["success"]:
                record["errors"].append("Some requirements failed or remain unverified; repair and rerun")
        except (ValueError, OSError, RuntimeError, TypeError) as error:
            record["errors"].append(str(error))
        record["time"] = now_beijing().isoformat()
        dump(folder / "result.json", record)
        self.last_check = record
        return record

    def finish(self, args):
        if args["status"] not in ("checked", "incomplete") or not _text(args["summary"]):
            raise ValueError("Provide status checked/incomplete and a nonempty summary")
        record = {**args, "time": now_beijing().isoformat(), "independently_verified": False}
        if args["status"] == "checked":
            check = self.last_check
            if not check or not check["success"]:
                raise ValueError("No successful check: run_task_checks and repair first")
            if self.manifest() != check["manifest"]:
                raise ValueError("Files changed after checking; rerun run_task_checks")
            # Submit the exact checked archive, NOT a mutable live directory.
            source = self.audit_root / check["check_id"] / "snapshot"
            if self.manifest(source) != check["manifest"]:
                raise ValueError("Checked snapshot changed; rerun checks")
            target = self.audit_root / "submission"
            with TemporaryDirectory(prefix="submission-", dir=self.audit_root) as tmp:
                staged = Path(tmp) / "files"
                shutil.copytree(source, staged, symlinks=True)
                if self.manifest(staged) != check["manifest"]:
                    raise ValueError("Submission copy differs from checked snapshot")
                staged.rename(target)
            record.update(check_id=check["check_id"], manifest=check["manifest"], submission=str(target))
        dump(self.audit_root / "completion.json", record)
        self.finished = record
        return {"success": True, "session_complete": True, **record}

    def execute(self, name, args):
        try:
            if self.finished:
                raise ValueError("Task is already finished")
            required = {"record_requirements": {"requirements"}, "run_task_checks": {"script_path"},
                        "finish_task": {"status", "summary"}}
            if name not in required or not isinstance(args, dict) or set(args) != required[name]:
                raise ValueError("Use exactly the advertised tool arguments")
            handler = {"record_requirements": self.record, "run_task_checks": self.check, "finish_task": self.finish}[name]
            result = handler(args)
        except (ValueError, OSError, TypeError, KeyError, RuntimeError) as error:
            result = {"success": False, "errors": [str(error)], "session_complete": False}
        # Full manifests/output stay in host audit; keep model feedback small.
        visible = {key: value for key, value in result.items() if key not in {"manifest", "submission"}}
        if "output" in visible:
            visible["output"] = visible["output"][:4000]
        if len(json.dumps(visible, ensure_ascii=False)) > 24000:
            # Avoid the generic ToolBox truncator cutting JSON in half. Full
            # evidence remains in the host record; request concise checks next.
            visible["checks"] = [{"id": row["id"], "status": row["status"],
                                  "expected_excerpt": json.dumps(row["expected"], ensure_ascii=False)[:120],
                                  "actual_excerpt": json.dumps(row["actual"], ensure_ascii=False)[:120]}
                                 for row in visible.get("checks", [])]
            visible["evidence_truncated"] = True
            visible["notice"] = "Evidence shortened for context; full report archived. Prefer concise expected/actual values."
        return json.dumps(visible, ensure_ascii=False, allow_nan=False)

    def final_review(self, _text):
        if not self.finished:
            return ("Task not submitted. Record requirements, run_task_checks and repair, then finish_task(checked); "
                    "or finish_task(incomplete) with remaining issues. A self-check is not independent grading.")
        return None


class VerifyingToolBox(ToolBox):
    """Opt-in wrapper; the default ToolBox and Agent loop are unchanged."""
    def __init__(self, workspace, approve, verification, *, runner=None, read_only_paths=None):
        self.verification = verification
        super().__init__(workspace, approve, runner=runner, read_only_paths=read_only_paths,
                         extra_tools=verification)

    def execute_detailed(self, name, arguments):
        if self.verification.finished:
            raise ValueError("Task already finished; frozen submission cannot be edited")
        if self.verification.requirements is None and name in {"write_file", "edit_file", "run_command"}:
            raise ValueError("Read task/inputs and record_requirements before writing or running code")
        result = super().execute_detailed(name, arguments)
        if name in self.verification.names:
            try:
                passed = json.loads(result.output).get("success") is True
            except ValueError:
                passed = False
            if not passed:
                result.success = False
                result.error_type = "verification_rejected"
        return result
