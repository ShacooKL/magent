"""Immediate sandbox test feedback around the existing structured Skill provider.

References and a passing self-authored test do not establish causal/scientific truth.
No registry activation. No task-specific diagnostics or reference answers.
"""
from copy import deepcopy
from pathlib import Path, PurePosixPath
import shlex
from tempfile import TemporaryDirectory

from rsagent.evolution.tool_registry import canonical, digest, safe_bytes
from rsagent.runtime.sandbox import NamespaceRunner


class CounterexampleSkillProvider:
    def __init__(self, structured, audit_root, runner=None):
        self.structured = structured
        self.workspace = structured.workspace
        self.audit_root = Path(audit_root).resolve()
        if self.audit_root.is_relative_to(self.workspace) or self.workspace.is_relative_to(self.audit_root):
            raise ValueError("Test audit store must be outside workspace")
        self.runner = runner or NamespaceRunner()
        self.tests = {}
        self.definitions = deepcopy(structured.definitions)
        for d in self.definitions:
            if d["name"] == "save_skill":
                d["input_schema"]["properties"]["test_id"] = {
                    "type": "string", "description": "Passed test returned by test_skill_counterexample"}
                d["input_schema"]["required"].append("test_id")
                d["description"] += " Requires a passed, unchanged synthetic test covering the cited evidence; this is not scientific certification."
        self.definitions.append({"name": "test_skill_counterexample",
            "description": "Run a self-contained Python counterexample in an isolated copy. Bind fully read E handles to the test. Contrast the actual observed operation and your proposed alternative. Return immediate errors/results; do not treat a passing script as proof of causality.",
            "input_schema": {"type": "object", "additionalProperties": False,
                "properties": {
                    "test_path": {"type": "string", "description": "Relative path or /workspace/...; Python only"},
                    "evidence_handles": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "observed_operation": {"type": "string"},
                    "proposed_alternative": {"type": "string"},
                    "expected_distinction": {"type": "string"}},
                "required": ["test_path", "evidence_handles", "observed_operation",
                             "proposed_alternative", "expected_distinction"]}})
        self.names = {d["name"] for d in self.definitions}

    def wrap_trajectory(self, provider):
        return self.structured.wrap_trajectory(provider)

    def inspection_progress(self):
        return self.structured.inspection_progress()

    def resolve_test(self, name):
        if not isinstance(name, str):
            raise ValueError("test_path must be text")
        path = PurePosixPath(name)
        if ".." in path.parts:
            raise ValueError("Parent traversal not allowed")
        if path.is_relative_to("/workspace"):
            path = path.relative_to("/workspace")
        if path.is_absolute() or path.suffix != ".py":
            raise ValueError("Use a workspace-relative Python path")
        actual = self.workspace
        for part in path.parts:
            actual /= part
            if actual.is_symlink():
                raise ValueError("Symlink test paths are not allowed")
        if not actual.is_file():
            raise ValueError("Test script does not exist")
        return actual, str(path)

    def code_bundle(self):
        bundle, total = {}, 0
        for path in self.workspace.rglob("*.py"):
            actual, name = self.resolve_test(str(path.relative_to(self.workspace)))
            data = safe_bytes(actual, 1024 * 1024)
            total += len(data)
            if total > 4 * 1024 * 1024:
                raise ValueError("Test code bundle exceeds 4 MiB")
            bundle[name] = data
        return bundle

    def test(self, args):
        required = {"test_path", "evidence_handles", "observed_operation",
                    "proposed_alternative", "expected_distinction"}
        if not isinstance(args, dict) or set(args) != required:
            raise ValueError("Pass exactly the advertised test parameters")
        if any(not isinstance(args[k], str) or not args[k].strip() for k in required - {"evidence_handles"}):
            raise ValueError("Operation descriptions and path must be nonempty text")
        handles = args["evidence_handles"]
        if not isinstance(handles, list) or not handles or not all(isinstance(h, str) for h in handles):
            raise ValueError("Provide returned E handles")
        _, errors, handles = self.structured._resolve(handles)
        if errors:
            return {"success": False, "errors": errors, "session_complete": False}
        _, name = self.resolve_test(args["test_path"])
        bundle = self.code_bundle()
        self.audit_root.mkdir(parents=True, exist_ok=True)
        tid = f"T{len(list(self.audit_root.glob('T[0-9]*'))) + 1}"
        folder = self.audit_root / tid
        folder.mkdir(exist_ok=False)
        archive = folder / "code"
        for relative, data in bundle.items():
            dest = archive / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        with TemporaryDirectory(prefix="replay-", dir=folder) as tmp:
            work = Path(tmp)
            for relative, data in bundle.items():
                dest = work / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            result = self.runner.run(work, "python " + shlex.quote(name), {}, 30)
        record = {"test_id": tid, "success": result.exit_code == 0 and not result.timed_out,
            "exit_code": result.exit_code, "timed_out": result.timed_out, "output": result.stdout,
            "test_path": name, "evidence_handles": handles,
            **{k: args[k] for k in ("observed_operation", "proposed_alternative", "expected_distinction")},
            "code_sha256": {k: digest(v) for k, v in bundle.items()},
            "scientific_correctness_verified": False, "transfer_tested": False,
            "warning": "Self-authored test execution only; descriptions and causal links are model claims.",
            "session_complete": False}
        self.tests[tid] = record
        (folder / "result.json").write_text(canonical(record) + "\n")
        return record

    def execute(self, name, arguments):
        try:
            if self.structured.complete:
                raise ValueError("Session already finished")
            if name == "test_skill_counterexample":
                result = self.test(arguments)
            elif name == "save_skill":
                args = dict(arguments)
                tid = args.pop("test_id", None)
                test = self.tests.get(tid)
                if not test or not test["success"]:
                    raise ValueError("Run a successful test_skill_counterexample before saving")
                if test["code_sha256"] != {k: digest(v) for k, v in self.code_bundle().items()}:
                    raise ValueError("Test code changed; rerun the counterexample before saving")
                diagnosis = self.structured.diagnosis
                if diagnosis is None:
                    raise ValueError("Save a diagnosis first")
                refs = set(diagnosis["evidence_handles"] + args.get("evidence_handles", []))
                if not refs.issubset(test["evidence_handles"]):
                    raise ValueError("Counterexample must cover all cited diagnosis/Skill handles")
                import json
                result = json.loads(self.structured.execute(name, args))
                if result.get("success"):
                    result["test_id"] = tid
                    (self.audit_root / "selected-test.json").write_text(canonical(test) + "\n")
            else:
                return self.structured.execute(name, arguments)
        except (ValueError, OSError, TypeError, KeyError, RuntimeError) as error:
            result = {"success": False, "errors": [{"message": str(error)}], "session_complete": False}
        if self.structured.recorder:
            self.structured.recorder.log("counterexample_feedback", operation=name, **result)
        return canonical(result)
