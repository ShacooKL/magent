"""Versioned single-file tools. Candidate code is never imported by the host.

The operator owns policies; the Agent may propose code, not expected answers.
V1 validates JSON contracts only, not raster scientific correctness.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import sys
import tempfile
from uuid import uuid4

from rsagent.runtime.sandbox import NamespaceRunner

LIMIT = 128 * 1024


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    canonical(value)  # Also reject overflowing float literals, e.g. 1e999.
    return value


def safe_bytes(path, limit=LIMIT):
    # Reject symlinks, devices and FIFOs written by untrusted candidates.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Expected bounded regular file")
        result = stream.read(limit + 1)
        if len(result) > limit:
            raise ValueError("File exceeds size limit")
        return result


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise ValueError("Invalid tool name")
    return value


class ToolRegistry:
    def __init__(self, root, recorder=None):
        self.root = Path(root).resolve()
        for exposed in [Path(sys.prefix), *map(Path, ("/usr", "/bin", "/lib", "/lib64"))]:
            exposed = exposed.resolve()
            if self.root.is_relative_to(exposed) or exposed.is_relative_to(self.root):
                raise ValueError("Registry cannot overlap sandbox runtime mounts")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.recorder = recorder

    @contextmanager
    def locked(self):
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def read(self, relative, default=None):
        path = self.root / relative
        return strict_json(safe_bytes(path, 2 * 1024 * 1024).decode()) if path.exists() else default

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        temporary.write_text(canonical(value) + "\n", encoding="utf-8")
        temporary.replace(path)

    def event(self, kind, **fields):
        record = {"type": kind, "time": datetime.now(timezone.utc).isoformat(), **fields}
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(canonical(record) + "\n")
        if self.recorder:
            self.recorder.log(kind, **fields)

    def set_policy(self, name, cases):
        """Operator-only, append-only regression cases. Never exposed as an Agent tool."""
        name = identifier(name)
        if not isinstance(cases, list) or not 1 <= len(cases) <= 20:
            raise ValueError("Policy needs 1..20 cases")
        seen = set()
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"id", "arguments", "expected"}:
                raise ValueError("Case requires exactly id, arguments, expected")
            if not isinstance(case["id"], str) or not case["id"] or case["id"] in seen:
                raise ValueError("Case IDs must be unique nonempty strings")
            seen.add(case["id"])
            if not isinstance(case["arguments"], dict):
                raise ValueError("Arguments must be an object")
        encoded = canonical(cases).encode()
        if len(encoded) > LIMIT:
            raise ValueError("Policy too large")
        with self.locked():
            previous = self.read(f"policies/{name}.json", [])
            by_id = {c["id"]: canonical(c) for c in cases}
            if any(by_id.get(c["id"]) != canonical(c) for c in previous):
                raise ValueError("Existing regression cases cannot be removed or changed")
            self.write(f"policies/{name}.json", cases)
            self.event("tool_policy_updated", tool=name, policy_hash=digest(encoded))

    def index(self):
        return self.read("index.json", {})

    def revision(self, name, revision):
        identifier(name)
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{32}", revision):
            raise ValueError("Invalid revision")
        base = f"versions/{name}/{revision}"
        manifest = self.read(base + "/manifest.json")
        if manifest is None:
            raise ValueError("Unknown revision")
        code = safe_bytes(self.root / base / "tool.py")
        if digest(code) != manifest["sha256"]:
            raise ValueError("Frozen tool code changed")
        return manifest, self.root / base / "tool.py"

    def propose(self, name, source, description, parent=None):
        identifier(name)
        code = safe_bytes(source)
        # Compile only: no execution/import on the host.
        compile(code, "candidate.py", "exec")
        if not isinstance(description, str) or not 1 <= len(description) <= 2000:
            raise ValueError("Provide a description of at most 2000 characters")
        with self.locked():
            active = self.index().get(name, {}).get("active")
            if parent != active:
                raise ValueError("Parent must equal current active revision (null for a new tool)")
            revision = uuid4().hex
            base = self.root / "versions" / name / revision
            base.mkdir(parents=True)
            (base / "tool.py").write_bytes(code)
            manifest = {"name": name, "revision": revision, "parent": parent,
                        "description": description, "sha256": digest(code),
                        "contract": "def main(arguments: dict) -> JSON-compatible value"}
            self.write(str(base.relative_to(self.root) / "manifest.json"), manifest)
            self.event("tool_proposed", tool=name, revision=revision, parent=parent, sha256=manifest["sha256"])
            return manifest

    def execute_revision(self, name, revision, arguments, readonly=(), workspace=None):
        """Each validation case gets a fresh namespace; no policy or expected output mount."""
        manifest, source = self.revision(name, revision)
        for exposed in [*map(Path, readonly), *([Path(workspace)] if workspace is not None else [])]:
            exposed = exposed.resolve()
            if self.root.is_relative_to(exposed) or exposed.is_relative_to(self.root):
                raise ValueError("Execution mounts cannot expose registry or policies")
        if not isinstance(arguments, dict) or len(canonical(arguments).encode()) > LIMIT:
            raise ValueError("Arguments must be a bounded JSON object")
        with tempfile.TemporaryDirectory(prefix="tool-execution-") as temporary:
            execution_workspace = Path(temporary) if workspace is None else Path(workspace)
            # Input/result paths are fresh even for calls in a persistent task workspace.
            request_name, response_name = f".tool-{uuid4().hex}.input.json", f".tool-{uuid4().hex}.output.json"
            (execution_workspace / request_name).write_text(canonical(arguments))
            script = (
                "import json,runpy; "
                f"module=runpy.run_path({str(source)!r}); "
                f"arguments=json.load(open({request_name!r})); "
                "result=module['main'](arguments); "
                f"open({response_name!r},'w').write(json.dumps(result,allow_nan=False))"
            )
            runner = NamespaceRunner(read_only_paths=[source, *map(Path, readonly)])
            result = runner.run(execution_workspace,
                "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; python -c " + shlex.quote(script), {}, 30)
            success, value = result.exit_code == 0 and not result.timed_out, None
            if success:
                try:
                    value = strict_json(safe_bytes(execution_workspace / response_name).decode())
                except (OSError, ValueError, UnicodeError):
                    success = False
            return {"success": success, "value": value, "exit_code": result.exit_code,
                    "timed_out": result.timed_out, "stdout": result.stdout[-4000:],
                    "revision": revision, "sha256": manifest["sha256"]}

    def validate(self, name, revision):
        with self.locked():
            manifest, _ = self.revision(name, revision)
            cases = self.read(f"policies/{name}.json")
            if not cases:
                raise ValueError("No operator validation policy; activation is blocked")
            checks = []
            for case in cases:
                result = self.execute_revision(name, revision, case["arguments"])
                checks.append({"case": case["id"], "passed": result["success"] and
                               canonical(result["value"]) == canonical(case["expected"]),
                               "execution_ok": result["success"]})
            report = {"tool": name, "revision": revision, "sha256": manifest["sha256"],
                      "policy_hash": digest(canonical(cases).encode()),
                      "success": all(c["passed"] for c in checks), "checks": checks,
                      "scope": "operator JSON contract cases, not proof of scientific correctness"}
            self.write(f"validations/{name}/{revision}.json", report)
            self.event("tool_validated", **report)
            return report

    def activate(self, name, revision, rollback=False):
        with self.locked():
            manifest, _ = self.revision(name, revision)
            policy = self.read(f"policies/{name}.json")
            report = self.read(f"validations/{name}/{revision}.json")
            if not policy or not report or not report["success"] or report["sha256"] != manifest["sha256"] or report["policy_hash"] != digest(canonical(policy).encode()):
                raise ValueError("Matching successful validation against current policy required")
            index = self.index()
            entry = index.get(name, {"active": None, "accepted": []})
            if rollback:
                if revision not in entry["accepted"]:
                    raise ValueError("Rollback target must be a previously accepted version")
            elif manifest["parent"] != entry["active"]:
                raise ValueError("Stale parent; propose against current active version")
            previous = entry["active"]
            entry["active"] = revision
            if revision not in entry["accepted"]:
                entry["accepted"].append(revision)
            index[name] = entry
            self.write("index.json", index)
            self.event("tool_rolled_back" if rollback else "tool_activated", tool=name,
                       revision=revision, previous=previous, policy_hash=report["policy_hash"])
            return {"tool": name, "active": revision, "previous": previous}

    def discover(self):
        with self.locked():
            return [self.revision(name, entry["active"])[0] for name, entry in sorted(self.index().items())]

    def contracts(self):
        """Expose names only so cold-start agents can target an operator contract."""
        with self.locked():
            return sorted(p.stem for p in (self.root / "policies").glob("*.json"))

    def run(self, name, arguments, readonly=(), workspace=None):
        with self.locked():
            identifier(name)
            revision = self.index().get(name, {}).get("active")
            if revision is None:
                raise ValueError("No active tool")
            result = self.execute_revision(name, revision, arguments, readonly, workspace)
            self.event("versioned_tool_result", tool=name, revision=revision,
                       sha256=result["sha256"], success=result["success"],
                       exit_code=result["exit_code"], timed_out=result["timed_out"])
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    policy = sub.add_parser("set-policy", help="Operator only; add immutable regression cases")
    policy.add_argument("name")
    policy.add_argument("file", type=Path)
    sub.add_parser("list")
    proposal = sub.add_parser("propose")
    proposal.add_argument("name")
    proposal.add_argument("source", type=Path)
    proposal.add_argument("--description", required=True)
    proposal.add_argument("--parent")
    for action in ("validate", "activate", "rollback"):
        command = sub.add_parser(action)
        command.add_argument("name")
        command.add_argument("revision")
    args = parser.parse_args()
    registry = ToolRegistry(args.root)
    if args.action == "set-policy":
        registry.set_policy(args.name, strict_json(safe_bytes(args.file).decode()))
        print("Policy registered; candidate code is not yet active.")
    elif args.action == "list":
        print(canonical(registry.discover()))
    elif args.action == "propose":
        print(canonical(registry.propose(args.name, args.source, args.description, args.parent)))
    elif args.action == "validate":
        result = registry.validate(args.name, args.revision)
        print(canonical(result))
        return 0 if result["success"] else 1
    else:
        print(canonical(registry.activate(args.name, args.revision, rollback=args.action == "rollback")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
