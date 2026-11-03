"""Command runners for isolated and explicit local execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from typing import Mapping


@dataclass
class SandboxResult:
    stdout: str
    exit_code: int | None
    timed_out: bool = False


class NamespaceRunner:
    """Rootless Linux sandbox using user/mount/PID/network namespaces."""

    name = "namespace"
    network = False

    def __init__(
        self,
        executable: str | None = None,
        worker: Path | None = None,
        read_only_paths: list[Path] | None = None,
        runtime_prefix: Path | None = None,
        source_workspace: Path | None = None,
    ) -> None:
        self.executable = executable or shutil.which("unshare")
        self.worker = (worker or Path(__file__).with_name("sandbox_worker.py")).resolve()
        self.read_only_paths = [path.resolve() for path in (read_only_paths or [])]
        self.runtime_prefix = (runtime_prefix or Path(sys.prefix)).resolve()
        self.source_workspace = source_workspace.resolve(strict=True) if source_workspace else None
        self.writable_mount = '/diagnostics' if self.source_workspace else '/workspace'

    @property
    def available(self) -> bool:
        return self.executable is not None and self.worker.is_file()

    def build_argv(self, workspace: Path, command: str, timeout: int) -> list[str]:
        if not self.executable:
            raise RuntimeError("The required 'unshare' command is unavailable")
        if not self.worker.is_file():
            raise RuntimeError(f"Sandbox worker is missing: {self.worker}")
        argv = [
            self.executable,
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--net",
            "--ipc",
            "--uts",
            sys.executable,
            str(self.worker),
            "--workspace",
            str(workspace.resolve()),
            "--cpu-seconds",
            str(timeout + 5),
        ]
        for path in self.read_only_paths:
            argv.extend(["--read-only-path", str(path)])
        if self.source_workspace:
            if workspace.resolve() == self.source_workspace:
                raise ValueError('Source and diagnostic workspaces must be different')
            argv.extend(['--source-workspace', str(self.source_workspace)])
        return argv + ["--", command]

    def run(self, workspace: Path, command: str, env: Mapping[str, str], timeout: int) -> SandboxResult:
        if not self.available:
            raise RuntimeError("Namespace sandbox is unavailable on this system")
        safe_env = {key: value for key, value in env.items() if key in {"PATH", "LANG", "LC_ALL", "TERM", "TZ"}}
        safe_env["PY_AGENT_RUNTIME_PREFIX"] = str(self.runtime_prefix)
        return _run_process(self.build_argv(workspace, command, timeout), workspace, timeout, env=safe_env)

    def spawn(
        self,
        workspace: Path,
        command: str,
        env: Mapping[str, str] | None = None,
        cpu_seconds: int = 86_400,
    ) -> subprocess.Popen[str]:
        """Start a persistent process inside the same namespace sandbox."""
        safe_env = {
            key: value
            for key, value in (env or {}).items()
            if key in {"PATH", "LANG", "LC_ALL", "TERM", "TZ"}
        }
        safe_env["PY_AGENT_RUNTIME_PREFIX"] = str(self.runtime_prefix)
        return subprocess.Popen(
            self.build_argv(workspace, command, cpu_seconds),
            cwd=workspace,
            env=safe_env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1,
            start_new_session=True,
        )


class LocalRunner:
    """Explicitly unsafe compatibility runner."""

    name = "local"
    network = True
    available = True

    def run(self, workspace: Path, command: str, env: Mapping[str, str], timeout: int) -> SandboxResult:
        return _run_process(["/bin/bash", "-c", command], workspace, timeout, env=dict(env))


def _run_process(
    argv: list[str],
    workspace: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> SandboxResult:
    process = subprocess.Popen(
        argv,
        cwd=workspace,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
        return SandboxResult(stdout=stdout, exit_code=process.returncode)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
        return SandboxResult(stdout=stdout, exit_code=process.returncode, timed_out=True)
