"""Adapter for exposing Earth-Agent's FastMCP tools to the minimal agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
from typing import Any, Callable
import uuid

from rsagent.runtime.sandbox import NamespaceRunner


class EarthToolProvider:
    def __init__(
        self,
        config: dict[str, Any],
        workspace: Path,
        read_only_paths: list[Path],
        approve: Callable[[str], bool],
        source_workspace: Path | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.approve = approve
        self.timeout = int(config.get("timeout_seconds", 300))
        self.tools_dir = Path(config["tools_dir"]).expanduser().resolve(strict=True)
        self.python = Path(config["python"]).expanduser().resolve(strict=True)
        output_setting = Path(config.get("output_dir", ".earth-tools/output")).expanduser()
        self.output_dir = (
            output_setting if output_setting.is_absolute() else self.workspace / output_setting
        ).resolve()
        if not self.output_dir.is_relative_to(self.workspace):
            raise ValueError("earth_tools.output_dir must be inside the writable workspace")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sandbox_output_dir = Path('/diagnostics' if source_workspace else '/workspace') / self.output_dir.relative_to(self.workspace)

        bridge = Path(__file__).with_name("earth_tools_bridge.py").resolve(strict=True)
        python_prefix = self.python.parent.parent
        earth_root = self.tools_dir.parent.parent
        model_results_setting = config.get(
            "model_results_file",
            earth_root / "benchmark" / "model_results.csv",
        )
        model_results_file = (
            Path(model_results_setting).expanduser().resolve(strict=True)
            if model_results_setting
            else None
        )
        internal_mounts = [earth_root, python_prefix, bridge.parent, *read_only_paths]
        unique_mounts = list(dict.fromkeys(path.resolve() for path in internal_mounts))
        self.runner = NamespaceRunner(read_only_paths=unique_mounts, source_workspace=source_workspace)
        command_parts = [
            "OPENBLAS_NUM_THREADS=1",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "PYTHONDONTWRITEBYTECODE=1",
            shlex.quote(str(self.python)),
            shlex.quote(str(bridge)),
            "--tools-dir",
            shlex.quote(str(self.tools_dir)),
            "--temp-dir",
            shlex.quote(str(sandbox_output_dir)),
        ]
        if model_results_file:
            command_parts.extend([
                "--model-results-file",
                shlex.quote(str(model_results_file)),
            ])
        command = " ".join(command_parts)
        self.process = self.runner.spawn(
            self.workspace,
            command,
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "TZ": "Asia/Shanghai"},
        )
        try:
            ready = self._read_response(60)
            if ready.get("type") != "ready" or not isinstance(ready.get("tools"), list):
                raise RuntimeError(f"Invalid Earth tools bridge startup response: {ready}")
            self.definitions = ready["tools"]
            self.names = {tool["name"] for tool in self.definitions}
            self.precomputed_results = int(ready.get("precomputed_results", 0))
        except BaseException:
            # Until construction returns, no caller can own/close this process.
            self.close()
            raise

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self.names:
            raise ValueError(f"Unknown Earth tool: {name}")
        if not self.approve(f"Run Earth tool {name} in namespace sandbox?"):
            return "Denied by user"
        request_id = uuid.uuid4().hex
        if not self.process.stdin:
            raise RuntimeError("Earth tools bridge stdin is unavailable")
        self.process.stdin.write(json.dumps({
            "id": request_id,
            "name": name,
            "arguments": arguments,
        }, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self._read_response(self.timeout)
        if response.get("id") != request_id:
            raise RuntimeError("Earth tools bridge returned a mismatched response")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Earth tool failed"))
        return str(response.get("result", ""))

    def _read_response(self, timeout: int) -> dict[str, Any]:
        if not self.process.stdout:
            raise RuntimeError("Earth tools bridge stdout is unavailable")
        readable, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not readable:
            if self.process.poll() is not None:
                raise RuntimeError(f"Earth tools bridge exited with code {self.process.returncode}")
            raise TimeoutError(f"Earth tools bridge timed out after {timeout} seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Earth tools bridge exited with code {self.process.poll()}")
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid Earth tools bridge response: {line[:500]}") from error

    def close(self) -> None:
        process = getattr(self, "process", None)
        if not process or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
