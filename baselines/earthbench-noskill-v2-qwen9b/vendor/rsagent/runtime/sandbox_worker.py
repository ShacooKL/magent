#!/usr/bin/env python3
"""Trusted helper that constructs a minimal root inside a mount namespace."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile


def _run(*arguments: str) -> None:
    subprocess.run(arguments, check=True, stdin=subprocess.DEVNULL)


def _bind(source: Path, target: Path, *, read_only: bool) -> None:
    source = source.resolve()
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch(exist_ok=True)
    _run("mount", "--bind", str(source), str(target))
    if read_only:
        _run("mount", "-o", "remount,bind,ro", str(target))


def _inside(root: Path, absolute: Path) -> Path:
    # Preserve lexical mount locations such as /bin even when the host uses
    # merged-/usr symlinks. Resolving the source and destination identically
    # would leave /bin absent inside the new root.
    return root / Path(os.path.abspath(absolute)).relative_to("/")


def _drop_capabilities() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    pr_capbset_drop = 24
    for capability in range(64):
        libc.prctl(pr_capbset_drop, capability, 0, 0, 0)

    class CapHeader(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class CapData(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    header = CapHeader(0x20080522, 0)  # _LINUX_CAPABILITY_VERSION_3
    data = (CapData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    pr_set_no_new_privs = 38
    if libc.prctl(pr_set_no_new_privs, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _set_limits(cpu_seconds: int) -> None:
    limits = (
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3)),
        (resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2)),
        (resource.RLIMIT_NOFILE, (1024, 1024)),
        (resource.RLIMIT_NPROC, (256, 256)),
        (resource.RLIMIT_CORE, (0, 0)),
    )
    for kind, value in limits:
        resource.setrlimit(kind, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument('--source-workspace', type=Path)
    parser.add_argument("--read-only-path", type=Path, action="append", default=[])
    parser.add_argument("--cpu-seconds", type=int, default=125)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if len(command) != 1:
        parser.error("exactly one shell command is required")

    workspace = args.workspace.resolve(strict=True)
    source_workspace = args.source_workspace.resolve(strict=True) if args.source_workspace else None
    read_only_paths = [path.resolve(strict=True) for path in args.read_only_path]
    runtime_prefix = Path(os.environ["PY_AGENT_RUNTIME_PREFIX"]).resolve(strict=True)
    _run("mount", "--make-rprivate", "/")

    with tempfile.TemporaryDirectory(prefix="py-agent-root-") as root_name:
        root = Path(root_name)
        _run("mount", "-t", "tmpfs", "-o", "nodev,nosuid,size=512m", "tmpfs", str(root))
        try:
            for directory in ("workspace", "tmp", "proc", "dev", "etc"):
                (root / directory).mkdir(parents=True, exist_ok=True)

            mounted: set[Path] = set()
            for runtime in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), runtime_prefix):
                resolved = runtime.resolve()
                if runtime.exists() and resolved not in mounted:
                    _bind(runtime, _inside(root, runtime), read_only=True)
                    mounted.add(resolved)

            if source_workspace:
                _bind(source_workspace, root / 'workspace', read_only=True)
                _bind(workspace, root / 'diagnostics', read_only=False)
            else:
                _bind(workspace, root / "workspace", read_only=False)
            for path in read_only_paths:
                _bind(path, _inside(root, path), read_only=True)
            for device in (Path("/dev/null"), Path("/dev/zero"), Path("/dev/random"), Path("/dev/urandom")):
                if device.exists():
                    _bind(device, root / device.relative_to("/"), read_only=False)
            _run("mount", "-t", "proc", "-o", "nosuid,nodev,noexec", "proc", str(root / "proc"))
            _run("mount", "-t", "tmpfs", "-o", "nodev,nosuid,size=256m", "tmpfs", str(root / "tmp"))

            (root / "etc" / "passwd").write_text("sandbox:x:0:0:sandbox:/tmp:/bin/sh\n", encoding="utf-8")
            (root / "etc" / "group").write_text("sandbox:x:0:\n", encoding="utf-8")

            os.chroot(root)
            os.chdir('/diagnostics' if source_workspace else '/workspace')
            _set_limits(args.cpu_seconds)
            _drop_capabilities()
            prefix_inside = "/" + str(runtime_prefix.relative_to("/"))
            environment = {
                "HOME": "/tmp",
                "PATH": f"{prefix_inside}/bin:/usr/local/bin:/usr/bin:/bin",
                "TMPDIR": "/tmp",
                "TZ": "Asia/Shanghai",
                # Geospatial Python environments commonly link NumPy against
                # OpenBLAS/MKL. Keep imports and small raster jobs within the
                # sandbox's process and memory limits.
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            for key in ("LANG", "LC_ALL", "TERM"):
                if os.environ.get(key):
                    environment[key] = os.environ[key]
            os.execve("/bin/bash", ["bash", "-c", command[0]], environment)
        finally:
            # Normally unreachable after exec. Namespace teardown drops mounts.
            pass


if __name__ == "__main__":
    main()
