"""Run a candidate's JSON interface in the existing namespace sandbox.

This checks executability, not scientific correctness or task utility. Mounts,
runtime and time budget belong to the host; only arguments reach candidate main.
"""

from pathlib import Path
import shlex
import sys
from time import monotonic
from uuid import uuid4

from rsagent.evolution.tool_registry import LIMIT, canonical, safe_bytes, strict_json
from rsagent.runtime.sandbox import NamespaceRunner


def execute_tool(source, arguments, *, workspace, readonly=(), protected_roots=(),
                 timeout=30, runtime_prefix=None, inspect_only=False):
    """Inspect or execute ``main(arguments) -> JSON`` without importing on host.

    Protected stores may provide the explicit source file, never directory or
    runtime mounts. A successful inspection does not call main or validate its
    return value; an ordinary call validates JSON transport, not its meaning.
    """
    source = Path(source)
    safe_bytes(source)
    source = source.resolve(strict=True)
    workspace = Path(workspace).resolve(strict=True)
    readonly = [Path(path).resolve(strict=True) for path in readonly]
    runtime_prefix = Path(runtime_prefix or sys.prefix).resolve(strict=True)
    exposed = [workspace, *readonly, runtime_prefix,
               *[Path(path).resolve() for path in ("/usr", "/bin", "/lib", "/lib64")]]
    for protected in map(Path, protected_roots):
        protected = protected.resolve()
        if any(path.is_relative_to(protected) or protected.is_relative_to(path)
               for path in exposed):
            raise ValueError("Execution mounts cannot expose protected storage")
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    if not isinstance(arguments, dict):
        raise ValueError("Arguments must be a JSON object")
    encoded = canonical(arguments)
    if len(encoded.encode()) > LIMIT:
        raise ValueError("Arguments exceed the JSON size limit")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer")

    request = workspace / f".trial-{uuid4().hex}.input.json"
    response = workspace / f".trial-{uuid4().hex}.output.json"
    request.write_text(encoded, encoding="utf-8")
    inside_source = (Path("/workspace") / source.relative_to(workspace)
                     if source.is_relative_to(workspace) else source)
    script = (
        "import inspect,json,runpy; "
        f"module=runpy.run_path({str(inside_source)!r}); "
        "entry=module.get('main'); "
        "assert callable(entry), 'Tool must define callable main(arguments)'; "
        "inspect.signature(entry).bind({}); "
        f"arguments=json.load(open({request.name!r},encoding='utf-8')); "
        + ("value={'callable_main':True,'accepts_arguments':True}; " if inspect_only
           else "value=entry(arguments); ")
        + f"open({response.name!r},'x',encoding='utf-8').write(json.dumps(value,allow_nan=False))"
    )
    runner = NamespaceRunner(read_only_paths=[source, *readonly], runtime_prefix=runtime_prefix)
    started = monotonic()
    report = {"success": False, "value": None, "exit_code": None,
              "timed_out": False, "stdout": ""}
    try:
        result = runner.run(workspace, "python -c " + shlex.quote(script), {}, timeout)
        report.update(exit_code=result.exit_code, timed_out=result.timed_out,
                      stdout=result.stdout[-4000:])
        if result.exit_code == 0 and not result.timed_out:
            report["value"] = strict_json(safe_bytes(response).decode("utf-8"))
            report["success"] = True
        else:
            report["error"] = "Tool timed out" if result.timed_out else "Sandbox execution failed"
    except (OSError, ValueError, UnicodeError, RuntimeError) as exc:
        report["error"] = str(exc)
    finally:
        report["duration_ms"] = round((monotonic() - started) * 1000)
        for path in (request, response):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass  # A candidate may have removed or replaced its scratch file.
    return report
