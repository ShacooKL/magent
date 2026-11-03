"""Separate path argument shape diagnostics from sandbox access policy.

Never coerce arguments. This is not a replacement for the tool's full schema
validator: only declared path containers are checked before path authorization.
"""
import json
from pathlib import Path


class ToolArgumentError(ValueError):
    """Malformed path argument, not an access violation."""


class ToolPathAccessError(ValueError):
    """A path lies outside the permitted task/session boundary."""


def _path_key(key):
    return any(token in key.lower() for token in ("path", "file", "save_name"))


def _types(schema):
    result = set()
    value = schema.get("type")
    if isinstance(value, str):
        result.add(value)
    elif isinstance(value, list):
        result.update(value)
    for keyword in ("anyOf", "oneOf"):
        for branch in schema.get(keyword, []):
            result.update(_types(branch))
    return result


def _child_schema(schema, keyword, key=None):
    """Use only unambiguous child declarations; delegate other validation."""
    if keyword == "properties":
        return schema.get("properties", {}).get(key, {})
    if isinstance(schema.get(keyword), dict):
        return schema[keyword]
    candidates = [branch[keyword] for option in ("anyOf", "oneOf")
                  for branch in schema.get(option, []) if isinstance(branch.get(keyword), dict)]
    return candidates[0] if candidates and all(c == candidates[0] for c in candidates) else {}


def validate_path_argument_shapes(arguments, schema=None, *, tool_name="tool"):
    """Report stringified lists and path container type errors without mutation."""
    if not isinstance(arguments, dict):
        raise ToolArgumentError(f"Invalid arguments for {tool_name}: expected a JSON object.")

    def walk(value, spec, field, is_path=False):
        allowed = _types(spec)
        if is_path and isinstance(value, str) and value.lstrip().startswith(("[", "{")):
            try:
                encoded = json.loads(value)
            except ValueError:
                encoded = None
            if isinstance(encoded, (list, dict)):
                container = "array" if isinstance(encoded, list) else "object"
                if not allowed or container in allowed:
                    hint = ('pass an actual JSON array, e.g. ["/path/one.tif", "/path/two.tif"], '
                            'without quotes around the array' if container == "array" else
                            'pass an actual JSON object, without quotes around the object')
                else:
                    hint = "pass a single file path; this field does not declare support for that container"
                raise ToolArgumentError(
                    f"Invalid argument {tool_name}.{field}: received a JSON-encoded {container} as a string; "
                    f"{hint}. No argument was converted and the official tool has not run.")
        # Restrict this check to declared path-container types, not domain values.
        actual = ("string" if isinstance(value, str) else "array" if isinstance(value, list) else
                  "object" if isinstance(value, dict) else "null" if value is None else "other")
        if is_path and allowed and actual not in allowed:
            raise ToolArgumentError(
                f"Invalid argument {tool_name}.{field}: expected {' or '.join(sorted(allowed))}, "
                f"received {actual}. No argument was converted and the official tool has not run.")
        if isinstance(value, dict):
            for key, child in value.items():
                child_spec = _child_schema(spec, "properties", key)
                if not child_spec:
                    child_spec = _child_schema(spec, "additionalProperties")
                walk(child, child_spec, f"{field}.{key}" if field else key, is_path or _path_key(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, _child_schema(spec, "items"), f"{field}[{index}]", is_path)

    walk(arguments, schema or {}, "")


def _path_entries(arguments):
    def walk(value, field="", is_path=False, output=False):
        if isinstance(value, dict):
            for key, child in value.items():
                child_is_path = is_path or _path_key(key)
                child_output = output or (not is_path and (key.startswith("output") or key == "save_name"))
                yield from walk(child, f"{field}.{key}" if field else key, child_is_path, child_output)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, f"{field}[{index}]", is_path, output)
        elif isinstance(value, str) and is_path:
            yield field, value, output
    yield from walk(arguments)


def check_tool_path_access(arguments, data_root, output_root, workspace, *, allowed_output_suffixes=None):
    """Authorize paths without changing them or diagnosing argument schemas."""
    workspace = Path(workspace).resolve()
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    if not output_root.is_relative_to(workspace):
        raise ValueError("Configured output root must be inside the workspace")
    for field, value, output in _path_entries(arguments):
        path = Path(value)
        def deny(reason):
            raise ToolPathAccessError(f"Path access denied for {field}: {reason}. Official tool has not run.")
        if ".." in path.parts or "\x00" in value or "://" in value:
            deny("parent traversal, NUL bytes and URLs are not permitted")
        if path.is_relative_to("/workspace"):
            resolved = (workspace / path.relative_to("/workspace")).resolve()
        elif path.is_absolute():
            resolved = path.resolve()
        elif output:
            resolved = (output_root / path).resolve()
        else:
            deny("input paths must be absolute within this task's data or session output")
        roots = (output_root,) if output else (data_root, output_root)
        if not any(resolved.is_relative_to(root) for root in roots):
            deny("outside the authorized task data/session output boundary (including symlink targets)")
        if output and allowed_output_suffixes is not None and path.suffix.lower() not in allowed_output_suffixes:
            raise ToolPathAccessError(
                f"Output policy denied for {field}: this experiment permits only data artifact file types. "
                "Official tool has not run.")
