#!/usr/bin/env python3
"""JSONL bridge that loads Earth-Agent FastMCP tools in its own environment."""

from __future__ import annotations

import argparse
import ast
import asyncio
import contextlib
import csv
import importlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any


MODULES = ("Analysis", "Index", "Inversion", "Perception", "Statistics")
PRECOMPUTED_TOOLS = {
    "MSCN": "MSCN",
    "RemoteCLIP": "RemoteCLIP",
    "Strip_R_CNN": "Strip-R-CNN",
    "SM3Det": "SM3Det",
    "RemoteSAM": "RemoteSAM",
    "InstructSAM": "InstructSAM",
    "SAM2": "SAM2",
    "ChangeOS": "ChangeOS",
}


class PrecomputedModelResults:
    """Resolve Earth-Bench perception calls from the released lookup table."""

    def __init__(self, path: Path, earth_root: Path) -> None:
        self.path = path.resolve(strict=True)
        self.earth_root = earth_root.resolve(strict=True)
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            self.rows = list(csv.DictReader(stream, delimiter=";"))
        required = {"file_path", "model", "text_prompt", "bbox", "result"}
        if not self.rows or not required.issubset(self.rows[0]):
            raise ValueError(f"Invalid precomputed model results file: {self.path}")

    @staticmethod
    def normalize_image_path(value: str) -> str:
        value = str(value).strip().strip('"').strip("'")
        if value.startswith("benchmark/data/"):
            return Path(value).as_posix()
        path = Path(value)
        question = next(
            (part for part in reversed(path.parts[:-1]) if part.startswith("question")),
            None,
        )
        if not question:
            raise ValueError(f"Cannot identify Earth-Bench question in image path: {value}")
        return f"benchmark/data/{question}/{path.name}"

    @staticmethod
    def _comparable(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return " ".join(value.lower().replace("_", " ").replace("-", " ").split())

    def lookup(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        model_name = PRECOMPUTED_TOOLS[tool_name]
        if tool_name == "ChangeOS":
            pre_path = self.normalize_image_path(arguments["pre_image_path"])
            post_path = self.normalize_image_path(arguments["post_image_path"])
            file_path = pre_path if pre_path == post_path else f"({pre_path}, {post_path})"
        else:
            file_path = self.normalize_image_path(arguments["input_image_path"])

        candidates = [
            row for row in self.rows
            if row["model"] == model_name and row["file_path"] == file_path
        ]
        for field in ("text_prompt", "bbox"):
            supplied = arguments.get(field)
            if supplied is None or not candidates:
                continue
            qualified = [row for row in candidates if row[field].strip()]
            if not qualified:
                continue
            exact = [
                row for row in qualified
                if self._comparable(row[field]) == self._comparable(supplied)
            ]
            if exact:
                candidates = exact
            else:
                raise LookupError(
                    f"No precomputed {model_name} result for {file_path} "
                    f"with {field}={supplied!r}"
                )
        if not candidates:
            raise LookupError(
                f"No precomputed {model_name} result for {file_path}; "
                "use the exact image path and prompt from the Earth-Bench task"
            )
        if len(candidates) > 1:
            available = [
                row["text_prompt"] or row["bbox"] or "(no qualifier)"
                for row in candidates
            ]
            raise LookupError(
                f"Ambiguous precomputed {model_name} result for {file_path}; "
                f"available qualifiers: {available}"
            )

        raw_result = candidates[0]["result"].strip()
        if not raw_result:
            raise LookupError(f"Precomputed {model_name} result is empty for {file_path}")
        try:
            result = ast.literal_eval(raw_result)
        except (SyntaxError, ValueError):
            result = raw_result
        if isinstance(result, str) and result.startswith("benchmark/model_out/"):
            artifact = self.earth_root / result
            if not artifact.exists():
                raise FileNotFoundError(
                    f"Precomputed artifact was not released: {result}; "
                    "this task requires the Earth-Agent online_infer models"
                )
            return str(artifact)
        return result


def json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def tool_result_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(json.dumps(block, ensure_ascii=False, default=json_default))
    if parts:
        return "\n".join(parts)
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, default=json_default)
    return json.dumps(result, ensure_ascii=False, default=json_default)


async def load_tools(tools_dir: Path, temp_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(tools_dir))
    sys.argv = [sys.argv[0], "--temp_dir", str(temp_dir)]
    tools: dict[str, Any] = {}
    for module_name in MODULES:
        with contextlib.redirect_stdout(sys.stderr):
            module = importlib.import_module(module_name)
            module_tools = await module.mcp.get_tools()
        overlap = set(tools) & set(module_tools)
        if overlap:
            raise RuntimeError(f"Duplicate Earth tool names: {sorted(overlap)}")
        tools.update(module_tools)
    return tools


def run_tool_with_diagnostics(tool: Any, arguments: dict[str, Any]) -> str:
    """Preserve printed diagnostics, including errors swallowed by official tools."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            result = asyncio.run(tool.run(arguments))
    except Exception as error:
        diagnostics = captured.getvalue()
        if diagnostics:
            raise RuntimeError(
                f"{type(error).__name__}: {error}\n"
                f"Tool diagnostics (stdout/stderr):\n{diagnostics}"
            ) from error
        raise
    result_text = tool_result_text(result)
    diagnostics = captured.getvalue()
    if diagnostics:
        result_text += "\n\nTool diagnostics (stdout/stderr):\n" + diagnostics
    return result_text


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=json_default), flush=True)


async def initialize(args: argparse.Namespace) -> dict[str, Any]:
    tools_dir = args.tools_dir.resolve(strict=True)
    temp_dir = args.temp_dir.resolve(strict=True)
    os.chdir(tools_dir.parent.parent)
    return await load_tools(tools_dir, temp_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, required=True)
    parser.add_argument("--model-results-file", type=Path)
    args = parser.parse_args()
    tools = asyncio.run(initialize(args))
    precomputed = None
    if args.model_results_file:
        precomputed = PrecomputedModelResults(
            args.model_results_file,
            args.tools_dir.resolve(strict=True).parent.parent,
        )

    emit({
        "type": "ready",
        "precomputed_results": len(precomputed.rows) if precomputed else 0,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description or f"Earth-Agent tool {tool.name}",
                "input_schema": tool.parameters,
            }
            for tool in tools.values()
        ],
    })

    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            name = request["name"]
            if name not in tools:
                raise ValueError(f"Unknown Earth tool: {name}")
            if precomputed and name in PRECOMPUTED_TOOLS:
                result_text = json.dumps(
                    precomputed.lookup(name, request.get("arguments", {})),
                    ensure_ascii=False,
                    default=json_default,
                )
            else:
                result_text = run_tool_with_diagnostics(tools[name], request.get("arguments", {}))
            emit({"id": request_id, "ok": True, "result": result_text})
        except Exception as error:
            emit({
                "id": request_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            })


if __name__ == "__main__":
    main()
