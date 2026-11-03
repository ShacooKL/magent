#!/usr/bin/env python3
"""Convert Earth-Bench runner logs into a leakage-safe layered trajectory packet."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from rsagent.benchmarks.earth_bench_runner import load_json, select_evaluation
from rsagent.trajectory.layered_trajectory import GROUPS, MAX_STORE, TrajectoryView
from rsagent.evolution.tool_registry import canonical, safe_bytes, strict_json


def parse_question_selection(value: str | None) -> set[str] | None:
    if value is None:
        return None
    selected: set[str] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Empty question selector")
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) != 2 or not all(piece.strip().isdigit() for piece in pieces):
                raise ValueError(f"Invalid question range: {part!r}")
            start, end = (int(piece) for piece in pieces)
            if start < 1 or end < start:
                raise ValueError(f"Invalid question range: {part!r}")
            selected.update(str(number) for number in range(start, end + 1))
        elif part.isdigit() and int(part) >= 1:
            selected.add(str(int(part)))
        else:
            raise ValueError(f"Invalid question id: {part!r}")
    return selected


def safe_question(question_id: str, item: dict[str, Any], mode: str) -> dict[str, Any]:
    evaluation, data = select_evaluation(item, mode)
    question = evaluation.get("question")
    choices = item.get("choices") or []
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Question {question_id} has no usable text")
    if not isinstance(choices, list) or not all(
        isinstance(choice, (str, int, float)) and not isinstance(choice, bool)
        for choice in choices
    ):
        raise ValueError(f"Question {question_id} has invalid choices")
    return {
        "question_id": question_id,
        "mode": mode,
        "question": question.strip(),
        "choices": choices,
        "data": str(data),
    }


def load_rows(
    results_files: list[Path],
    selection: set[str] | None,
    allow_repeated_questions: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    modes: set[str] = set()
    for results_path in results_files:
        document = load_json(results_path)
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            raise ValueError(f"Invalid Earth-Bench results manifest: {results_path}")
        mode = document.get("mode")
        if mode not in {"autonomous_planning", "instruction_following"}:
            raise ValueError(f"Invalid or missing mode in {results_path}")
        modes.add(mode)
        for value in document["results"]:
            if not isinstance(value, dict) or "question_id" not in value:
                raise ValueError(f"Invalid result row in {results_path}")
            question_id = str(value["question_id"])
            if selection is not None and question_id not in selection:
                continue
            if question_id in seen and not allow_repeated_questions:
                raise ValueError(f"Question {question_id} occurs in more than one results manifest")
            seen[question_id] = seen.get(question_id, 0) + 1
            rows.append({
                "result": value,
                "results_source": str(results_path),
                "mode": mode,
                "attempt_index": seen[question_id],
            })
    if len(modes) != 1:
        raise ValueError("All selected results manifests must use the same Earth-Bench mode")
    if selection is not None:
        missing = sorted(selection - set(seen), key=int)
        if missing:
            raise ValueError("Selected questions missing from results: " + ", ".join(missing))
    if not rows:
        raise ValueError("No Earth-Bench results selected")
    rows.sort(key=lambda row: (int(str(row["result"]["question_id"])), row["attempt_index"]))
    totals = Counter(str(row["result"]["question_id"]) for row in rows)
    for row in rows:
        question_id = str(row["result"]["question_id"])
        row["episode_id"] = (
            f"q{question_id}_a{row['attempt_index']}" if totals[question_id] > 1 else f"q{question_id}"
        )
    return rows, modes.pop()


def load_events(path: Path) -> list[tuple[int, dict[str, Any]]]:
    events = []
    for line_number, line in enumerate(safe_bytes(path, 64 * 1024 * 1024).decode().splitlines(), 1):
        value = strict_json(line)
        if not isinstance(value, dict):
            raise ValueError(f"Non-object event at {path}:{line_number}")
        events.append((line_number, value))
    return events


def build_earthbench_trajectory(
    directory: Path,
    results_files: list[Path],
    question_file: Path,
    question_ids: set[str] | None = None,
    allow_repeated_questions: bool = False,
) -> dict[str, Any]:
    """Build layered_trajectory_v2 without importing answers or API response text."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"Trajectory destination already exists: {directory}")
    results_files = [Path(path).resolve(strict=True) for path in results_files]
    question_file = Path(question_file).resolve(strict=True)
    questions = load_json(question_file)
    if not isinstance(questions, dict):
        raise ValueError("Earth-Bench question file must contain an object")
    rows, mode = load_rows(results_files, question_ids, allow_repeated_questions)

    records: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    nodes: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    session_ids = [row["episode_id"] for row in rows]
    packet_task = {
        "kind": "earthbench_historical_solve_packet",
        "mode": mode,
        "question_ids": [str(row["result"]["question_id"]) for row in rows],
        "question_count": len(rows),
        "note": "Historical solve evidence only; no expected answers or ground-truth tool trajectories.",
    }
    records["task"] = {
        "text": canonical(packet_task),
        "source": ", ".join(map(str, results_files)),
        "scope": {"kind": "shared_packet", "session_ids": session_ids},
    }

    def record(evidence_id: str, value: Any, source: str, scope: dict[str, Any]) -> str:
        records[evidence_id] = {"text": canonical(value), "source": source, "scope": scope}
        return evidence_id

    missing_logs = 0
    imported_calls = 0
    imported_results = 0
    for row in rows:
        result = row["result"]
        question_id = str(result["question_id"])
        episode_id = row["episode_id"]
        session_name = (
            f"question{question_id}-attempt{row['attempt_index']}"
            if episode_id != f"q{question_id}" else f"question{question_id}"
        )
        scope = {"kind": "session", "session_ids": [episode_id], "session_name": session_name}
        if question_id not in questions or not isinstance(questions[question_id], dict):
            raise ValueError(f"Question {question_id} is absent from {question_file}")
        task_id = f"{episode_id}_task"
        record(task_id, safe_question(question_id, questions[question_id], mode), str(question_file), scope)

        trajectory_value = result.get("trajectory_log")
        trajectory_path = Path(trajectory_value).resolve() if isinstance(trajectory_value, str) and trajectory_value else None
        events: list[tuple[int, dict[str, Any]]] = []
        if trajectory_path is not None and trajectory_path.is_file():
            events = load_events(trajectory_path)
        else:
            missing_logs += 1

        session_start = next((event for _, event in events if event.get("type") == "session_start"), {})
        start = {
            key: session_start[key]
            for key in ("model", "profile", "provider", "max_steps")
            if key in session_start
        }
        start.update({"phase": "solve", "task_id": question_id, "mode": mode})
        start_id = record(
            f"{episode_id}_start",
            start,
            f"{trajectory_path}:session_start" if trajectory_path and events else row["results_source"],
            scope,
        )

        calls: dict[tuple[Any, Any], dict[str, Any]] = {}
        paired_results: dict[tuple[Any, Any], tuple[str, dict[str, Any]]] = {}
        pending_nodes: list[dict[str, Any]] = []
        for line_number, event in events:
            if event.get("type") not in {"tool_call", "tool_result"}:
                continue
            call_id = event.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"Tool event needs a call ID at {trajectory_path}:{line_number}")
            key = (event.get("turn_index", 0), call_id)
            evidence_id = f"{episode_id}_e{line_number}"
            record(evidence_id, event, f"{trajectory_path}:{line_number}", scope)
            if event["type"] == "tool_result":
                if key not in calls or key in paired_results or calls[key]["name"] != event.get("name"):
                    raise ValueError(f"Unpaired, duplicate or mismatched result at {trajectory_path}:{line_number}")
                paired_results[key] = (evidence_id, event)
                imported_results += 1
                continue

            if key in calls:
                raise ValueError(f"Duplicate tool call at {trajectory_path}:{line_number}")
            name = event.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Tool call needs a name at {trajectory_path}:{line_number}")
            category = GROUPS.get(name, "execute")
            if not pending_nodes or pending_nodes[-1]["category"] != category:
                pending_nodes.append({
                    "node_id": f"{episode_id}_n{len(pending_nodes) + 1}",
                    "episode_id": episode_id,
                    "category": category,
                    "actions": [],
                })
            action = {
                "action_id": evidence_id,
                "name": name,
                "step": event.get("step_index"),
                "call_evidence_id": evidence_id,
                "result_evidence_id": None,
                "outcome": "missing_result",
                "artifact_ids": [],
            }
            arguments = event.get("arguments") or {}
            if name == "write_file" and isinstance(arguments, dict) and isinstance(arguments.get("content"), str):
                artifact_id = evidence_id + "_text"
                artifacts[artifact_id] = {
                    "text": arguments["content"],
                    "evidence_id": evidence_id,
                    "kind": "submitted_write_content",
                    "warning": "Requested file contents, not proof of successful write or execution.",
                    "scope": scope,
                }
                action["artifact_ids"].append(artifact_id)
            calls[key] = action
            pending_nodes[-1]["actions"].append(action)
            imported_calls += 1

        for key, action in calls.items():
            if key in paired_results:
                result_id, event = paired_results[key]
                action["result_evidence_id"] = result_id
                action["outcome"] = (
                    "returned_success" if event.get("success") is True else
                    "returned_failure" if event.get("success") is False else
                    "unknown"
                )
        nodes.extend(pending_nodes)

        evaluation = next((event for _, event in reversed(events) if event.get("type") == "evaluation"), {})
        ending = {
            "status": result.get("status"),
            "correct": result.get("correct"),
            "predicted": result.get("predicted"),
            "termination_reason": evaluation.get("termination_reason"),
            "duration_seconds": result.get("duration_seconds"),
            "timed_out": result.get("timed_out"),
            "returncode": result.get("returncode"),
            "tool_calls": len(calls),
            "tool_results": len(paired_results),
            "trajectory_available": bool(events),
        }
        end_id = record(
            f"{episode_id}_end",
            ending,
            f"{trajectory_path}:evaluation" if trajectory_path and events else row["results_source"],
            scope,
        )
        episodes.append({
            "episode_id": episode_id,
            "name": session_name,
            "task_evidence_id": task_id,
            "node_ids": [node["node_id"] for node in pending_nodes],
            "start": start,
            "start_evidence_id": start_id,
            "ending": ending,
            "ending_evidence_id": end_id,
        })

    payload = {
        "schema": "layered_trajectory_v2",
        "builder": "earthbench_jsonl_adapter_v1",
        "task": {
            "task_id": "earthbench_packet",
            "question": "Historical Earth-Bench solve trajectories selected by the host.",
            "input_count": len(rows),
            "evidence_id": "task",
        },
        "episodes": episodes,
        "nodes": nodes,
        "records": records,
        "artifacts": artifacts,
        "boundaries": [
            "Authorized task text, choices, tool calls/results, terminal predictions and correctness labels only.",
            "Expected answers, gt_answer fields, official reference trajectories, API requests/responses and model reasoning are excluded.",
            "A correct final choice does not prove every intermediate claim or tool call was correct.",
            "An incorrect final choice does not identify a cause; tool evidence must be inspected before drawing conclusions.",
            "Missing trajectories contain terminal runner metadata only and cannot support claims about Agent tool behavior.",
        ],
    }
    encoded = canonical(payload).encode()
    if len(encoded) > MAX_STORE:
        raise ValueError(
            f"Layered trajectory is {len(encoded)} bytes, above the {MAX_STORE}-byte limit; "
            "select a smaller question packet"
        )
    directory.mkdir(parents=True)
    (directory / "trajectory.json").write_bytes(encoded)
    return {
        "questions": len(rows),
        "unique_questions": len({str(row['result']['question_id']) for row in rows}),
        "repeated_attempts": len(rows) - len({str(row['result']['question_id']) for row in rows}),
        "episodes": len(episodes),
        "nodes": len(nodes),
        "evidence_records": len(records),
        "artifacts": len(artifacts),
        "tool_calls": imported_calls,
        "tool_results": imported_results,
        "missing_trajectory_logs": missing_logs,
        "correct": sum(row["result"].get("correct") is True for row in rows),
        "incorrect_or_error": sum(row["result"].get("correct") is not True for row in rows),
        "trajectory_bytes": len(encoded),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", type=Path, required=True,
                        help="Earth-Bench results.json; repeat to combine disjoint runs")
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory that will contain trajectory.json and inspection summaries")
    parser.add_argument("--questions", help="Optional IDs/ranges, for example 1:20,72,90")
    parser.add_argument(
        "--allow-repeated-questions",
        action="store_true",
        help="Import repeated attempts of the same question as distinct episode IDs",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    report = build_earthbench_trajectory(
        output / "private-store",
        args.results,
        args.question_file,
        parse_question_selection(args.questions),
        args.allow_repeated_questions,
    )
    view = TrajectoryView(output / "private-store")
    overview = view.overview()
    (output / "overview.json").write_text(canonical(overview) + "\n")
    (output / "build-report.json").write_text(canonical(report) + "\n")
    print(canonical(report))


if __name__ == "__main__":
    main()
