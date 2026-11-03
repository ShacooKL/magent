#!/usr/bin/env python3
"""Run selected Earth-Bench questions through the minimal coding agent."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import selectors
import signal
import subprocess
import sys
from threading import Event
from time import perf_counter
from typing import Any
from uuid import uuid4

from rsagent.runtime.agent import DEFAULT_CONFIG_PATH, load_config
from rsagent.runtime.clock import now_beijing
from rsagent.runtime.recorder import append_event
from rsagent.paths import REPOSITORY_ROOT, VAR_ROOT


ANSWER_PATTERN = re.compile(r"<Answer>\s*([A-F])\s*</Answer>", re.IGNORECASE)
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_question_ids(value: str) -> list[str]:
    ids: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part or not part.isdigit() or int(part) < 1:
            raise ValueError(f"Invalid question id: {part!r}")
        normalized = str(int(part))
        if normalized not in ids:
            ids.append(normalized)
    if not ids:
        raise ValueError("No question ids were selected")
    return ids


def parse_question_range(value: str) -> list[str]:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", value)
    if not match:
        raise ValueError("Range must use START:END, for example 1:10")
    start, end = map(int, match.groups())
    if start < 1 or end < start:
        raise ValueError("Range requires 1 <= START <= END")
    return [str(number) for number in range(start, end + 1)]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON file {path}: {error}") from error


def expected_answers(answer_data: Any) -> dict[str, str]:
    if not isinstance(answer_data, list):
        raise ValueError("Earth-Bench answer file must contain a JSON array")
    answers: dict[str, str] = {}
    for item in answer_data:
        if not isinstance(item, dict) or "id" not in item:
            continue
        answer = item.get("answer")
        if isinstance(answer, str):
            answers[str(item["id"])] = answer.strip().upper()
    return answers


def select_evaluation(item: dict[str, Any], mode: str) -> tuple[dict[str, Any], str]:
    evaluations = item.get("evaluation") or []
    keyword = "autonomous" if mode == "autonomous_planning" else "instruction"
    selected = next(
        (
            evaluation
            for evaluation in evaluations
            if keyword in str(evaluation.get("type", "")).lower()
        ),
        None,
    )
    if not selected:
        raise ValueError(f"Question has no {mode} evaluation")
    data = selected.get("data") or next(
        (evaluation.get("data") for evaluation in evaluations if evaluation.get("data")),
        None,
    )
    if not data:
        raise ValueError("Question has no data directory")
    return selected, str(data)


def build_prompt(
    question_id: str,
    item: dict[str, Any],
    mode: str,
    data_directory: str | None = None,
) -> str:
    evaluation, data = select_evaluation(item, mode)
    data = data_directory or data
    question = str(evaluation.get("question") or "").strip()
    if not question:
        raise ValueError("Question text is empty")
    choices = item.get("choices") or []
    choice_text = "\n".join(
        f"{chr(ord('A') + index)}. {choice}"
        for index, choice in enumerate(choices)
    )
    return (
        ("You are solving an Earth-Bench multiple-choice task as a geoscientist.\n"
         if choices else "You are solving an Earth-Bench numerical task as a geoscientist.\n")
        + ("Use the available Earth-Agent tools to inspect and calculate the data; do not guess "
           "from the choices and do not inspect answer or evaluation files.\n"
           if choices else "Use the available Earth-Agent tools to inspect and calculate the data; "
           "do not inspect answer or evaluation files.\n")
        + "Treat a successful Earth-Agent tool result as the primary evidence when its inputs "
        "match the task. Do not discard or replace that result merely because it seems "
        "implausible. Deviate only when you can identify concrete evidence of an input "
        "mismatch, a tool failure, or a conflict with the task definition, and state that "
        "evidence before using an alternative result.\n"
        "The Earth tools run with the Earth-Agent repository as their working directory. "
        f"The data directory is {data}.\n"
        "When a tool creates a file, choose an output_path under "
        f"question{question_id}/ and pass the exact returned path to later tools.\n"
        "If a tool fails, inspect the error and retry only when changing the arguments can fix it.\n"
        "Finish with exactly one line in the form <Answer>X</Answer>.\n\n"
        f"Question: {question}\n{choice_text}"
    )


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def update_summary(manifest: dict[str, Any], selected_count: int) -> None:
    completed = [
        item for item in manifest.get("results", [])
        if item.get("status") == "completed"
    ]
    correct = sum(item.get("correct") is True for item in completed)
    manifest["summary"] = {
        "selected": selected_count,
        "completed": len(completed),
        "correct": correct,
        "accuracy": round(correct / selected_count, 4),
        "completed_accuracy": round(correct / len(completed), 4) if completed else None,
        "completion_rate": round(len(completed) / selected_count, 4),
        "updated_at": now_beijing().isoformat(timespec="seconds"),
    }


def extract_answer(output: str) -> str | None:
    matches = ANSWER_PATTERN.findall(ANSI_PATTERN.sub("", output))
    return matches[-1].upper() if matches else None


def run_question(
    *,
    agent_path: Path,
    config_path: Path,
    profile: str | None,
    workspace: Path,
    prompt: str,
    max_steps: int,
    temperature: float | None,
    timeout: int,
    show_tools: bool,
    read_only_path: Path,
    console_path: Path,
    run_id: str,
    task_id: str,
    attempt_id: str,
    previous_attempt_id: str | None,
    task_fingerprint: str,
    stream_output: bool = True,
    cancel_event: Event | None = None,
) -> tuple[str, int, bool]:
    command = [
        sys.executable,
        str(agent_path),
        str(workspace),
        "--config",
        str(config_path),
        "--new-session",
        "--read-only-path",
        str(read_only_path),
        "--log-dir",
        str(workspace / "agent-logs"),
        "--max-steps",
        str(max_steps),
        "--show-tools" if show_tools else "--no-show-tools",
        "--prompt",
        prompt,
        "--trace-run-id",
        run_id,
        "--trace-task-id",
        task_id,
        "--trace-attempt-id",
        attempt_id,
        "--trace-task-fingerprint",
        task_fingerprint,
    ]
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if previous_attempt_id:
        command.extend(["--trace-previous-attempt-id", previous_attempt_id])
    if profile:
        command.extend(["--profile", profile])
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    output_parts: list[str] = []
    timed_out = False
    deadline = perf_counter() + timeout
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        with console_path.open("w", encoding="utf-8", buffering=1) as console:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                for key, _ in selector.select(timeout=min(0.5, remaining)):
                    line = key.fileobj.readline()
                    if line:
                        output_parts.append(line)
                        console.write(line)
                        if stream_output:
                            print(line, end="", flush=True)
            remainder = process.stdout.read()
            if remainder:
                output_parts.append(remainder)
                console.write(remainder)
                if stream_output:
                    print(remainder, end="", flush=True)
    except KeyboardInterrupt:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()
    return "".join(output_parts), process.returncode or 0, timed_out


def evaluate_question(
    *,
    question_id: str,
    prompt: str,
    question_data: Path,
    expected: str | None,
    previous_result: dict[str, Any] | None,
    previous_attempt_count: int,
    run_id: str,
    run_dir: Path,
    agent_path: Path,
    config_path: Path,
    profile: str | None,
    max_steps: int,
    temperature: float | None,
    timeout: int,
    show_tools: bool,
    stream_output: bool,
    cancel_event: Event | None,
) -> dict[str, Any]:
    """Run and score one question without mutating the shared manifest."""
    question_workspace = run_dir / f"question{question_id}"
    question_workspace.mkdir(parents=True, exist_ok=True)
    console_path = question_workspace / "console.log"
    previous_attempt_id = previous_result.get("attempt_id") if previous_result else None
    attempt_number = previous_attempt_count + 1
    attempt_id = f"{run_id}-q{question_id}-a{attempt_number}-{uuid4().hex[:8]}"
    task_fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    started = perf_counter()
    output, returncode, timed_out = run_question(
        agent_path=agent_path,
        config_path=config_path,
        profile=profile,
        workspace=question_workspace,
        prompt=prompt,
        max_steps=max_steps,
        temperature=temperature,
        timeout=timeout,
        show_tools=show_tools,
        read_only_path=question_data,
        console_path=console_path,
        run_id=run_id,
        task_id=question_id,
        attempt_id=attempt_id,
        previous_attempt_id=previous_attempt_id,
        task_fingerprint=task_fingerprint,
        stream_output=stream_output,
        cancel_event=cancel_event,
    )
    duration = round(perf_counter() - started, 3)
    predicted = extract_answer(output)
    trajectory_logs = sorted(
        (question_workspace / "agent-logs").glob("*.jsonl"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    status = "completed" if returncode == 0 and predicted else "error"
    result: dict[str, Any] = {
        "question_id": question_id,
        "attempt_id": attempt_id,
        "previous_attempt_id": previous_attempt_id,
        "task_fingerprint": task_fingerprint,
        "predicted": predicted,
        "expected": expected,
        "correct": predicted == expected if predicted and expected else None,
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": duration,
        "max_steps": max_steps,
        "temperature": temperature,
        "timeout_seconds": timeout,
        "workspace": str(question_workspace),
        "console_log": str(console_path),
        "trajectory_log": str(trajectory_logs[0]) if trajectory_logs else None,
    }
    if status == "error":
        result["error"] = "timeout" if timed_out else "Agent did not return a valid answer"
    if trajectory_logs:
        termination_reason = (
            "timeout" if timed_out else
            "step_limit" if "Stopped after " in output and " tool steps" in output else
            "completed" if status == "completed" else
            "agent_error" if returncode != 0 else
            "invalid_answer"
        )
        append_event(
            trajectory_logs[0],
            "evaluation",
            turn_index=None,
            step_index=None,
            run_id=run_id,
            task_id=question_id,
            attempt_id=attempt_id,
            previous_attempt_id=previous_attempt_id,
            task_fingerprint=task_fingerprint,
            outcome="success" if result.get("correct") is True else "failure",
            correct=result.get("correct"),
            score=1.0 if result.get("correct") is True else 0.0,
            termination_reason=termination_reason,
            duration_seconds=duration,
            max_steps=max_steps,
            timeout_seconds=timeout,
        )
    return result


def resolve_selection(
    args: argparse.Namespace,
    settings: dict[str, Any],
    available_ids: list[str],
) -> list[str]:
    if args.all_questions:
        selected = list(available_ids)
    elif args.questions:
        selected = parse_question_ids(args.questions)
    elif args.question_range:
        selected = parse_question_range(args.question_range)
    elif args.sample is not None:
        if args.sample < 1 or args.sample > len(available_ids):
            raise ValueError(f"sample must be between 1 and {len(available_ids)}")
        selected = random.Random(args.seed).sample(available_ids, args.sample)
    else:
        configured = settings.get("question_ids")
        if not isinstance(configured, list) or not configured:
            raise ValueError(
                "Select questions with --questions, --range, --sample, or earth_bench.question_ids"
            )
        selected = parse_question_ids(",".join(str(value) for value in configured))
    missing = [question_id for question_id in selected if question_id not in available_ids]
    if missing:
        raise ValueError(f"Unknown question id(s): {', '.join(missing)}")
    configured_exclusions = settings.get("exclude_question_ids", [])
    if args.exclude is not None:
        excluded = parse_question_ids(args.exclude) if args.exclude.strip() else []
    elif configured_exclusions:
        if not isinstance(configured_exclusions, list):
            raise ValueError("earth_bench.exclude_question_ids must be a JSON array")
        excluded = parse_question_ids(",".join(str(value) for value in configured_exclusions))
    else:
        excluded = []
    unknown_exclusions = [item for item in excluded if item not in available_ids]
    if unknown_exclusions:
        raise ValueError(f"Unknown excluded question id(s): {', '.join(unknown_exclusions)}")
    selected = [question_id for question_id in selected if question_id not in set(excluded)]
    if not selected:
        raise ValueError("All selected questions were excluded")
    return selected


def main() -> None:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    preliminary.add_argument("--profile")
    known, _ = preliminary.parse_known_args()
    try:
        config = load_config(known.config.expanduser(), known.profile)
    except ValueError as error:
        preliminary.error(str(error))
    settings = config.get("earth_bench") or {}

    parser = argparse.ArgumentParser(description="Run selected Earth-Bench questions")
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--profile", default=config.get("active_profile"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", dest="all_questions", action="store_true", help="Select every question")
    selection.add_argument("--questions", help="Comma-separated ids, for example 2,5,17")
    selection.add_argument("--range", dest="question_range", help="Inclusive range, for example 1:10")
    selection.add_argument("--sample", type=int, help="Randomly sample N questions")
    parser.add_argument(
        "--exclude",
        help="Comma-separated ids to exclude; overrides earth_bench.exclude_question_ids",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the final selection without running agents")
    parser.add_argument("--seed", type=int, default=int(settings.get("seed", 42)))
    parser.add_argument(
        "--mode",
        choices=("autonomous_planning", "instruction_following"),
        default=settings.get("mode", "autonomous_planning"),
    )
    parser.add_argument("--max-steps", type=int, default=int(settings.get("max_steps_per_question", 20)))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(settings.get("parallel_workers", 1)),
        help="Number of questions to run concurrently (default: earth_bench.parallel_workers or 1)",
    )
    parser.add_argument("--temperature", type=float, default=config.get("temperature"))
    parser.add_argument("--timeout", type=int, default=int(settings.get("timeout_seconds", 300)))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(settings.get("output_dir", VAR_ROOT / "benchmarks/earthbench-runs")).expanduser(),
        help="Directory under which a timestamped run directory is created",
    )
    parser.add_argument("--resume", type=Path, help="Resume an existing run directory or results.json")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=bool(settings.get("continue_on_error", True)),
    )
    parser.add_argument(
        "--show-tools",
        action=argparse.BooleanOptionalAction,
        default=bool(settings.get("show_tools", True)),
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.timeout < 1 or args.workers < 1:
        parser.error("--max-steps, --timeout, and --workers must be positive")
    if args.workers > 1 and not args.continue_on_error:
        parser.error("--no-continue-on-error requires --workers 1")
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")

    question_path = Path(settings.get("question_file", "")).expanduser()
    answer_path = Path(settings.get("answer_file", "")).expanduser()
    output_base = args.output_dir.expanduser()
    try:
        questions = load_json(question_path)
        answers = expected_answers(load_json(answer_path))
        if not isinstance(questions, dict):
            raise ValueError("Earth-Bench question file must contain a JSON object")
        available_ids = sorted((str(key) for key in questions), key=int)
        selected = resolve_selection(args, settings, available_ids)
    except ValueError as error:
        parser.error(str(error))

    if args.dry_run:
        print(f"Selected {len(selected)} question(s): {', '.join(selected)}")
        return

    if args.resume:
        resume_path = args.resume.expanduser().resolve()
        results_path = resume_path if resume_path.is_file() else resume_path / "results.json"
        manifest = load_json(results_path)
        run_dir = results_path.parent
        previous_selection = manifest.get("selected_question_ids")
        if previous_selection != selected:
            parser.error("Resume selection does not match this run's selected question ids")
    else:
        run_id = now_beijing().strftime("%Y%m%d-%H%M%S")
        run_dir = (output_base / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        results_path = run_dir / "results.json"
        manifest = {
            "run_id": run_id,
            "created_at": now_beijing().isoformat(timespec="seconds"),
            "timezone": "Asia/Shanghai",
            "profile": args.profile,
            "mode": args.mode,
            "temperature": args.temperature,
            "parallel_workers": args.workers,
            "selected_question_ids": selected,
            "excluded_question_ids": [
                question_id for question_id in available_ids if question_id not in selected
            ] if args.all_questions else [],
            "results": [],
        }
        update_summary(manifest, len(selected))
        save_manifest(results_path, manifest)

    finished = {
        str(item.get("question_id"))
        for item in manifest.get("results", [])
        if item.get("status") == "completed"
    }
    print(f"Run directory: {run_dir}")
    print(f"Questions: {', '.join(selected)}")
    print(f"Profile: {args.profile or '(default)'} · mode: {args.mode}")
    print(f"Temperature: {args.temperature if args.temperature is not None else '(provider default)'}")
    print(f"Parallel workers: {args.workers}")
    agent_path = Path(__file__).resolve().parents[1] / "runtime/entry.py"
    earth_settings = config.get("earth_tools") or {}
    tools_dir_setting = earth_settings.get("tools_dir")
    if not tools_dir_setting:
        parser.error("earth_tools.tools_dir is required for Earth-Bench runs")
    earth_root = Path(tools_dir_setting).expanduser().resolve().parent.parent

    jobs: list[dict[str, Any]] = []
    for index, question_id in enumerate(selected, 1):
        if question_id in finished:
            print(f"[{index}/{len(selected)}] question {question_id}: already completed, skipping")
            continue
        try:
            _, configured_data = select_evaluation(questions[question_id], args.mode)
            question_data = (earth_root / configured_data).resolve(strict=True)
            if not question_data.is_dir():
                raise ValueError(f"Question data path is not a directory: {question_data}")
            prompt = build_prompt(
                question_id,
                questions[question_id],
                args.mode,
                data_directory=str(question_data),
            )
        except (OSError, ValueError) as error:
            result = {
                "question_id": question_id,
                "status": "error",
                "error": f"question data unavailable: {error}",
            }
            manifest["results"].append(result)
            update_summary(manifest, len(selected))
            save_manifest(results_path, manifest)
            if not args.continue_on_error:
                break
            continue
        previous_result = next(
            (
                item for item in reversed(manifest.get("results", []))
                if str(item.get("question_id")) == question_id
            ),
            None,
        )
        attempt_number = 1 + sum(
            str(item.get("question_id")) == question_id
            for item in manifest.get("results", [])
        )
        jobs.append({
            "index": index,
            "question_id": question_id,
            "prompt": prompt,
            "question_data": question_data,
            "expected": answers.get(question_id),
            "previous_result": previous_result,
            "previous_attempt_count": attempt_number - 1,
        })

    def accept_result(index: int, result: dict[str, Any]) -> bool:
        manifest["results"].append(result)
        update_summary(manifest, len(selected))
        save_manifest(results_path, manifest)
        marker = "✓" if result["correct"] else "✗"
        print(
            f"[{index}/{len(selected)}] question {result['question_id']}: {marker} "
            f"predicted={result.get('predicted') or 'FAIL'} "
            f"expected={result.get('expected') or '?'} "
            f"({result.get('duration_seconds', 0):.1f}s)",
            flush=True,
        )
        return result.get("status") == "error"

    common = {
        "run_id": str(manifest["run_id"]),
        "run_dir": run_dir,
        "agent_path": agent_path,
        "config_path": args.config.expanduser().resolve(),
        "profile": args.profile,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "show_tools": args.show_tools,
    }
    cancel_event = Event()
    if args.workers == 1:
        for job in jobs:
            index = int(job.pop("index"))
            question_id = str(job["question_id"])
            print(f"[{index}/{len(selected)}] question {question_id}: running", flush=True)
            result = evaluate_question(
                **job,
                **common,
                stream_output=True,
                cancel_event=cancel_event,
            )
            if accept_result(index, result) and not args.continue_on_error:
                break
    elif jobs:
        worker_count = min(args.workers, len(jobs))
        executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="earthbench")
        future_jobs: dict[Future[dict[str, Any]], tuple[int, str]] = {}
        try:
            for job in jobs:
                index = int(job.pop("index"))
                question_id = str(job["question_id"])
                print(f"[{index}/{len(selected)}] question {question_id}: queued", flush=True)
                future = executor.submit(
                    evaluate_question,
                    **job,
                    **common,
                    stream_output=False,
                    cancel_event=cancel_event,
                )
                future_jobs[future] = (index, question_id)
            for future in as_completed(future_jobs):
                index, question_id = future_jobs[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "question_id": question_id,
                        "status": "error",
                        "correct": None,
                        "predicted": None,
                        "expected": answers.get(question_id),
                        "duration_seconds": 0.0,
                        "error": f"parallel worker failed: {type(error).__name__}: {error}",
                    }
                accept_result(index, result)
        except KeyboardInterrupt:
            cancel_event.set()
            for future in future_jobs:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    completed = [item for item in manifest["results"] if item.get("status") == "completed"]
    correct = sum(item.get("correct") is True for item in completed)
    update_summary(manifest, len(selected))
    save_manifest(results_path, manifest)
    print(f"Results: {results_path}")
    print(f"Completed: {len(completed)}/{len(selected)} · correct: {correct}/{len(completed)}")


if __name__ == "__main__":
    main()
