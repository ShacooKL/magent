"""EarthBench baseline v2: 248 tasks, image files, no skill, original tool schemas."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))

from magents.adapters import api_settings, client_for, WorkspaceTools
from magents.engine import Store, Team
from rsagent.billing.qwen import policy_for, quote


def submitted_choice(state):
    final = state.get("final_submission")
    if final and final.get("status") != "completed":
        return None
    answer = (final.get("answer", "") if final else state.get("answer", "")).strip()
    if final and re.fullmatch(r"[A-D]", answer):
        return answer
    if final:
        label = re.fullmatch(r"([A-D])[.、:：)]\s*[^\n]+", answer)
        if label:
            return label[1]
    found = re.findall(r"<Answer>\s*([A-D])\s*</Answer>", answer)
    return found[-1] if found else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, help="Optional raster-only input directory shared by paired controls")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument('--ledger-root', type=Path, help='Host-owned shared batch budget')
    parser.add_argument("--score-only", action="store_true", help="Offline append-only scoring correction; no API")
    args = parser.parse_args()
    if args.task_id < 1:
        parser.error("task-id must be positive")
    task_id = str(args.task_id)
    baseline = ROOT / 'earthbench'
    if args.output.resolve().is_relative_to(ROOT):
        parser.error('Run outputs must be outside the frozen baseline directory')
    if args.score_only:
        state = json.loads((args.output / "state.json").read_text())
        manifest = json.loads((args.output / "run.json").read_text())
        expected = json.loads((baseline / "private/answers.json").read_text())[manifest["task_id"]]
        answer = submitted_choice(state)
        events = args.output / "events.jsonl"
        last_id = max(json.loads(line)["id"] for line in events.read_text().splitlines())
        result = {"id": last_id + 1, "kind": "host_score_correction", "answer": answer,
                  "expected": expected, "correct": answer in expected if answer else False,
                  "workflow_status": state["status"],
                  "note": "Offline structured submit_final scoring. Original score preserved; no feedback sent to agents."}
        with events.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        print(json.dumps(result))
        return
    data = (args.data_dir or Path("/srv/user/yangxiao/linzw/Agent/datasets/Earth-Bench") / ("question" + task_id)).resolve()
    if not data.is_dir():
        raise ValueError(f"Missing task data: {data}")
    # Read only the published question portion, never historical agent instructions.
    source = json.loads((baseline / "tasks" / (task_id + ".json")).read_text())
    question = source["question"].split("\n\nQuestion:", 1)[1].strip()
    if any(p.is_symlink() or (p.is_file() and p.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"})
           for p in data.rglob("*")):
        raise ValueError("Task mount must contain image data only")
    config = json.loads((ROOT / "configs/default.json").read_text())
    config["official_tools"] = True
    config["official_only"] = True
    config['hypothesis_verification'] = False
    config["skill_retrieval"] = False
    api_path = ROOT / 'configs/api.json'
    api = api_settings(api_path, None)
    api["earth_tools"] = {**api["earth_tools"],
                          "tools_dir": str(baseline / "fixed-earth/agent/tools"),
                          "model_results_file": str(baseline / "fixed-earth/benchmark/model_results.csv")}
    policy = policy_for(api)
    task = (question + "\nData directory: " + str(data) +
            "\nThe command sandbox's writable directory is /workspace. For official tools use absolute output paths under /workspace; pass exact returned paths downstream. "
            "Use official tools for supported calculations. Multiple independent calls can be made in one response. "
            "Inspect actual data coverage and report missing inputs rather than inventing them. "
            "Final answer must end with <Answer>X</Answer>.")
    store = Store(args.output, task, {"mode": "preflight" if args.preflight else "live",
                  "baseline_version": "earthbench-noskill-v2",
                  "task_id": task_id, "data_dir": str(data), "skill": None, "model": api["model"], "config": config,
                  "billing_policy": policy, "api_config_path": str(api_path),
                  "earth_python": api['earth_tools']['python'],
                  "tools_dir": api["earth_tools"]["tools_dir"]})
    tools = None
    try:
        tools = WorkspaceTools(store.root / "workspace", [data], api, official=True,
                               official_only=True,
                               skill_dir=None,
                               hypothesis_verification=config['hypothesis_verification'])
        if tools.skills:
            store.event('skill_library_snapshot', cards=list(tools.skills.cards.values()))
        store.event("tool_catalog", definitions=tools.definitions)
        # Host-only environment check; not an agent capability in official-only mode.
        check = tools.runner.run(tools.workspace,
            "python -c 'import rasterio; print(rasterio.__version__)'",
            {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, timeout=120)
        probe = {"stdout": check.stdout, "exit_code": check.exit_code, "timed_out": check.timed_out}
        store.event("sandbox_preflight", result=probe)
        if probe["exit_code"] != 0 or probe["timed_out"]:
            raise RuntimeError("Sandbox/geospatial preflight failed")
        print(f"Preflight: {len(tools.earth.definitions)} official tools; rasterio {probe['stdout'].strip()}", flush=True)
        if args.preflight:
            store.state["status"] = "preflight_passed"
            return
        client = client_for(api_path, api)
        if args.ledger_root:
            from rsagent.billing.client import BillingClient
            from rsagent.billing.qwen import budget_for
            from rsagent.evolution.request_ledger import RequestLedger
            batch = json.loads((args.ledger_root.parent / 'batch.json').read_text())
            ledger = RequestLedger(args.ledger_root, max_requests=248 * 40,
                max_api_failures=50, budget=budget_for(api, batch['budget_cny']))
            ledger.register_session(task_id, max_requests=40)
            client = BillingClient(client, ledger, task_id)
        Team(client, tools, store, config).run()
        print(store.state["answer"], flush=True)
        # Private answers are read by the host only AFTER the final response.
        answers = json.loads((baseline / "private/answers.json").read_text())
        expected = answers[task_id]
        answer = submitted_choice(store.state)
        store.event("host_score", answer=answer, expected=expected,
                    correct=answer in expected if answer else False,
                    note="Host-only evaluation; no feedback supplied to team")
    except Exception as exc:
        if store.state["status"] not in {"failed", "budget_exhausted"}:
            store.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if tools:
            tools.close()
        from decimal import Decimal
        costs, missing = [], 0
        events_path = store.root / "events.jsonl"
        for line in events_path.read_text().splitlines() if events_path.exists() else []:
            row = json.loads(line)
            if row["kind"] == "response":
                price = quote(row["response"].get("raw_usage", row["response"].get("usage")), policy)
                if price.get("estimated_cost") is None:
                    missing += 1
                else:
                    costs.append(Decimal(price["estimated_cost"]))
        store.state["billing"] = {"estimated_cny": str(sum(costs, Decimal(0))),
                                  "responses_without_price": missing, "provider_bill_verified": False}
        store.save()


if __name__ == "__main__":
    main()
