"""One public EarthBench raster task; legacy filename retained for compatibility."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "python-agent/src"))

from magents.adapters import api_settings, client_for, WorkspaceTools
from magents.engine import Store, Team
from rsagent.billing.qwen import policy_for, quote


def array_only_ndvi_inputs(definitions):
    """Experimental model-facing schema; leave official numerical code untouched."""
    result = deepcopy(definitions)
    tool = next(t for t in result if t['name'] == 'calculate_mean_lst_by_ndvi')
    for name in ('red_paths', 'nir_paths', 'lst_paths'):
        prop = tool['input_schema']['properties'][name]
        prop.pop('anyOf', None)
        prop.update(type='array', items={'type': 'string'})
    return result


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
    parser.add_argument('--profile', help='Configured model profile override')
    parser.add_argument('--earth-python', type=Path, help='Experiment-only official-tool Python override')
    parser.add_argument('--max-requests', type=int, help='Shared solver request cap')
    parser.add_argument("--data-dir", type=Path, help="Optional raster-only input directory shared by paired controls")
    parser.add_argument("--skill-file", type=Path, help="Frozen advisory skill; no source trajectories")
    parser.add_argument("--no-skill-retrieval", action="store_true", help="Disable library tools for clean controls")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument('--ndvi-array-inputs', action='store_true',
                        help='Experimental array-only schema for NDVI-conditioned mean LST paths')
    parser.add_argument('--hypothesis-verification', action='store_true', help='Allow evidence-linked diagnostic commands; final computations still follow official-only policy when selected')
    parser.add_argument("--official-only", action="store_true", help="Disable auxiliary code and require official-tool computations for all roles")
    parser.add_argument("--score-only", action="store_true", help="Offline append-only scoring correction; no API")
    args = parser.parse_args()
    if args.task_id < 1:
        parser.error("task-id must be positive")
    if args.max_requests is not None and args.max_requests < 1:
        parser.error('max-requests must be positive')
    task_id = str(args.task_id)
    baseline = ROOT.parent / "evolution-lab/official-tool-baseline-v1/runs/qwen35-01"
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
    if any(p.is_symlink() or (p.is_file() and p.suffix.lower() not in {".tif", ".tiff"})
           for p in data.rglob("*")):
        raise ValueError("Task mount must contain raster data only")
    config = json.loads((ROOT / "configs/default.json").read_text())
    if args.profile:
        config['profile'] = args.profile
    if args.max_requests is not None:
        config['max_requests'] = args.max_requests
    config["official_tools"] = True
    config["official_only"] = args.official_only
    config['ndvi_array_inputs'] = args.ndvi_array_inputs
    config['hypothesis_verification'] = args.hypothesis_verification
    config["skill_retrieval"] = config.get("skill_retrieval", False) and not args.no_skill_retrieval and not args.skill_file
    api_path = ROOT.parent / "config/py-agent/config.json"
    api = api_settings(api_path, config["profile"])
    api["earth_tools"] = {**api["earth_tools"],
                          "tools_dir": str(baseline / "fixed-earth/agent/tools"),
                          "model_results_file": str(baseline / "fixed-earth/benchmark/model_results.csv")}
    if args.earth_python:
        api['earth_tools']['python'] = str(args.earth_python.resolve(strict=True))
    policy = policy_for(api)
    task = (question + "\nData directory: " + str(data) +
            "\nThe command sandbox's writable directory is /workspace. For official tools use absolute output paths under /workspace; pass exact returned paths downstream. "
            "Use official tools for supported calculations. Multiple independent calls can be made in one response. "
            "Inspect actual data coverage and report missing inputs rather than inventing them. "
            "Final answer must end with <Answer>X</Answer>.")
    skill = args.skill_file.read_text() if args.skill_file else None
    if skill:
        task += "\nAdvisory skill (fallible; obey current tool restrictions):\n" + skill
    store = Store(args.output, task, {"mode": "preflight" if args.preflight else "live",
                  "task_id": task_id, "data_dir": str(data), "skill": skill, "model": api["model"], "config": config,
                  "billing_policy": policy, "api_config_path": str(api_path),
                  "earth_python": api['earth_tools']['python'],
                  "tools_dir": api["earth_tools"]["tools_dir"]})
    tools = None
    try:
        tools = WorkspaceTools(store.root / "workspace", [data], api, official=True,
                               official_only=args.official_only,
                               skill_dir=ROOT / 'library/skills' if config['skill_retrieval'] else None,
                               hypothesis_verification=config['hypothesis_verification'])
        if args.ndvi_array_inputs:
            # tool_inspect and model calls both consume this same catalog.
            tools.definitions = array_only_ndvi_inputs(tools.definitions)
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
        Team(client_for(api_path, api), tools, store, config).run()
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
