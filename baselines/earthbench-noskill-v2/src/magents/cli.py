import argparse
import json
from pathlib import Path

from magents.engine import Store, Team

ROOT = Path(__file__).resolve().parents[2]


class DemoClient:
    def __init__(self):
        self.index = 0

    def create_message(self, system, messages, tools, max_tokens):
        replies = ["列出三个数，计算均值，再核查。", "2、4、6 的均值是 4。离线演示，未调用真实工具。",
                   "INSUFFICIENT：这是调度演示，没有工具执行证据。", "ACCEPT：仅演示，不作为真实计算验证。",
                   "均值为 4。本次仅验证离线角色交接。"]
        reply = replies[self.index]
        self.index += 1
        if self.index in (1, 2, 3, 5):
            name, value = {
                1: ("submit_plan", {"plan": reply}),
                2: ("submit_progress", {"status": "complete", "completed": [reply], "pending": [],
                    "issues": ["离线演示，没有真实计算证据"], "artifacts": [], "evidence_ids": [1], "next_step": "核查"}),
                3: ("submit_review", {"verdict": "INSUFFICIENT", "checked": [], "issues": [reply],
                    "unchecked": ["真实工具计算"], "evidence_ids": [1]}),
                5: ("submit_final", {"status": "incomplete", "answer": "", "reason": reply,
                    "evidence_ids": [1]}),
            }[self.index]
            return {"content": [{"type": "tool_use", "id": f"demo-{self.index}",
                                 "name": name, "input": value}], "usage": {}, "stop_reason": "tool_calls"}
        return {"content": [{"type": "text", "text": reply}], "usage": {}, "stop_reason": "stop"}


class EmptyTools:
    definitions = []


def main():
    parser = argparse.ArgumentParser(description="Minimal multi-agent runner")
    parser.add_argument("command", choices=["demo", "run", "status", "preflight"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--task", type=Path, help="Public UTF-8 task text, no GT")
    parser.add_argument("--data", type=Path, action="append", default=[], help="Public data-only directory; never mount benchmark/repository root")
    parser.add_argument("--output", type=Path, default=ROOT / "var/runs/demo")
    parser.add_argument("--profile", help="Override existing API profile")
    parser.add_argument("--no-skill-retrieval", action="store_true")
    args = parser.parse_args()
    if args.command == "status":
        print((args.output / "state.json").read_text(encoding="utf-8"))
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config['skill_retrieval'] = config.get('skill_retrieval', False) and not args.no_skill_retrieval
    for key in ("max_requests", "max_tokens", "max_role_requests", "tool_output_chars"):
        if type(config[key]) is not int or config[key] < 1:
            parser.error(f"{key} must be a positive integer")
    if type(config["review_rounds"]) is not int or not 0 <= config["review_rounds"] <= 3:
        parser.error("review_rounds must be between 0 and 3")
    reserve = config.get("downstream_reserve", 0)
    if type(reserve) is not int or not 0 <= reserve < config["max_requests"]:
        parser.error("downstream_reserve must be nonnegative and below max_requests")
    if args.command == "demo":
        store = Store(args.output, "求 2、4、6 的均值", {"mode": "offline_demo", "config": config})
        Team(DemoClient(), EmptyTools(), store, config).run()
        print(json.dumps(store.state, ensure_ascii=False, indent=2))
        return
    from magents.adapters import api_settings, client_for, WorkspaceTools
    path = (args.config.resolve().parent / config["api_config"]).resolve()
    profile = args.profile or config["profile"]
    api = api_settings(path, profile)
    if args.command == "preflight":
        print(json.dumps({"profile": profile, "model": api["model"], "base_url": api["base_url"],
                          "max_requests": config["max_requests"], "max_tokens": config["max_tokens"],
                          "official_tools_enabled": config["official_tools"],
                          "note": "Configuration only; no API call, credential read or sandbox execution."}, indent=2))
        return
    if not args.task:
        parser.error("run requires --task with public task text")
    data = [p.resolve(strict=True) for p in args.data]
    # Conservative v1 input contract: mount data-only directories, not EarthBench
    # directories containing question.json (which can include answers/traces).
    allowed = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".csv", ".npy"}
    for folder in data:
        if not folder.is_dir() or folder.is_symlink():
            parser.error("--data must be a public data directory")
        for child in folder.rglob("*"):
            if child.is_symlink() or (child.is_file() and child.suffix.lower() not in allowed):
                parser.error(f"Data-only contract rejects {child.name}; prepare public input separately")
    task = args.task.read_text(encoding="utf-8") + "\nRead-only input directories: " + json.dumps([str(p) for p in data])
    client = client_for(path, api)
    manifest = {"mode": "live", "profile": profile, "model": api["model"],
                "base_url": api["base_url"], "config": config, "data": [str(p) for p in data]}
    store = Store(args.output, task, manifest)
    toolbox = None
    try:
        toolbox = WorkspaceTools(store.root / "workspace", data, api, config["official_tools"],
                                 official_only=config.get('official_only', False),
                                 skill_dir=ROOT / 'library/skills' if config['skill_retrieval'] else None)
        if toolbox.skills:
            store.event('skill_library_snapshot', cards=list(toolbox.skills.cards.values()))
        store.event("tool_catalog", definitions=toolbox.definitions)
        Team(client, toolbox, store, config).run()
        print(store.state["answer"] or store.state.get("final_submission", {}).get("reason", "No final submission"))
    except Exception as exc:
        if store.state["status"] not in ("failed", "budget_exhausted"):
            store.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            store.save()
        raise
    finally:
        if toolbox:
            toolbox.close()


if __name__ == "__main__":
    main()
