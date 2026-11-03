"""Offline reports only. No credentials or network access required."""
import argparse
import json
from pathlib import Path

from rsagent.billing.audit import audit_events, file_sha, ledger_report, reconcile
from rsagent.billing.qwen import ENDPOINT, policy_for
from rsagent.evolution.tool_registry import ToolRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("report", "audit-events", "reconcile"))
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--events", nargs="+", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--invoice", type=Path)
    parser.add_argument("--cache-mode", choices=("explicit", "none"), default="explicit")
    parser.add_argument("--output", type=Path, help="Fresh output directory; never overwrite archived ledgers")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output directory must be new")
    if args.action == "report":
        if not args.ledger:
            parser.error("--ledger required")
        result = ledger_report(args.ledger)
    elif args.action == "audit-events":
        if not args.events:
            parser.error("--events required")
        policy = policy_for({"provider": "openai", "model": "qwen3.5-flash", "base_url": ENDPOINT,
                             "cache_mode": None if args.cache_mode == "none" else "explicit"})
        result = audit_events(args.events, policy)
    else:
        if not args.report or not args.invoice:
            parser.error("--report and --invoice required")
        result = reconcile(json.loads(args.report.read_text()), json.loads(args.invoice.read_text()))
        result.update(report_sha256=file_sha(args.report), invoice_sha256=file_sha(args.invoice))
    if args.output:
        ToolRegistry(args.output).write("report.json", result)
        print(json.dumps({"report": str(args.output / "report.json"), "requests": len(result.get("requests", [])),
            "summary": result.get("summary"), "tariff_cost_complete_rows": result.get("tariff_cost_complete_rows"),
            "complete_rows": result.get("complete_rows"), "unknown_rows": result.get("unknown_rows")}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
