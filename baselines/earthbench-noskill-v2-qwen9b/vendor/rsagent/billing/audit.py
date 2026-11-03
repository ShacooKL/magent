"""Read-only tariff audit and optional normalized provider-invoice comparison."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from rsagent.billing.qwen import MODELS, quote, validate_policy
from rsagent.evolution.request_ledger import RequestLedger, request_quote


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def ledger_report(root):
    root = Path(root)
    state = json.loads((root / "ledger.json").read_text())
    validate_policy(state["budget"]["billing_policy"])
    rows, seen = [], set()
    for key, request in state["requests"].items():
        provider_id = request.get("response_metadata", {}).get("id")
        value = request_quote(request, state["budget"])
        rows.append({"request_id": key, "provider_request_id": provider_id,
            "session_id": request["session_id"], "started_at": request.get("started_at"),
            "finished_at": request.get("finished_at"), "error": request["error"],
            "raw_usage": request.get("raw_usage"), "quote": value,
            "stored_quote_matches_recomputation": request.get("billing_quote", value) == value,
            "duplicate_provider_id": bool(provider_id and provider_id in seen)})
        if provider_id:
            seen.add(provider_id)
    return {"schema": "qwen_billing_audit_v1", "source_ledger_sha256": file_sha(root / "ledger.json"),
        "summary": RequestLedger.summarize(state), "requests": rows,
        "provider_bill_verified": False, "basis": "API usage multiplied by frozen public tariff"}


def audit_events(paths, policy):
    """Historical read-only repricing; no claim that today's tariff applied then."""
    validate_policy(policy)
    rows, seen, sources = [], set(), {}
    for path in map(Path, paths):
        sources[str(path.resolve())] = file_sha(path)
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                event = json.loads(line)
                if event.get("type") != "api_response":
                    continue
                response = event.get("response", {})
                provider_id = response.get("id")
                if response.get("model") not in MODELS:
                    raise ValueError("Non-Qwen35 response in historical audit")
                duplicate = bool(provider_id and provider_id in seen)
                if provider_id:
                    seen.add(provider_id)
                raw = response.get("raw_usage", response.get("usage"))
                value = quote(raw, policy)
                rows.append({"source": str(path.resolve()), "line": line_number, "time": event.get("time"),
                    "provider_request_id": provider_id, "raw_usage": raw, "quote": value,
                    "duplicate_provider_id": duplicate})
    eligible = [r for r in rows if not r["duplicate_provider_id"]]
    total = sum((Decimal(r["quote"]["tariff_cost"]) for r in eligible if r["quote"]["usage_complete"]), Decimal(0))
    return {"schema": "qwen_billing_audit_v1", "sources": sources, "policy": policy, "requests": rows,
        "tariff_cost_complete_rows": str(total), "complete_rows": sum(r["quote"]["usage_complete"] for r in eligible),
        "unknown_rows": sum(not r["quote"]["usage_complete"] for r in eligible),
        "duplicates_excluded": len(rows) - len(eligible), "provider_bill_verified": False,
        "basis": "Historical API usage repriced at selected tariff; not the original invoice"}


def reconcile(report, invoice):
    """Join operator-supplied invoice rows by provider request id, never timestamps.

    gross_cny means undiscounted model charge; net_cny is optional cash charge.
    Aggregate bills without request ids need a separate aggregate reconciliation,
    not guessed joins. This function does not mutate operational budget balances.
    """
    if invoice.get("schema") != "normalized_provider_invoice_v1" or invoice.get("currency") != "CNY":
        raise ValueError("Expected normalized CNY provider invoice")
    bills = {}
    for row in invoice["rows"]:
        key = row.get("provider_request_id")
        if not isinstance(key, str) or not key or key in bills:
            raise ValueError("Missing or duplicate invoice request id")
        for field in ("gross_cny", "net_cny"):
            if field in row:
                value = Decimal(str(row[field]))
                if not value.is_finite() or value < 0:
                    raise ValueError("Invalid invoice amount")
        bills[key] = row
    results, visited = [], set()
    for row in report["requests"]:
        key = row.get("provider_request_id")
        if row.get("duplicate_provider_id"):
            raise ValueError("Duplicate local provider id; resolve before reconciliation")
        bill = bills.get(key)
        if bill:
            visited.add(key)
        tariff = row["quote"]["tariff_cost"]
        difference = str(Decimal(str(bill["gross_cny"])) - Decimal(tariff)) if (
            bill and "gross_cny" in bill and tariff is not None) else None
        results.append({"provider_request_id": key, "tariff_cny": tariff,
            "provider_gross_cny": bill.get("gross_cny") if bill else None,
            "provider_net_cny": bill.get("net_cny") if bill else None,
            "gross_minus_tariff_cny": difference,
            "status": "matched" if difference is not None else "unmatched_or_unknown"})
    return {"schema": "qwen_invoice_comparison_v1", "rows": results,
        "unmatched_invoice_ids": sorted(set(bills) - visited),
        "all_gross_equal": bool(results) and len(visited) == len(bills)
            and all(r["gross_minus_tariff_cny"] is not None and Decimal(r["gross_minus_tariff_cny"]) == 0 for r in results),
        "basis": "Operator-supplied invoice; net discounts are reported separately, not treated as tariff errors"}
