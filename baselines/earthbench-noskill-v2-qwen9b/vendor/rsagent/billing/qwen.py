"""Versioned Beijing Qwen3.5 Flash Chat Completions tariff (CNY, text only).

Use Decimal without per-request rounding. Raw API usage is authoritative for
token counts, not a local tokenizer/byte estimate. Invoice discounts are separate.
"""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json

PROFILE = "dashscope-qwen35-flash"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODELS = ("qwen3.5-flash", "qwen3.5-flash-2026-02-23")
PRICEBOOK = {
    "id": "dashscope-cn-beijing-qwen35-flash-text-20261026-v1",
    "provider": "dashscope", "region": "cn-beijing", "currency": "CNY",
    "endpoint": ENDPOINT, "model": "qwen3.5-flash", "mode": "online_text_chat",
    "verified_on": "2026-10-26", "unit_tokens": 1000000,
    "sources": ["https://help.aliyun.com/zh/model-studio/qwen3-5-flash",
                "https://help.aliyun.com/zh/model-studio/context-cache"],
    "tiers": [
        {"max_prompt_tokens": 128000, "input": "0.2", "creation": "0.25", "hit": "0.02", "output": "2"},
        {"max_prompt_tokens": 256000, "input": "0.8", "creation": "1", "hit": "0.08", "output": "8"},
        {"max_prompt_tokens": 1000000, "input": "1.2", "creation": "1.5", "hit": "0.12", "output": "12"},
    ],
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def policy_for(config):
    if (config.get("provider") != "openai" or config.get("model") != "qwen3.5-flash"
            or config.get("base_url", "").rstrip("/") != ENDPOINT
            or config.get("cache_mode") not in (None, "explicit")):
        raise ValueError("Billing v1 requires Beijing qwen3.5-flash online text Chat Completions")
    return {"schema": "qwen35_tariff_v1", "cache_mode": config.get("cache_mode"),
            "pricebook": deepcopy(PRICEBOOK), "pricebook_sha256": fingerprint(PRICEBOOK)}


def validate_policy(policy):
    # Version-specific verification: add a NEW version when prices change; never
    # silently reinterpret an old archived snapshot under new prices.
    if (not isinstance(policy, dict) or policy.get("schema") != "qwen35_tariff_v1"
            or policy.get("pricebook") != PRICEBOOK
            or policy.get("pricebook_sha256") != fingerprint(PRICEBOOK)
            or policy.get("cache_mode") not in (None, "explicit")):
        raise ValueError("Unknown or altered Qwen tariff snapshot")
    return deepcopy(policy)


def budget_for(config, cap):
    amount = Decimal(str(cap))
    if not amount.is_finite() or amount <= 0:
        raise ValueError("A new positive CNY cap is required")
    return {"currency": "CNY", "max_estimated_cost": str(amount), "carried_cost": "0",
            "billing_policy": policy_for(config)}


def _count(value):
    return type(value) is int and value >= 0


def _field(raw, name, aliases, details, detail_key):
    values = [raw[k] for k in (name, *aliases) if k in raw]
    if detail_key in details:
        values.append(details[detail_key])
    if not values:
        return None
    if not all(_count(v) for v in values) or any(v != values[0] for v in values):
        raise ValueError("Invalid or conflicting " + name)
    return values[0]


def quote(raw, policy, *, reservation=False):
    """Return exact public-tariff arithmetic or a clearly labeled uncertain quote.

    creation/hit are disjoint SUBSETS of prompt_tokens. Reasoning tokens are
    already inside completion_tokens, so they are never charged a second time.
    """
    validate_policy(policy)
    result = {"currency": "CNY", "pricebook_sha256": policy["pricebook_sha256"],
        "estimated_cost": None, "tariff_cost": None, "cache_breakdown_complete": False,
        "usage_complete": False, "tier_max_prompt_tokens": None, "components": {},
        "basis": "missing_usage", "provider_bill_verified": False}
    if not isinstance(raw, dict):
        return result
    prompt, output = raw.get("prompt_tokens"), raw.get("completion_tokens")
    if not _count(prompt) or not _count(output):
        return {**result, "basis": "missing_or_invalid_tokens"}
    tier = next((t for t in policy["pricebook"]["tiers"] if prompt <= t["max_prompt_tokens"]), None)
    if tier is None:
        return {**result, "basis": "outside_tariff_scope"}
    result["tier_max_prompt_tokens"] = tier["max_prompt_tokens"]
    worst_input = max(Decimal(tier["input"]), Decimal(tier["creation"]))
    result["estimated_cost"] = str((prompt * worst_input + output * Decimal(tier["output"])) / 1000000)
    if reservation:
        return {**result, "basis": "preflight_reservation_not_spend"}
    try:
        total = raw.get("total_tokens")
        if total is not None and (not _count(total) or total != prompt + output):
            raise ValueError("total_tokens mismatch")
        details = raw.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            raise ValueError("Invalid prompt details")
        created = _field(raw, "prompt_cache_creation_tokens", ("cache_creation_tokens",),
                         details, "cache_creation_input_tokens")
        hit = _field(raw, "prompt_cache_hit_tokens", ("cache_hit_tokens",), details, "cached_tokens")
        if policy["cache_mode"] is None:
            # This pricebook has no implicit-cache SKU. Unexpected cache use
            # must not be charged at an invented implicit discount.
            if created not in (None, 0) or hit not in (None, 0):
                raise ValueError("Unexpected caching without explicit mode")
            created, hit = 0, 0
        if created is None or hit is None:
            return {**result, "basis": "missing_cache_breakdown"}
        if created + hit > prompt:
            raise ValueError("Cache subsets exceed prompt_tokens")
        reasoning = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if reasoning is not None and (not _count(reasoning) or reasoning > output):
            raise ValueError("Invalid reasoning subset")
    except (ValueError, TypeError, AttributeError) as error:
        return {**result, "basis": "inconsistent_usage", "reason": str(error)}
    counts = {"input": prompt - created - hit, "creation": created, "hit": hit, "output": output}
    parts = {k: {"tokens": n, "per_million_cny": tier[k],
                 "cost_cny": str(n * Decimal(tier[k]) / 1000000)} for k, n in counts.items()}
    total = sum((Decimal(v["cost_cny"]) for v in parts.values()), Decimal(0))
    return {**result, "estimated_cost": str(total), "tariff_cost": str(total),
            "usage_complete": True, "cache_breakdown_complete": True, "components": parts,
            "basis": "provider_usage_public_tariff"}
