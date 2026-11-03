"""Persist request/usage facts and optional caller-estimated spending reservations."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import os
import tempfile

from rsagent.evolution.tool_registry import ToolRegistry, canonical, safe_bytes, strict_json
from rsagent.runtime.clock import now_beijing
from rsagent.billing.qwen import validate_policy, quote as qwen_quote


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens",
                "cache_hit_tokens", "cache_miss_tokens")


class _LedgerStorage(ToolRegistry):
    """Host-only atomic ledger snapshots, flushed before requests/settlements.

    Unlike small Tool definitions, an accounting ledger may contain thousands
    of raw usage rows. It is still explicitly bounded (64 MiB per batch).
    """
    def read(self, relative, default=None):
        path = self.root / relative
        return strict_json(safe_bytes(path, 64 * 1024 * 1024).decode()) if path.exists() else default

    def write(self, relative, value):
        path = self.root / relative
        data = canonical(value) + "\n"
        if len(data.encode()) > 64 * 1024 * 1024:
            raise ValueError("Ledger snapshot exceeds 64 MiB; start a separately authorized batch")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=".ledger-", delete=False) as stream:
                temporary = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


class RequestBudgetExceeded(RuntimeError):
    """A request would exceed an established request or API-failure budget."""


def _limit(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _decimal(value, name):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a nonnegative finite number") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")
    return number


def _budget_config(budget):
    if budget is None:
        return None
    token_limit = budget.get("max_total_tokens")
    if token_limit is not None:
        _limit(token_limit, "max_total_tokens")
    cost_limit = budget.get("max_estimated_cost")
    prices = (budget.get("input_per_million"), budget.get("output_per_million"))
    currency = budget.get("currency")
    tiers = budget.get("pricing_tiers")
    billing_policy = budget.get("billing_policy")
    if billing_policy is not None:
        validate_policy(billing_policy)
        if currency != "CNY" or tiers is not None or any(price is not None for price in prices):
            raise ValueError("Qwen tariff cannot be mixed with manually supplied prices")
    if tiers is not None:
        if not isinstance(tiers, list) or not tiers:
            raise ValueError("pricing_tiers must be a nonempty ordered list")
        normalized, previous = [], 0
        for tier in tiers:
            boundary = _limit(tier["max_prompt_tokens"], "max_prompt_tokens", 1)
            if boundary <= previous:
                raise ValueError("Pricing tier boundaries must be strictly increasing")
            normalized.append({"max_prompt_tokens": boundary, **{
                key: str(_decimal(tier.get(key), key).normalize()) for key in (
                    "input_per_million", "output_per_million", "cache_creation_per_million", "cache_hit_per_million")}})
            previous = boundary
        tiers = normalized
        if not isinstance(currency, str) or not currency:
            raise ValueError("Tiered cost accounting requires currency")
    elif billing_policy is None and (cost_limit is not None or any(price is not None for price in prices)):
        if None in prices or not isinstance(currency, str) or not currency:
            raise ValueError("Cost accounting requires both prices and currency")
    result = {"max_total_tokens": token_limit,
            "max_estimated_cost": str(_decimal(cost_limit, "max_estimated_cost").normalize()) if cost_limit is not None else None,
            "input_per_million": str(_decimal(prices[0], "input_per_million").normalize()) if prices[0] is not None else None,
            "output_per_million": str(_decimal(prices[1], "output_per_million").normalize()) if prices[1] is not None else None,
            "currency": currency}
    if tiers is not None:
        result["pricing_tiers"] = tiers
    if billing_policy is not None:
        result["billing_policy"] = deepcopy(billing_policy)
    if "carried_cost" in budget:
        result["carried_cost"] = str(_decimal(budget["carried_cost"], "carried_cost").normalize())
        if Decimal(result["carried_cost"]) and tiers is None and billing_policy is None and None in prices:
            raise ValueError("Carried cost requires configured prices and currency")
    return result


def validate_budget(budget):
    """Normalize a frozen budget without reading/writing any ledger files."""
    return _budget_config(budget)


def _reservation(reservation):
    if not isinstance(reservation, dict):
        raise ValueError("Budgeted requests need estimated prompt_tokens and completion_tokens reservations")
    return {key: _limit(reservation.get(key), key) for key in ("prompt_tokens", "completion_tokens")}


def _quote(usage, budget, reservation=False):
    if budget and budget.get("billing_policy"):
        return qwen_quote(usage, budget["billing_policy"], reservation=reservation)
    result = {"estimated_cost": None, "currency": budget.get("currency") if budget else None,
              "basis": "missing_usage", "tier_max_prompt_tokens": None, "cache_breakdown_complete": False}
    if not usage or any(usage.get(key) is None for key in ("prompt_tokens", "completion_tokens")):
        return result
    if budget is None or not (budget.get("pricing_tiers") or budget.get("input_per_million") is not None):
        return {**result, "basis": "pricing_unavailable"}
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    tiers = budget.get("pricing_tiers")
    if tiers:
        tier = next((tier for tier in tiers if prompt <= tier["max_prompt_tokens"]), None)
        if tier is None:
            raise ValueError("Prompt token count exceeds configured pricing tiers")
        result["tier_max_prompt_tokens"] = tier["max_prompt_tokens"]
        input_price, creation_price, hit_price = map(Decimal, (
            tier["input_per_million"], tier["cache_creation_per_million"], tier["cache_hit_per_million"]))
        created, hit = usage.get("cache_creation_tokens"), usage.get("cache_hit_tokens")
        complete = created is not None and hit is not None and created + hit <= prompt
        if not reservation and complete:
            input_cost = (prompt - created - hit) * input_price + created * creation_price + hit * hit_price
            result.update(basis="tier_cache_breakdown", cache_breakdown_complete=True)
        else:
            input_cost = prompt * max(input_price, creation_price)
            result["basis"] = "tier_reservation_no_cache_discount" if reservation else "tier_cache_missing_or_invalid_conservative"
        output_price = Decimal(tier["output_per_million"])
    else:
        input_cost = prompt * Decimal(budget["input_per_million"])
        output_price = Decimal(budget["output_per_million"])
        result["basis"] = "flat_uncached"
    result["estimated_cost"] = str((input_cost + completion * output_price) / Decimal(1_000_000))
    return result


def cost_quote(usage, budget, *, reservation=False):
    """Pure estimated quote from raw API usage; never an actual provider bill.

    Tier limits are inclusive. A reservation assumes no future cache hits.
    Missing/inconsistent cache counts price the entire prompt conservatively;
    missing prompt/completion counts cannot yield a complete quote. carried_cost
    is deliberately excluded: it is added once by the run ledger, not per call.
    """
    normalized = _budget_config(budget)
    return _quote(usage if normalized and normalized.get("billing_policy") else _usage(usage),
                  normalized, reservation)


def _estimated_cost(tokens, budget):
    amount = _quote(tokens, budget, reservation=True)["estimated_cost"]
    if amount is None:
        raise ValueError("Cannot price request reservation")
    return Decimal(amount) if amount is not None else None


def request_quote(request, budget):
    """Pure per-attempt audit using frozen request identity and raw API usage."""
    policy = deepcopy(budget["billing_policy"])
    metadata = request.get("metadata", {})
    if "cache_mode" in metadata:
        policy["cache_mode"] = metadata["cache_mode"]
    value = qwen_quote(request.get("raw_usage", request.get("usage")), policy)
    response_model = request.get("response_metadata", {}).get("model")
    from rsagent.billing.qwen import MODELS
    if response_model is not None and response_model not in MODELS:
        value.update(usage_complete=False, tariff_cost=None, basis="unexpected_response_model")
    if request.get("duplicate_provider_id"):
        value.update(usage_complete=False, tariff_cost=None, basis="duplicate_provider_id_needs_review")
    return value


def _qwen_budget_summary(state):
    budget = state["budget"]
    settled, inflight, uncertain = Decimal(0), Decimal(0), Decimal(0)
    settled_n = inflight_n = uncertain_n = exceeded = tokens = reserved_tokens = 0
    for request in state["requests"].values():
        reserved = request["reservation"]
        raw = request.get("raw_usage") or request.get("usage") or {}
        if not isinstance(raw, dict):
            raw = {}
        observed = {k: raw.get(k) if type(raw.get(k)) is int and raw[k] >= 0 else 0 for k in reserved}
        charged = {k: max(reserved[k], observed[k]) for k in reserved}
        exceeded += int(any(observed[k] > reserved[k] for k in reserved))
        value = request.get("billing_quote") or request_quote(request, budget)
        if request["error"] is False and value["usage_complete"]:
            settled += Decimal(value["tariff_cost"])
            settled_n += 1
            tokens += sum(observed.values())
        else:
            hold = _estimated_cost(reserved, budget)
            try:
                hold = max(hold, _estimated_cost(charged, budget))
            except ValueError:
                # Usage outside known tariff cannot safely release a reservation.
                # Consume the cap and require human review, not a guessed price.
                hold = max(hold, Decimal(budget["max_estimated_cost"] or "0"))
            tokens += sum(charged.values())
            reserved_tokens += sum(charged.values())
            if request["error"] is None:
                inflight += hold
                inflight_n += 1
            else:
                uncertain += hold
                uncertain_n += 1
    carried = Decimal(budget.get("carried_cost", "0"))
    committed = settled + inflight + uncertain + carried
    cap = Decimal(budget["max_estimated_cost"]) if budget["max_estimated_cost"] is not None else None
    token_cap = budget["max_total_tokens"]
    return {**budget, "basis": "qwen_usage_tariff_plus_separate_reservations", "provider_bill_hard_limit": False,
        "provider_bill_verified": False, "settled_tariff_cost": str(settled),
        "inflight_reserved_cost": str(inflight), "uncertain_reserved_cost": str(uncertain),
        "settled_requests": settled_n, "inflight_requests": inflight_n, "uncertain_requests": uncertain_n,
        "reserved_requests": inflight_n + uncertain_n, "estimated_committed_cost": str(committed),
        "estimated_current_cost": str(settled + inflight + uncertain), "estimated_carried_cost": str(carried),
        "estimated_committed_tokens": tokens, "estimated_reserved_tokens": reserved_tokens,
        "remaining_estimated_cost": str(cap - committed) if cap is not None else None,
        "remaining_estimated_tokens": token_cap - tokens if token_cap is not None else None,
        "requests_exceeding_reservation": exceeded,
        "overrun": (cap is not None and committed > cap) or (token_cap is not None and tokens > token_cap),
        "can_start": not ((cap is not None and committed >= cap) or (token_cap is not None and tokens >= token_cap))}


def _budget_summary(state):
    budget = state.get("budget")
    if budget is None:
        return None
    if budget.get("billing_policy"):
        return _qwen_budget_summary(state)
    committed_tokens, reserved_tokens, reserved_requests, settled_requests = 0, 0, 0, 0
    current_cost = Decimal(0) if budget.get("pricing_tiers") or budget["input_per_million"] is not None else None
    carried_cost = Decimal(budget.get("carried_cost", "0"))
    exceeding = 0
    for request in state["requests"].values():
        reserved, usage = request["reservation"], request["usage"] or {}
        complete = all(usage.get(key) is not None for key in reserved)
        settled = request["error"] is False and complete
        if settled:
            charged = {key: usage[key] for key in reserved}
            settled_requests += 1
        else:
            # An API error/partial response never releases a reservation. If a
            # reported component exceeds it, retain at least that observed use.
            charged = {key: max(value, usage.get(key) or 0) for key, value in reserved.items()}
            reserved_requests += 1
            reserved_tokens += sum(charged.values())
        exceeding += int(any((usage.get(key) or 0) > value for key, value in reserved.items()))
        committed_tokens += sum(charged.values())
        if current_cost is not None:
            known_cost = _quote(usage, budget)["estimated_cost"]
            if settled:
                current_cost += Decimal(known_cost)
            else:
                # Preserve the original financial reservation even if crossing
                # a caller's price tier happens to lower its numerical rate.
                current_cost += max(_estimated_cost(reserved, budget), _estimated_cost(charged, budget),
                                    Decimal(known_cost) if known_cost is not None else Decimal(0))
    committed_cost = current_cost + carried_cost if current_cost is not None else None
    token_limit = budget["max_total_tokens"]
    cost_limit = Decimal(budget["max_estimated_cost"]) if budget["max_estimated_cost"] is not None else None
    token_exhausted = token_limit is not None and committed_tokens >= token_limit
    cost_exhausted = cost_limit is not None and committed_cost >= cost_limit
    return {**budget, "basis": ("tiered_cache_pricing_with_conservative_reservations" if budget.get("pricing_tiers")
                                else "caller_estimated_reservations_and_configured_uncached_prices"),
            "provider_bill_hard_limit": False,
            "estimated_committed_tokens": committed_tokens,
            "estimated_reserved_tokens": reserved_tokens,
            "estimated_committed_cost": str(committed_cost) if committed_cost is not None else None,
            "estimated_current_cost": str(current_cost) if current_cost is not None else None,
            "estimated_carried_cost": str(carried_cost),
            "remaining_estimated_tokens": token_limit - committed_tokens if token_limit is not None else None,
            "remaining_estimated_cost": str(cost_limit - committed_cost) if cost_limit is not None else None,
            "reserved_requests": reserved_requests, "settled_requests": settled_requests,
            "requests_exceeding_reservation": exceeding,
            "overrun": (token_limit is not None and committed_tokens > token_limit)
                       or (cost_limit is not None and committed_cost > cost_limit),
            "can_start": not (token_exhausted or cost_exhausted)}


def _assert_reservation_fits(state, reservation):
    current = _budget_summary(state)
    if current is None:
        return
    if (current["max_total_tokens"] is not None
            and current["estimated_committed_tokens"] + sum(reservation.values()) > current["max_total_tokens"]):
        raise RequestBudgetExceeded("Next reservation exceeds estimated token budget")
    if current["max_estimated_cost"] is not None:
        cost = Decimal(current["estimated_committed_cost"]) + _estimated_cost(reservation, current)
        if cost > Decimal(current["max_estimated_cost"]):
            raise RequestBudgetExceeded("Next reservation exceeds estimated cost budget")


def _usage(usage):
    if usage is None:
        return None
    if not isinstance(usage, dict):
        raise ValueError("usage must be an API usage object or None")
    details = usage.get("prompt_tokens_details") or {}
    result = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", usage.get(
            "cache_hit_tokens", details.get("cached_tokens"))),
        "cache_miss_tokens": usage.get("prompt_cache_miss_tokens", usage.get("cache_miss_tokens")),
    }
    creation = usage.get("prompt_cache_creation_tokens", usage.get(
        "cache_creation_tokens", details.get("cache_creation_input_tokens")))
    if creation is not None:
        result["cache_creation_tokens"] = creation
    for key, value in result.items():
        if value is not None:
            _limit(value, key)
    if result["prompt_tokens"] is not None and result["completion_tokens"] is not None:
        # Cache counts describe the prompt; they are not additional tokens.
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result if any(value is not None for value in result.values()) else None


def _counts(max_requests, *, cache_creation=False):
    fields = (*TOKEN_FIELDS, "cache_creation_tokens") if cache_creation else TOKEN_FIELDS
    return {"max_requests": max_requests, "requests_started": 0, "requests_finished": 0,
            "requests_pending": 0, "api_failures": 0, "remaining_requests": max_requests,
            "usage": {**dict.fromkeys(fields, 0), "reported_requests": 0,
                      "missing_requests": 0, "field_missing_requests": dict.fromkeys(fields, 0)}}


def _add(counts, request):
    counts["requests_started"] += 1
    counts["remaining_requests"] -= 1
    if request["error"] is None:
        counts["requests_pending"] += 1
        return
    counts["requests_finished"] += 1
    counts["api_failures"] += int(request["error"])
    usage = request["usage"]
    aggregate = counts["usage"]
    aggregate["reported_requests" if usage is not None else "missing_requests"] += 1
    for key in aggregate["field_missing_requests"]:
        value = (usage or {}).get(key)
        if value is None:
            aggregate["field_missing_requests"][key] += 1
        else:
            aggregate[key] += value


class RequestLedger:
    """One host-owned ledger shared by workers and preserved across restarts.

    Call begin_request immediately before sending to the API. A reservation is
    never refunded: a crash cannot establish whether the remote call was sent.
    Missing token counts remain unknown; aggregate sums cover observed values.

    Optional token/cost admission uses caller estimates, NOT a provider billing
    guarantee. Full successful usage settles using configured flat prices or
    cache-aware pricing tiers. Errors and partial usage keep their reservations.
    """

    def __init__(self, root, max_requests, max_api_failures=3, *, budget=None):
        self._storage = _LedgerStorage(root)
        self.root = self._storage.root
        limits = {"max_requests": _limit(max_requests, "max_requests"),
                  "max_api_failures": _limit(max_api_failures, "max_api_failures", 1)}
        budget = _budget_config(budget)
        self.budget = deepcopy(budget)
        with self._storage.locked():
            state = self._storage.read("ledger.json")
            if state is None:
                self._storage.write("ledger.json", {"schema": "request_ledger_v1", **limits, "budget": budget,
                                                    "sessions": {}, "groups": {}, "requests": {}})
            elif any(state[key] != value for key, value in limits.items()) or state.get("budget") != budget:
                raise ValueError("Existing ledger limits cannot be changed by reopening it")

    def register_session(self, session_id, *, max_requests, group=None, group_limit=None):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        maximum = _limit(max_requests, "max_requests")
        if group is None and group_limit is not None:
            raise ValueError("group_limit requires a group")
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError("group must be a nonempty string")
        with self._storage.locked():
            state = self._storage.read("ledger.json")
            if group is not None:
                previous = state["groups"].get(group)
                if group_limit is None:
                    if previous is None:
                        raise ValueError("A new group needs group_limit")
                    group_limit = previous
                _limit(group_limit, "group_limit")
                if previous is not None and previous != group_limit:
                    raise ValueError("An existing group limit cannot be changed")
                state["groups"][group] = group_limit
            session = {"max_requests": maximum, "group": group}
            previous = state["sessions"].get(session_id)
            if previous is not None and previous != session:
                raise ValueError("An existing session cannot change its limit or group")
            state["sessions"][session_id] = session
            self._storage.write("ledger.json", state)
        return deepcopy(session)

    @staticmethod
    def _assert_global(state):
        if len(state["requests"]) >= state["max_requests"]:
            raise RequestBudgetExceeded("Global API request budget exhausted")
        failures = sum(request["error"] is True for request in state["requests"].values())
        if failures >= state["max_api_failures"]:
            raise RequestBudgetExceeded("Cumulative API failure budget exhausted")
        budget = _budget_summary(state)
        if budget is not None and not budget["can_start"]:
            raise RequestBudgetExceeded("Estimated token or cost budget exhausted")

    def assert_can_start(self):
        """Check global limits before starting a task; begin_request rechecks."""
        with self._storage.locked():
            self._assert_global(self._storage.read("ledger.json"))

    def begin_request(self, session_id, *, reservation=None, metadata=None):
        with self._storage.locked():
            state = self._storage.read("ledger.json")
            if (state.get("budget") or {}).get("billing_policy"):
                from rsagent.billing.qwen import ENDPOINT
                metadata = deepcopy(metadata or {})
                allowed = {"model", "base_url", "cache_mode", "attempt", "reservation_basis"}
                if set(metadata) - allowed:
                    raise ValueError("Billing metadata must not contain credentials or prompts")
                if (metadata.get("model", "qwen3.5-flash") != "qwen3.5-flash"
                        or metadata.get("base_url", ENDPOINT).rstrip("/") != ENDPOINT
                        or metadata.get("cache_mode") not in (None, "explicit")):
                    raise ValueError("Request does not match the frozen billing provider")
            self._assert_global(state)
            if session_id not in state["sessions"]:
                raise ValueError("Register the session before reserving a request")
            session = state["sessions"][session_id]
            requests = list(state["requests"].values())
            if sum(r["session_id"] == session_id for r in requests) >= session["max_requests"]:
                raise RequestBudgetExceeded(f"Session API request budget exhausted: {session_id}")
            group = session["group"]
            if group is not None:
                count = sum(state["sessions"][r["session_id"]]["group"] == group for r in requests)
                if count >= state["groups"][group]:
                    raise RequestBudgetExceeded(f"Group API request budget exhausted: {group}")
            if state.get("budget") is not None:
                reservation = _reservation(reservation)
                try:
                    _assert_reservation_fits(state, reservation)
                except ValueError as exc:
                    raise RequestBudgetExceeded(str(exc)) from exc
            request_id = f"request-{len(requests) + 1}"
            state["requests"][request_id] = {"session_id": session_id, "usage": None, "error": None}
            if (state.get("budget") or {}).get("billing_policy"):
                state["requests"][request_id].update(started_at=now_beijing().isoformat(), metadata=metadata)
            if state.get("budget") is not None:
                state["requests"][request_id]["reservation"] = reservation
            self._storage.write("ledger.json", state)
            return request_id

    def finish_request(self, request_id, usage=None, error=False, *, response_metadata=None):
        with self._storage.locked():
            state = self._storage.read("ledger.json")
            precise = bool((state.get("budget") or {}).get("billing_policy"))
            try:
                normalized = _usage(usage)
            except (ValueError, TypeError, AttributeError):
                if not precise:
                    raise
                normalized = None  # Keep malformed raw usage for investigation.
            request = state["requests"].get(request_id)
            if request is None:
                raise ValueError("Unknown request reservation")
            if request["error"] is not None:
                if (precise and request.get("raw_usage") == usage and request["error"] == bool(error)
                        and request.get("response_metadata") == (response_metadata or {})):
                    return  # Crash recovery of an identical settlement is idempotent.
                raise ValueError("Request already finished; usage cannot be counted twice")
            request.update(usage=normalized, error=bool(error))
            if precise:
                response_metadata = deepcopy(response_metadata or {})
                if set(response_metadata) - {"id", "model"}:
                    raise ValueError("Only response id/model belong in billing metadata")
                request.update(raw_usage=deepcopy(usage), response_metadata=response_metadata,
                               finished_at=now_beijing().isoformat())
                remote_id = response_metadata.get("id")
                request["duplicate_provider_id"] = bool(remote_id and any(
                    k != request_id and r.get("response_metadata", {}).get("id") == remote_id
                    for k, r in state["requests"].items()))
                request["billing_quote"] = request_quote(request, state["budget"])
            self._storage.write("ledger.json", state)

    def summary(self):
        with self._storage.locked():
            state = self._storage.read("ledger.json")
        return self.summarize(state)

    @staticmethod
    def summarize(state):
        """Summarize an atomic persisted snapshot without creating or changing files."""
        creation = bool((state.get("budget") or {}).get("pricing_tiers")) or any(
            "cache_creation_tokens" in (request["usage"] or {}) for request in state["requests"].values())
        result = {**_counts(state["max_requests"], cache_creation=creation), "max_api_failures": state["max_api_failures"],
                  "per_session": {key: {**_counts(value["max_requests"], cache_creation=creation), "group": value["group"]}
                                  for key, value in state["sessions"].items()},
                  "per_group": {key: _counts(value, cache_creation=creation) for key, value in state["groups"].items()},
                  "pending": []}
        for request_id, request in state["requests"].items():
            session_id = request["session_id"]
            session = result["per_session"][session_id]
            _add(result, request)
            _add(session, request)
            if session["group"] is not None:
                _add(result["per_group"][session["group"]], request)
            if request["error"] is None:
                result["pending"].append({"request_id": request_id, "session_id": session_id})
        result["can_start"] = (result["remaining_requests"] > 0
                               and result["api_failures"] < result["max_api_failures"])
        budget = _budget_summary(state)
        if budget is not None:
            result["budget"] = budget
            result["can_start"] = result["can_start"] and budget["can_start"]
        for group in result["per_group"].values():
            group["can_start"] = result["can_start"] and group["remaining_requests"] > 0
        for session in result["per_session"].values():
            group_open = session["group"] is None or result["per_group"][session["group"]]["can_start"]
            session["can_start"] = result["can_start"] and session["remaining_requests"] > 0 and group_open
        return result
