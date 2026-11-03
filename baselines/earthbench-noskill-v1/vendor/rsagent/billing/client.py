"""One physical request per call; retries must go THROUGH this wrapper again."""
import json

from rsagent.billing.qwen import policy_for


def metadata(client, attempt=1):
    # No API key, prompts, response text or environment variables in the ledger.
    config = {"provider": "openai", "model": client.model, "base_url": client.base_url,
              "cache_mode": client.cache_mode}
    policy_for(config)
    return {k: config[k] for k in ("model", "base_url", "cache_mode")} | {
        "attempt": attempt, "reservation_basis": "utf8_bytes_plus_4096_upper_estimate_not_billed_tokens"}


def reservation(system, messages, tools, max_tokens):
    if type(max_tokens) is not int or not 1 <= max_tokens <= 65536:
        raise ValueError("Unsupported output token limit")
    # Unknown server-side framing prevents claiming a hard mathematical bound.
    # A conservative reservation is released once full usage is received.
    size = len(json.dumps([system, messages, tools], ensure_ascii=False, sort_keys=True).encode())
    return {"prompt_tokens": size + 4096, "completion_tokens": max_tokens}


def settle(ledger, request_id, response):
    ledger.finish_request(request_id, response.get("raw_usage", response.get("usage")),
        response_metadata={"id": response.get("id"), "model": response.get("model")})


class BillingClient:
    """Explicit host-owned ledger and session required, no hidden global budget."""
    def __init__(self, client, ledger, session_id):
        if not ledger.budget or not ledger.budget.get("billing_policy"):
            raise ValueError("Qwen billing client requires a frozen Qwen tariff budget")
        metadata(client)  # Reject unsupported providers BEFORE sending anything.
        self.client, self.ledger, self.session_id = client, ledger, session_id

    def create_message(self, system, messages, tools, max_tokens=8192):
        request_id = self.ledger.begin_request(self.session_id,
            reservation=reservation(system, messages, tools, max_tokens), metadata=metadata(self.client))
        try:
            response = self.client.create_message(system, messages, tools, max_tokens=max_tokens)
        except Exception:
            self.ledger.finish_request(request_id, error=True)
            raise
        settle(self.ledger, request_id, response)
        return response
