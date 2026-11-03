"""Host-bounded, one-request output-limit overrides requested by the Agent."""
from rsagent.evolution.tool_registry import canonical


class NextOutputLimit:
    name = "set_next_output_limit"

    def __init__(self, default, ceiling, remaining_requests, recorder=None):
        if type(default) is not int or type(ceiling) is not int or not 0 < default < ceiling:
            raise ValueError("Output limits must be positive integers with ceiling > default")
        self.default, self.ceiling = default, ceiling
        self.remaining_requests, self.recorder = remaining_requests, recorder
        self.pending = None
        self.names = {self.name}
        self.definitions = [{
            "name": self.name,
            "description": (
                f"Temporarily raise the output-token limit of ONLY the next API request. "
                f"Default={default}, maximum allowed={ceiling}. Use before a long generation or "
                "after a truncation notice. Supply an absolute token limit, not an increment. "
                "Automatically resets to default after that request attempt, even on error or truncation. "
                "Does not raise request-count, context, or provider limits and cannot resume an already "
                "truncated reply. A later successful call before the next request replaces the pending value."
            ),
            "input_schema": {"type": "object", "properties": {
                "max_output_tokens": {"type": "integer", "minimum": default + 1, "maximum": ceiling},
                "reason": {"type": "string", "description": "Why the next response needs more output space."}},
                "required": ["max_output_tokens", "reason"], "additionalProperties": False}
        }]

    def execute(self, name, arguments):
        if name != self.name or not isinstance(arguments, dict) or set(arguments) != {"max_output_tokens", "reason"}:
            raise ValueError("Expected max_output_tokens and reason for set_next_output_limit")
        limit, reason = arguments["max_output_tokens"], arguments["reason"]
        if type(limit) is not int or not self.default < limit <= self.ceiling:
            raise ValueError(f"max_output_tokens must be an integer in {self.default + 1}..{self.ceiling}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a nonempty string")
        remaining = self.remaining_requests()
        if remaining <= 0:
            raise ValueError("No API requests remain; this tool cannot increase the request-count budget")
        previous = self.pending
        self.pending = limit
        result = {"success": True, "applies_to": "next_api_request_only", "max_output_tokens": limit,
                  "default_after_next_request": self.default, "remaining_api_requests": remaining,
                  "replaced_pending_limit": previous}
        if self.recorder:
            self.recorder.log("next_output_limit_requested", **result, reason=reason)
        return canonical(result)

    def consume(self):
        """Called once immediately before an API attempt; never sticky or auto-retried."""
        limit = self.pending if self.pending is not None else self.default
        self.pending = None
        return limit
