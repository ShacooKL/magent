"""Session-local checked argument dispatch; no new domain algorithms or auto-repair."""
from copy import deepcopy
import json
from uuid import uuid4

from rsagent.evolution.official_return import OfficialReturnWorkflow, _definition


PREPARED_RETURN_PROTOCOL = """
CHECKED OFFICIAL DISPATCH, EARLY RETURN WORKFLOW.
You may explore with temporary code, but answer-critical computations must be reproduced
by the fixed official library. Its functions are available through tool calls, NOT importable
Python packages inside run_command. Use read-only search_official_tools and
inspect_official_tool to find and understand operations, not shell module guessing.

If direct calls repeatedly produce malformed arguments, use check_official_arguments:
provide tool_name and arguments_json (a JSON object encoded as text). INSIDE that object,
path arrays must be genuine JSON arrays, not strings containing arrays. The host does not
repair types, choose files, fill parameters, or change thresholds. A passing path/key check
returns a session-local call_id retaining exactly your parsed arguments; no tool has run yet.
Then invoke execute_checked_official_call(call_id). Do NOT regenerate the argument object
in another direct call. The dispatcher runs the named official tool through the SAME access
checks and runtime, and returns its ordinary O evidence. Each ID is single-use even on error;
to change parameters or retry, check a new object. This is a transport adapter, not an extra
calculator. Passing prechecks does not prove files exist or satisfy the full tool contract.

After auxiliary calculations, map their operations to existing official tools and reproduce
them on the original task inputs. Inspect selection conditions, units, dates and aggregation.
Do not just pass auxiliary answers into an official difference or division. Batch independent
tool calls; use the last third of the SAME model API request budget for official reproduction and final
reporting. The host gives early, appended workflow reminders without resetting context.
The budget counts model API requests, NOT tool invocations. A single model response may contain
multiple independent official tool calls, executed sequentially by the host. Split large file
collections into manageable batches; inspect returned results before dependent aggregation.
Do not infer that full coverage is impossible just because there are more files than API requests.

Use record_official_return to briefly map operations to official O IDs and identify repairs
and remaining gaps. Record inspection_only or unresolved honestly when appropriate. If the
budget is running out, prioritize actual computation and an honest final response over record
format repair. A missing record is audited but DOES NOT block final submission in this version;
accepting your answer does NOT certify the official chain. Explicitly separate auxiliary
results from official results in the final response; do not claim reproduction unless achieved.
Do not fit numbers to options, persist a generated tool library, or modify official code.
"""


class PreparedOfficialReturnWorkflow(OfficialReturnWorkflow):
    """An explicit, one-use adapter binds execution to model-supplied checked values."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepared = {}
        self.parameter_repair_pending = False
        definition = _definition("execute_checked_official_call",
            "Execute the exact official tool/arguments retained by a passing check_official_arguments. "
            "Supply only its call_id; no argument override, auto-repair, or new algorithm. "
            "The one-use ID is consumed even if execution fails. Ordinary official evidence is returned.",
            {"call_id": {"type": "string"}}, ["call_id"])
        self.definitions.append(definition)
        self.names.add(definition["name"])

    def execute(self, name, arguments):
        if name == "execute_checked_official_call":
            if set(arguments) != {"call_id"} or not isinstance(arguments["call_id"], str):
                raise ValueError("Supply only the call_id; overrides are not accepted")
            call_id = arguments["call_id"]
            if call_id not in self._prepared:
                raise ValueError("Unknown, foreign-session or already consumed call_id; check a new argument object")
            tool, stored = self._prepared.pop(call_id)
            values = deepcopy(stored)
            self.recorder.log("checked_official_dispatch", call_id=call_id, name=tool,
                              arguments=values, automatic_repair=False)
            # Recheck the current access state (e.g. changed symlinks) before execution.
            # GuardedEarth will also apply its normal checks; no alternate execution path.
            if self.argument_check:
                self.argument_check(values, self.catalog[tool])
            result = super().execute(tool, values)
            self.parameter_repair_pending = False
            return result
        if name == "check_official_arguments" and len(self._prepared) >= 64:
            raise ValueError("64 pending calls retained; execute existing IDs before checking more")
        try:
            output = super().execute(name, arguments)
        except Exception as exc:
            if name in self.catalog and "JSON-encoded array" in str(exc):
                self.parameter_repair_pending = True
            raise
        if name == "check_official_arguments":
            result = json.loads(output)
            if result["checks_passed"]:
                values = json.loads(arguments["arguments_json"])
                call_id = "checked_" + uuid4().hex
                self._prepared[call_id] = (arguments["tool_name"], deepcopy(values))
                self.recorder.log("official_arguments_retained", call_id=call_id,
                    name=arguments["tool_name"], arguments=values, automatic_repair=False)
                result.update(call_id=call_id, next_action="Call execute_checked_official_call with only this ID; do not regenerate arguments",
                              single_use=True)
                output = json.dumps(result, ensure_ascii=False)
        return output

    def step_feedback(self, step, total):
        """Appended to the budget message; never edit the cached system/history prefix."""
        remaining = total - step
        if remaining == 0:
            return ("Official-return workflow: final model API request. Submit your supported answer and disclose "
                    "any unverified auxiliary dependence; missing return paperwork will not block it.")
        parts = []
        if self.parameter_repair_pending:
            parts.append("Malformed array call detected. Use check_official_arguments then "
                         "execute_checked_official_call; do not repeat stringified arrays.")
        if self._prepared:
            parts.append(f"{len(self._prepared)} checked calls await explicit execution; a passing check is not execution.")
        if self.auxiliary_count and self.record is None:
            parts.append("Auxiliary exploration has no current return record. Now map its operations "
                         "to official tools and reproduce the answer-critical calculations.")
        if step >= max(1, total * 2 // 3):
            parts.append("Return phase: stop optional exploration; prioritize official computation "
                         "and reserve one model API request for the final answer. Report unresolved gaps honestly.")
        return "Official-return workflow: " + " ".join(parts) if parts else ""

    def review_final(self, final_text):
        self.recorder.log("official_return_final_audit",
            record_present=self.record is not None, auxiliary_count=self.auxiliary_count,
            pending_checked_calls=len(self._prepared), final_submission_blocked=False,
            semantic_compliance="not_certified_requires_trajectory_audit")
        return None

    def close(self):
        self._prepared.clear()
        super().close()
