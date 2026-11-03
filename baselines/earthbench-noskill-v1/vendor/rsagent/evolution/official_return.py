"""Read-only library assistance and an explicit auxiliary-to-official return checkpoint.

The checkpoint records model claims, not semantic certification. No arguments,
official algorithms or numerical results are repaired on the model's behalf.
"""
import ast
import json
from pathlib import Path
import re

from rsagent.integrations.tool_argument_guard import validate_path_argument_shapes


RETURN_PROTOCOL = """
OFFICIAL RETURN WORKFLOW (same continuous context, no extra request budget).
Temporary code is exploration, not a completed solution. After auxiliary calculations,
decompose the computation into operations and look for matching official tools using
search_official_tools; inspect_official_tool exposes exact schemas and read-only source.
Compare inputs, dates, units, selection conditions and aggregation with your exploratory
code. If a call failed, check_official_arguments can diagnose path shapes, required keys
and access without executing anything; it does NOT convert strings into arrays.
Use actual arrays when declared; blaming the tool after repeating the same malformed
argument is not a repair. Helpers are metadata/workflow tools, not additional calculators.

Return to ordinary official tool calls on task inputs and reproduce the answer-critical
operations. A final difference/division of auxiliary-derived numbers does not suffice.
Once done, call record_official_return: state the auxiliary role, map operations to official
tool names and observed O IDs, describe the actual repair, and list remaining gaps.
For mere file inspection say inspection_only. For reproduced calculations say reproduced.
If reproduction remains impossible say unresolved, identify the tried tools and exact gap;
do not claim a verified official answer or tune calculations to the choices.
The host checks only record structure and that cited calls exist; it cannot certify semantic
equivalence. It will ask you to complete a missing return record before accepting a final
response after auxiliary commands. A further command or official call makes the record stale.
Reserve roughly the final third of the model API request budget for official reproduction, the short record,
and a final response. The budget counts model API requests, NOT tool invocations. Batch multiple
independent official tool calls in one response; the host executes them sequentially and returns
their results together. Split large collections into manageable batches within output/context/tool
limits, and wait for results before dependent aggregation. More files than API requests does not
by itself justify sampling or replacing official computations with code. Do not waste the budget on
repeated format repair; a truthful unresolved record is acceptable. Advice is fallible and
must not override this workflow. Do not persist scripts as tools or change official code.
"""


def _definition(name, description, properties, required):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties,
                             "required": required, "additionalProperties": False}}


class OfficialReturnWorkflow:
    def __init__(self, official, recorder, source_dir, argument_check=None):
        self.official, self.recorder = official, recorder
        self.source_dir = Path(source_dir).resolve()
        self.argument_check = argument_check
        self.catalog = {d["name"]: d for d in official.definitions}
        self.auxiliary_count = 0
        self.last_aux_official_count = 0
        self.record = None
        text = {"type": "string"}
        helpers = [
            _definition("search_official_tools", "Search the fixed official catalog by English operation keywords; no computation.",
                        {"query": text}, ["query"]),
            _definition("inspect_official_tool", "Read exact official schema and paginated implementation; no code execution.",
                        {"tool_name": text, "source_offset": {"type": "integer", "minimum": 0}}, ["tool_name"]),
            _definition("check_official_arguments", "Dry-run path shape/access and required-key checks, NOT full domain validation. Supply a JSON object as text; nested arrays must be arrays, not strings. No repair or execution.",
                        {"tool_name": text, "arguments_json": text}, ["tool_name", "arguments_json"]),
            _definition("record_official_return", "Record how exploration returns to official computation. This records claims, not automatic correctness certification.",
                        {"status": {"type": "string", "enum": ["inspection_only", "reproduced", "unresolved"]},
                         "auxiliary_role": text, "repair": text, "remaining_gap": text,
                         "operations": {"type": "array", "items": {"type": "object", "properties": {
                             "operation": text, "tool_name": text,
                             "evidence_ids": {"type": "array", "items": text}},
                             "required": ["operation", "tool_name", "evidence_ids"]}}},
                        ["status", "auxiliary_role", "repair", "remaining_gap", "operations"]),
        ]
        self.definitions = list(official.definitions) + helpers
        self.names = set(self.catalog) | {d["name"] for d in helpers}

    def note_auxiliary(self):
        self.auxiliary_count += 1
        self.last_aux_official_count = len(self.official.observations)
        self.record = None
        self.recorder.log("official_return_pending", auxiliary_count=self.auxiliary_count)
        return ("\n[Workflow notice, not command output] Auxiliary results are provisional. "
                "Find/inspect the corresponding official tools, check for misuse, reproduce the "
                "answer-critical calculations and record_official_return before your final answer. "
                "For inspection-only commands or unresolved issues, report that honestly.")

    def review_final(self, final_text):
        if self.auxiliary_count and self.record is None:
            return ("Official-return checkpoint is missing or stale after auxiliary commands. "
                    "The previous answer is a draft, not accepted completion. Use the remaining "
                    "budget to inspect/repair/reproduce with official tools and record_official_return, "
                    "or record an honest unresolved gap. Do not fabricate reproduction.")
        return None

    def _source(self, name, offset):
        if type(offset) is not int or offset < 0:
            raise ValueError("source_offset must be a nonnegative integer")
        matches = []
        for path in sorted(self.source_dir.rglob("*.py")):
            if not path.resolve().is_relative_to(self.source_dir):
                continue
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    lines = source.splitlines()[node.lineno - 1:node.end_lineno]
                    matches.append({"file": str(path.relative_to(self.source_dir)),
                        "line_start": node.lineno + offset, "source": "\n".join(lines[offset:offset + 80]),
                        "next_offset": offset + 80 if offset + 80 < len(lines) else None})
        return matches

    def execute(self, name, arguments):
        if name in self.catalog:
            self.record = None
            return self.official.execute(name, arguments)
        if name == "search_official_tools":
            words = set(re.findall(r"[a-z0-9]+", arguments["query"].lower()))
            ranked = []
            for tool, definition in self.catalog.items():
                title = set(tool.split("_"))
                body = set(re.findall(r"[a-z0-9]+", definition.get("description", "").lower()))
                rank = 3 * len(words & title) + len(words & body)
                if rank:
                    ranked.append((rank, tool))
            result = {"matches": [{"name": n, "description": self.catalog[n].get("description", "")}
                       for _, n in sorted(ranked, key=lambda x: (-x[0], x[1]))[:8]],
                       "note": "Lexical catalog search only; try alternate English keywords if needed."}
        elif name in {"inspect_official_tool", "check_official_arguments"}:
            tool = arguments["tool_name"]
            if tool not in self.catalog:
                raise ValueError("Unknown official tool: " + tool)
            definition = self.catalog[tool]
            if name == "inspect_official_tool":
                result = {"definition": definition, "implementation": self._source(tool, arguments.get("source_offset", 0))}
            else:
                try:
                    values = json.loads(arguments["arguments_json"])
                    schema = definition["input_schema"]
                    validate_path_argument_shapes(values, schema, tool_name=tool)
                    missing = set(schema.get("required", [])) - values.keys()
                    unexpected = values.keys() - schema.get("properties", {}).keys()
                    if missing or unexpected:
                        raise ValueError(f"Missing keys: {sorted(missing)}; unexpected keys: {sorted(unexpected)}")
                    if self.argument_check:
                        self.argument_check(values, definition)
                    result = {"checks_passed": True, "argument_types": {k: type(v).__name__ for k, v in values.items()}}
                except (ValueError, TypeError) as exc:
                    result = {"checks_passed": False, "error": str(exc)}
                result.update(executed=False, converted=False, scope="Required keys and path shape/access only; not full domain validation")
        elif name == "record_official_return":
            status = arguments.get("status")
            if status not in {"inspection_only", "reproduced", "unresolved"}:
                raise ValueError("Use inspection_only, reproduced or unresolved")
            for field in ("auxiliary_role", "repair", "remaining_gap"):
                if not isinstance(arguments.get(field), str):
                    raise ValueError(field + " must be text")
            operations = arguments.get("operations")
            if not isinstance(operations, list):
                raise ValueError("operations must be an array")
            known = {o["official_id"]: o for o in self.official.observations}
            cited = []
            for operation in operations:
                tool = operation.get("tool_name")
                if tool not in self.catalog or not isinstance(operation.get("operation"), str):
                    raise ValueError("Each operation needs an explanation and an existing official tool_name")
                ids = operation.get("evidence_ids")
                if not isinstance(ids, list) or any(not isinstance(oid, str) for oid in ids):
                    raise ValueError("evidence_ids must be an array of O IDs")
                for oid in ids:
                    if oid not in known or known[oid]["name"] != tool:
                        raise ValueError("Evidence ID must name an observed call of the mapped tool: " + oid)
                    if status == "reproduced" and not known[oid]["success"]:
                        raise ValueError("Failed calls cannot certify reproduction")
                cited.extend(ids)
            if status == "reproduced":
                if not operations or any(not o["evidence_ids"] for o in operations):
                    raise ValueError("Reproduction needs cited official calls for each mapped operation")
                if self.auxiliary_count and not any(int(oid[1:]) > self.last_aux_official_count for oid in cited):
                    raise ValueError("No cited official reproduction after the last auxiliary command; report unresolved if necessary")
            if status == "unresolved" and not arguments["remaining_gap"].strip():
                raise ValueError("State the unresolved gap")
            self.record = {**arguments, "auxiliary_count": self.auxiliary_count,
                           "semantic_compliance": "model_claim_requires_trajectory_audit"}
            self.recorder.log("official_return_record", **self.record)
            result = {"recorded": True, "semantic_certification": False}
        else:
            raise ValueError("Unknown workflow tool: " + name)
        self.recorder.log("official_library_assistance", name=name, arguments=arguments, result=result)
        return json.dumps(result, ensure_ascii=False)


class OfficialToolLookup(OfficialReturnWorkflow):
    """Expose only read-only discovery and argument checks, without a return gate."""

    LOOKUP_NAMES = {"search_official_tools", "inspect_official_tool", "check_official_arguments"}

    def __init__(self, official, recorder, source_dir, argument_check=None):
        super().__init__(official, recorder, source_dir, argument_check)
        self.definitions = [tool for tool in self.definitions if tool["name"] in self.LOOKUP_NAMES]
        self.names = self.LOOKUP_NAMES

    def execute(self, name, arguments):
        if name not in self.names:
            raise ValueError("Unknown lookup tool: " + name)
        return super().execute(name, arguments)

    def close(self):
        self.official.close()
