"""Fixed official tool library, ephemeral exploration and auditable answer evidence."""
import re


SOLVE_PROTOCOL = """
FIXED OFFICIAL LIBRARY, OPEN AUXILIARY EXPLORATION.
This benchmark protocol assumes the task can be solved using the 104 available official
Earth tools. This is a task-design premise, not proof that every input or reference answer is valid.
You MAY use run_command, write_file and edit_file for temporary investigation, numerical checks,
debugging and hypothesis tests in your isolated workspace. Such code is session-local: do not
register new tools, modify official implementations, or establish a reusable generated-tool library.
The host retains code and logs for audit only, not as reusable capabilities for future tasks.

Your FINAL solution's answer-determining numerical operations must be executed with the official
104 tools, including simple arithmetic where provided. File inspection, choosing a workflow and
explaining a result are allowed. Auxiliary computations may suggest a method, but cannot substitute
for the official computation chain. Re-execute the relevant computation with official tools on
authorized task inputs. Do not feed auxiliary-derived answers/artifacts into a trivial official
operation merely to make them appear official. Do not silently overwrite official outputs with
auxiliary results. Generated scripts are not additional official tools.

When a result is unreasonable or does not match the question/options, FIRST check whether your
answer actually follows these restrictions and has an official calculation chain. SECOND inspect
whether you omitted or misused a tool: selection, parameters/defaults, file/date pairing, output
semantics, units and operation order. Correct and rerun the affected official operations. Only
then consider unresolved data/tool/task inconsistencies. Record uncertainty if evidence remains
insufficient; do not reverse-engineer constants, adjust quantities to fit options, or invent success.

Official calls return evidence labels O1, O2, etc. Before the final answer briefly state the
official tool chain, cite the key successful results as [O1], [O2], etc, and explain any unresolved
gap. Report any auxiliary-code checks separately; they are not official answer evidence. Cite all
answer-critical operations, not merely an unrelated last call. Finish with <Answer>...</Answer>.
Reference checks establish only that calls occurred, not that your chain is correct or complete.
"""

SKILL_FOCUS = """
LEARNING TARGET: reusable guidance for using the fixed 104-tool library, not new tool generation.
Focus on observable conditions for tool selection, input/parameter/default semantics, interpretation
of returned quantities, composition/order of official operations, and diagnosing omitted or misused
tools. Ground advice in actual recorded choices and results; avoid generic reminders without an
applicable action. Distinguish auxiliary exploration code from the official final calculation chain.
The target solver MAY use temporary run_command/code for investigation, but answer-determining
calculations must be reproduced with the official tools. Do not recommend publishing or reusing
temporary code as a new capability. Cite a genuine capability gap rather than inventing a tool.
When diagnosing unreasonable results, first check final-chain restriction compliance, then omitted
or misused tools, then unresolved data/tool/task issues. The task is assumed solvable by official
tools, but that is not ground truth and does not justify fitting results to answer options.
Keep Skills conditional and fallible. Do not claim experimental verification just because code ran.
"""


class OfficialEvidenceProvider:
    """Label unchanged official results; the labels are not semantic certification."""
    def __init__(self, provider, recorder):
        self.provider, self.recorder = provider, recorder
        self.definitions, self.names = provider.definitions, provider.names
        self.observations = []

    def execute(self, name, arguments):
        if name not in self.names:
            raise ValueError("Not an official tool")
        oid = "O" + str(len(self.observations) + 1)
        row = {"official_id": oid, "name": name, "arguments": arguments}
        try:
            result = self.provider.execute(name, arguments)
        except Exception as exc:
            row.update(success=False, error=str(exc))
            self.observations.append(row)
            self.recorder.log("official_tool_observation", **row)
            raise RuntimeError(f"Official call {oid} failed: {exc}") from exc
        row.update(success=True, output=result)
        self.observations.append(row)
        self.recorder.log("official_tool_observation", **row)
        return f"[Official evidence {oid}]\n{result}"

    def close(self):
        self.provider.close()


def audit_official_citations(final_text, observations):
    cited = list(dict.fromkeys(re.findall(r"\[(O[1-9][0-9]*)\]", final_text)))
    known = {row["official_id"]: row for row in observations}
    unknown = [oid for oid in cited if oid not in known]
    failed = [oid for oid in cited if oid in known and not known[oid]["success"]]
    return {"cited_ids": cited, "unknown_ids": unknown, "failed_ids": failed,
            "references_valid": bool(cited) and not unknown and not failed,
            "official_calls": len(observations),
            "semantic_compliance": "not_certified_requires_trajectory_audit",
            "warning": "A real citation does not prove numerical provenance, completeness or correctness."}
