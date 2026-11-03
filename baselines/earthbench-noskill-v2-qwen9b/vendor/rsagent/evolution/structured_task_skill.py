"""Structured diagnosis/Skill submission with session-local evidence handles.

Handles certify authorized read coverage, never correctness, relevance or causality.
"""
from copy import deepcopy
import json
from pathlib import Path

from rsagent.evolution.failure_task_skill import FailureTaskSkillExtractionProvider
from rsagent.evolution.grounded_task_skill import GroundedTaskSkillExtractionProvider
from rsagent.evolution.tool_registry import canonical, digest
from rsagent.runtime.jsonio import dump


class StructuredTaskSkillProvider:
    def __init__(self, workspace, diagnosis_output, skill_output, trajectory, failure_episode_ids,
                 skill_prompt, recorder=None, *, include_successful_sources=False,
                 allow_same_task_contrast=False, deferred_validation=False):
        self.workspace = Path(workspace).resolve()
        self.diagnosis_output, self.skill_output = Path(diagnosis_output), Path(skill_output)
        for output in (self.diagnosis_output, self.skill_output):
            resolved = output.resolve()
            if resolved.is_relative_to(self.workspace) or self.workspace.is_relative_to(resolved):
                raise ValueError("Publication stores must be outside workspace")
        if type(include_successful_sources) is not bool:
            raise ValueError("include_successful_sources must be boolean")
        if type(deferred_validation) is not bool:
            raise ValueError("deferred_validation must be boolean")
        self.deferred_validation = deferred_validation
        self.focus_episode_ids = set(failure_episode_ids)
        if type(allow_same_task_contrast) is not bool:
            raise ValueError("allow_same_task_contrast must be boolean")
        self.allow_same_task_contrast = allow_same_task_contrast
        if allow_same_task_contrast:
            selected = [e for e in trajectory["episodes"] if e["episode_id"] in self.focus_episode_ids]
            task_refs = {e.get("task_evidence_id") for e in selected}
            if (not include_successful_sources or len(selected) < 2
                    or len(task_refs) != 1 or None in task_refs
                    or not task_refs.issubset(trajectory["records"])):
                raise ValueError("Contrast requires explicitly authorized repeats sharing one task evidence record")
        if include_successful_sources:
            self.tracker = GroundedTaskSkillExtractionProvider(workspace, diagnosis_output, trajectory)
            if not self.focus_episode_ids or any(
                    eid not in self.tracker._episodes or
                    type(self.tracker._episodes[eid].get("ending", {}).get("correct")) is not bool
                    for eid in self.focus_episode_ids):
                raise ValueError("Require authorized source episodes with explicit outcome feedback")
        else:
            self.tracker = FailureTaskSkillExtractionProvider(workspace, diagnosis_output, trajectory, failure_episode_ids)
        self.trajectory, self.recorder, self.skill_prompt = trajectory, recorder, skill_prompt
        self.actions = {a["call_evidence_id"]: (n["episode_id"], a)
                        for n in trajectory["nodes"] for a in n["actions"]}
        self.handles, self.by_call = {}, {}
        self.stage, self.complete, self.diagnosis = "diagnosis", False, None
        string = {"type": "string", "minLength": 1}
        refs = {"type": "array", "items": string, "description": "Optional additional E handles; the saved diagnosis anchor is inherited automatically."}
        self.definitions = []
        def tool(name, description, properties, required):
            self.definitions.append({"name": name, "description": description,
                "input_schema": {"type": "object", "properties": properties,
                                 "required": required, "additionalProperties": False}})
        diagnosis = {k: dict(string) for k in ("evidence_handle", "task_requirement", "actual_operation",
            "explanation", "final_link", "alternative", "uncertainty")}
        diagnosis["alignment"] = {"type": "string", "enum": ["consistent", "inconsistent", "unresolved"]}
        tool("save_diagnosis", "Save ONE provisional diagnosis from a fully read E handle. IDs, original references and persistence are automatic; no JSON file or exact quotation. Returns a nonterminal checkpoint. Checks coverage, not truth.",
             {**diagnosis, "evidence_handles": refs}, list(diagnosis))
        skill = {k: dict(string) for k in ("title", "when", "rationale", "self_check")}
        skill["guidance"] = {"type": "array", "items": string, "minItems": 1, "maxItems": 6}
        skill["limitations"] = {"type": "array", "items": string, "minItems": 1, "maxItems": 4}
        if deferred_validation:
            skill.update(claim=dict(string), validation_method=dict(string),
                claim_type={"type": "string", "enum": ["failure_mechanism", "successful_procedure",
                    "implementation_equivalence", "delivery_requirement", "strategy"]})
        tool("save_skill", "Save ONE advisory task Skill after diagnosis, inheriting its evidence automatically. self_check explains consistency with your diagnosis and uncertainty. No file, ID, nested self_checks or quotation required. Successful save ends the session; not a utility certification.",
             {**skill, "evidence_handles": refs}, list(skill))
        tool("finish_without_skill", "Finish explicitly with no candidate and explain why. Valid during either phase.", {"reason": string}, ["reason"])
        self.names = {d["name"] for d in self.definitions}

    def inspection_progress(self):
        inspected = {eid for eid, refs in self.tracker._triples
                     if eid in self.focus_episode_ids and all(self.tracker._fully_read(ref) for ref in refs)}
        return {"fully_inspected_focus_episode_ids": sorted(inspected),
                "remaining_focus_episode_ids": sorted(self.focus_episode_ids - inspected),
                "scope": "Read coverage, not correctness or learning benefit"}

    def wrap_trajectory(self, provider):
        owner = self
        class Wrapped:
            view, names = provider.view, provider.names
            definitions = deepcopy(provider.definitions)
            for definition in definitions:
                if definition["name"] in {"trajectory_action_evidence", "trajectory_evidence"}:
                    definition["description"] += " Returned E handles bind original evidence automatically; read any missing required pages before save_diagnosis."
                    if provider.view.data.get("decision_trace"):
                        definition["description"] += " Fully read historical statements linked to your selected call are included automatically as model claims, never execution certification. Statements alone cannot create ready E handles."

            def execute(self, name, arguments):
                result = json.loads(provider.execute(name, arguments))
                if name not in {"trajectory_action_evidence", "trajectory_evidence"}:
                    return canonical(result)
                pages = result["records"] if name == "trajectory_action_evidence" else [result]
                for page in pages:
                    if "text" in page:
                        owner.tracker.note_evidence_read(page["id"], page["offset"], max(1, len(page["text"])))
                    if page["id"] in owner.actions:
                        owner._bind(page["id"])
                touched = {page["id"] for page in pages}
                handles = [owner._info(h) for h in owner.handles
                           if touched.intersection(owner._required(h))]
                if provider.view.data.get("presentation") == "execution_view_v2" and name == "trajectory_action_evidence":
                    handles = [h for h in handles if h["call_evidence_id"] == arguments["call_evidence_id"]]
                result["evidence_handles"] = handles
                result["citation_instruction"] = "Use a ready E handle in save_diagnosis; original IDs/read ranges are bound automatically. Coverage is not understanding or proof."
                if owner.recorder:
                    owner.recorder.log("evidence_handle_update", handles=handles)
                return canonical(result)
        return Wrapped()

    def _bind(self, call_id):
        if call_id not in self.by_call:
            handle = "E" + str(len(self.handles) + 1)
            self.handles[handle] = call_id
            self.by_call[call_id] = handle

    def _required(self, handle):
        episode_id, action = self.actions[self.handles[handle]]
        ep = self.tracker._episodes[episode_id]
        return [ep.get("task_evidence_id"), action["call_evidence_id"], action.get("result_evidence_id"),
                ep.get("ending_evidence_id"), "tool_definition:" + action["name"],
                *action.get("evidence_requirements", []),
                *[eid for eid in action.get("decision_evidence_ids", []) if self.tracker._fully_read(eid)]]

    def _info(self, handle):
        episode_id, _ = self.actions[self.handles[handle]]
        required = self._required(handle)
        missing = []
        for eid in required:
            if eid not in self.tracker.evidence:
                missing.append({"evidence_id": eid, "unavailable": True})
                continue
            if not self.tracker._fully_read(eid):
                size = len(self.tracker.evidence[eid]["text"])
                start = 0
                for left, right in self.tracker._read_intervals.get(eid, []):
                    if left > start:
                        break
                    start = max(start, right)
                missing.append({"evidence_id": eid, "offset": start, "length": max(1, min(8000, size - start))})
        return {"handle": handle, "episode_id": episode_id, "call_evidence_id": self.handles[handle],
                "ready": not missing and episode_id in self.focus_episode_ids,
                "missing_reads": missing, "scope": "Authorized full reads only; no correctness claim"}

    def _resolve(self, handles, selected_episode=None):
        refs, errors, normalized = [], [], list(dict.fromkeys(handles))
        for handle in normalized:
            if handle not in self.handles:
                errors.append({"field": "evidence_handles", "message": "Unknown/unreturned handle: " + handle})
                continue
            info = self._info(handle)
            if not info["ready"]:
                errors.append({"field": "evidence_handles", "message": "Evidence not ready", **info})
            if selected_episode and info["episode_id"] != selected_episode and not self.allow_same_task_contrast:
                errors.append({"field": "evidence_handles", "message": "Keep evidence within the selected episode"})
            refs.extend(self._required(handle))
        return list(dict.fromkeys(refs)), errors, normalized

    def _validate(self, name, args):
        definition = next(d for d in self.definitions if d["name"] == name)["input_schema"]
        if not isinstance(args, dict):
            return [{"field": "arguments", "message": "Expected an object of tool parameters"}]
        errors = [{"field": k, "message": "Required field missing"} for k in definition["required"] if k not in args]
        errors += [{"field": k, "message": "Unknown field; pass only advertised parameters"} for k in args if k not in definition["properties"]]
        for key, value in args.items():
            spec = definition["properties"].get(key)
            if spec is None:
                continue
            if spec["type"] == "string":
                if not isinstance(value, str) or not value.strip():
                    errors.append({"field": key, "message": "Expected nonempty text"})
                elif "enum" in spec and value not in spec["enum"]:
                    errors.append({"field": key, "message": "Choose " + "/".join(spec["enum"])})
            elif not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
                errors.append({"field": key, "message": "Expected a list of nonempty strings"})
            elif not spec.get("minItems", 0) <= len(value) <= spec.get("maxItems", 10000):
                errors.append({"field": key, "message": "Item count outside advertised bounds"})
        return errors

    def _publish(self, directory, document, handles, validation=None):
        directory.mkdir(parents=True, exist_ok=True)
        refs = document["diagnoses"][0]["evidence_ids"] if document.get("diagnoses") else (
            document["skills"][0]["evidence_ids"] if document.get("skills") else [])
        with (directory / "submission.json").open("x") as stream:
            json.dump({"schema": "structured_task_skill_v1", "document": document,
                "status": "candidate_pending_behavior_evaluation", "effectiveness_verified": False,
                **({"validation": validation} if validation is not None else {}),
                "evidence_audit": {"handles": {h: self._info(h) for h in handles},
                    "read_ranges": {eid: self.tracker._read_intervals.get(eid, []) for eid in refs},
                    "scope": "References/coverage, not semantic support"}}, stream, ensure_ascii=False, indent=2)

    def execute(self, name, arguments):
        if name not in self.names or self.complete:
            raise ValueError("Unknown tool or session already finished")
        errors = self._validate(name, arguments)
        expected = "diagnosis" if name == "save_diagnosis" else "skills"
        if name != "finish_without_skill" and self.stage != expected:
            errors.append({"field": "stage", "message": "Current stage: " + self.stage})
        refs, handles = [], []
        if not errors and name != "finish_without_skill":
            anchor = arguments["evidence_handle"] if name == "save_diagnosis" else self.diagnosis["evidence_handles"][0]
            inherited = [anchor] if name == "save_diagnosis" else self.diagnosis["evidence_handles"]
            selected = self._info(anchor)["episode_id"] if anchor in self.handles else None
            refs, errors, handles = self._resolve(inherited + arguments.get("evidence_handles", []), selected)
        if errors:
            result = {"success": False, "errors": errors, "stage": self.stage, "session_complete": False}
        elif name == "save_diagnosis":
            self.diagnosis = {"id": "D1", "episode_id": self._info(handles[0])["episode_id"],
                **{k: v for k, v in arguments.items() if k not in {"evidence_handle", "evidence_handles"}},
                "evidence_ids": refs, "evidence_handles": handles}
            self._publish(self.diagnosis_output, {"diagnoses": [self.diagnosis]}, handles)
            self.stage = "skills"
            result = {"success": True, "diagnosis_id": "D1", "session_complete": False,
                      "stage": "skills", "saved_diagnosis": self.diagnosis, "next_instructions": self.skill_prompt}
            if self.recorder:
                self.recorder.log("diagnosis_checkpoint", document={"diagnoses": [self.diagnosis]})
        else:
            if name == "finish_without_skill":
                if self.stage == "diagnosis":
                    self._publish(self.diagnosis_output, {"diagnoses": [], "reason": arguments["reason"]}, [])
                document = {"skills": [], "self_checks": [], "reason": arguments["reason"]}
            else:
                card = {k: arguments[k] for k in ("title", "when", "guidance", "limitations", "rationale")}
                document = {"skills": [{**card, "evidence_ids": refs}], "self_checks": [{
                    "skill_index": 0, "diagnosis_ids": [self.diagnosis["id"]], "explanation": arguments["self_check"]}]}
            validation = None
            if self.deferred_validation:
                validation = {"schema": "deferred_skill_validation_v1", "status": "not_run",
                    "published": False, "candidate_sha256": digest(canonical(document).encode()),
                    "claims": [] if name == "finish_without_skill" else [{"claim_id": "C1",
                        **{k: arguments[k] for k in ("claim", "claim_type", "validation_method")},
                        "evidence_ids": refs, "assessment": "unverified"}],
                    "scope": "Proposed validation only. No test execution, semantic certification or transfer claim."}
            self._publish(self.skill_output, document, handles, validation)
            self.complete = True
            result = {"success": True, "session_complete": True, "count": len(document["skills"]),
                      "warning": "Saved provisionally; utility untested"}
        if self.recorder:
            self.recorder.log("structured_submission", operation=name, arguments=arguments, **result)
        return canonical(result)
