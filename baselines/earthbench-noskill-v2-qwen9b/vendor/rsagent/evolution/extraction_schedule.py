"""Enforce retrieval/repair/publication budgets without changing the tool catalog."""
import json
from pathlib import Path


class ExtractionSchedule:
    base_tools = {"read_file", "list_files", "write_file", "edit_file", "task_skill_check",
                  "task_skill_submit", "set_next_output_limit"}
    retrieval_tools = {"trajectory_overview", "trajectory_expand", "trajectory_search",
                       "trajectory_evidence", "trajectory_artifact", "trajectory_action_evidence", "trajectory_decisions"}

    def __init__(self, workspace, trajectory, explore_calls=18, repair_calls=6, finalize_calls=6, recorder=None,
                 focus_episode_ids=None, allow_sandbox_exploration=False, structured_submission=False):
        counts = (explore_calls, repair_calls, finalize_calls)
        if any(type(n) is not int or n < 1 for n in counts):
            raise ValueError("Each extraction phase needs a positive integer budget")
        self.workspace = Path(workspace).resolve()
        if type(allow_sandbox_exploration) is not bool:
            raise ValueError("allow_sandbox_exploration must be boolean")
        self.allow_sandbox_exploration = allow_sandbox_exploration
        self.base_tools = set(type(self).base_tools)
        if type(structured_submission) is not bool:
            raise ValueError("structured_submission must be boolean")
        self.structured_submission = structured_submission
        if structured_submission:
            self.base_tools -= {"task_skill_check", "task_skill_submit"}
            self.base_tools |= {"save_diagnosis", "save_skill", "finish_without_skill"}
        if allow_sandbox_exploration:
            self.base_tools.add("run_command")
        self.explore_calls, self.repair_calls, self.finalize_calls = counts
        self.total = sum(counts)
        self.recorder, self.step, self.submitted = recorder, 0, False
        self.opened = set()
        self.focus_progress = None
        episodes = {e["episode_id"]: e for e in trajectory["episodes"]}
        if focus_episode_ids is not None and (
                not isinstance(focus_episode_ids, list) or not focus_episode_ids or
                any(not isinstance(eid, str) or eid not in episodes or
                    episodes[eid].get("ending", {}).get("correct") is not False for eid in focus_episode_ids) or
                len(set(focus_episode_ids)) != len(focus_episode_ids)):
            raise ValueError("Focus entrypoints must be distinct authorized explicitly failed episodes")
        self.focus_episode_ids = set(focus_episode_ids or [])
        self.record_episodes, self.node_episodes, self.call_episodes = {}, {}, {}
        def add(record_id, episode_id):
            if record_id:
                self.record_episodes.setdefault(record_id, set()).add(episode_id)
        for episode in trajectory["episodes"]:
            eid = episode["episode_id"]
            for key in ("task_evidence_id", "start_evidence_id", "ending_evidence_id"):
                add(episode.get(key), eid)
        for note in trajectory.get("decision_trace", {}).get("notes", []):
            add(note['evidence_id'], note['episode_id'])
        for node in trajectory["nodes"]:
            eid = node["episode_id"]
            self.node_episodes[node["node_id"]] = eid
            for action in node["actions"]:
                self.call_episodes[action["call_evidence_id"]] = eid
                add(action["call_evidence_id"], eid)
                add(action.get("result_evidence_id"), eid)
                if trajectory.get("presentation") == "execution_view_v2":
                    for record in [action.get("raw_call_evidence_id"), "tool_definition:" + action["name"],
                                   *action.get("nested_event_evidence_ids", []), *action.get("evidence_requirements", [])]:
                        add(record, eid)

    @property
    def phase(self):
        if self.step <= self.explore_calls:
            return "explore"
        if self.step <= self.explore_calls + self.repair_calls:
            return "repair"
        return "finalize"

    def notice(self, step, maximum):
        if maximum != self.total or type(step) is not int or not 1 <= step <= maximum:
            raise ValueError("Extraction schedule must match the session request budget")
        previous = self.phase
        self.step = step
        if self.recorder and (step == 1 or previous != self.phase):
            self.recorder.log("extraction_phase", phase=self.phase, step=step,
                              total=maximum, opened_episode_ids=sorted(self.opened))
        rules = {
            "explore": "Choose experiences yourself. Prefer trajectory_action_evidence to read task/call/result together. Write a small draft before exploration ends.",
            "repair": "No new overview/search/expand. Only evidence or action bundles from episodes you already opened are allowed. Repair or remove unsupported cards; empty skills is valid.",
            "finalize": "All trajectory retrieval is closed. Use already-read evidence to write/edit draft.json, check, and task_skill_submit. Remove unsupported cards or submit empty skills if needed.",
        }
        progress_notice = ""
        if self.focus_episode_ids:
            if self.focus_progress is None:
                raise ValueError("Focus feedback requires a trusted inspection progress callback")
            progress = self.focus_progress()
            inspected = progress["fully_inspected_focus_episode_ids"]
            progress_notice = (
                f" Designated focus failures={sorted(self.focus_episode_ids)}; "
                f"fully read task/call/result from focus={inspected}. "
                "This reports reading, not whether any advice is correct or useful. ")
            if not inspected and self.phase != "finalize":
                progress_notice += (
                    "No designated focus failure has been fully inspected yet. Choose one from the "
                    "entrypoint block in the initial prompt, expand its node, then read an actual CALL ID "
                    "with trajectory_action_evidence. Historical failures outside this list do not substitute. ")
            elif not inspected:
                progress_notice += "Retrieval is now closed; omit unsupported cards or submit empty skills. "
            if self.phase == "repair":
                rules["repair"] = (
                    "Repair evidence from already opened episodes OR designated focus failures. "
                    "You may expand a designated focus node even if not previously opened, then read its "
                    "paired evidence. No new history search/overview or non-focus node expansion. "
                    "Finish evidence repair now; finalization still closes all retrieval.")
            if self.recorder:
                self.recorder.log("extraction_focus_progress", step=step, phase=self.phase, **progress)
        notice = (f"Host extraction budget: model API request {step}/{maximum}, phase={self.phase}, "
                f"future model API requests={maximum-step}. Explore 1..{self.explore_calls}; "
                f"repair through {self.explore_calls+self.repair_calls}; remaining {self.finalize_calls} "
                f"model API requests reserved for publication. {rules[self.phase]} {progress_notice} "
                "A successful final task_skill_submit ends the session, even on the last model API request; "
                "an explicit session_complete=false checkpoint continues in the same context. "
                "A plain-text response or a file write is NOT a submission; never claim utility from submission alone.")
        if self.structured_submission:
            notice = notice.replace("Write a small draft", "Save a structured diagnosis")
            notice = notice.replace("write/edit draft.json, check, and task_skill_submit", "call save_diagnosis then save_skill")
            notice = notice.replace("final task_skill_submit", "save_skill or finish_without_skill")
            notice = notice.replace("submit empty skills", "call finish_without_skill")
        return notice

    def validate(self, name, arguments):
        if self.submitted:
            raise ValueError("Candidate already submitted; no further tools")
        if name not in self.base_tools | self.retrieval_tools:
            raise ValueError("Tool disabled in scheduled extraction")
        if name in {"write_file", "edit_file"} and not self.allow_sandbox_exploration:
            path = Path(arguments.get("path", ""))
            if path.is_relative_to("/workspace"):
                path = self.workspace / path.relative_to("/workspace")
            elif not path.is_absolute():
                path = self.workspace / path
            if path.resolve() != self.workspace / "draft.json":
                raise ValueError("Scheduled extraction may write only draft.json")
        if name not in self.retrieval_tools or self.phase == "explore":
            return
        if self.phase == "finalize":
            raise ValueError("Publication reserve: trajectory retrieval is closed; check/repair/submit from read evidence")
        if name == "trajectory_evidence":
            episodes = self.record_episodes.get(arguments.get("evidence_id"), set())
        elif name == "trajectory_action_evidence":
            episodes = {self.call_episodes.get(arguments.get("call_evidence_id"))}
        elif name == "trajectory_expand" and self.node_episodes.get(arguments.get("node_id")) in self.focus_episode_ids:
            return
        else:
            raise ValueError("Repair reserve: no new overview/search/expand; use evidence from already opened episodes")
        if not episodes.intersection(self.opened | self.focus_episode_ids):
            raise ValueError("Repair reserve only permits already opened episodes or designated focus failures")

    def observe(self, name, arguments, output, success):
        if not success:
            return
        if name == "task_skill_submit" or (self.structured_submission and name in {"save_diagnosis", "save_skill", "finish_without_skill"}):
            result = json.loads(output)
            if result.get("success") is True and result.get("session_complete") is not False:
                self.submitted = True
            return
        if self.phase == "finalize":
            return
        if name == "trajectory_expand":
            eid = self.node_episodes.get(arguments.get("node_id"))
            if eid:
                self.opened.add(eid)
        elif name == "trajectory_evidence":
            self.opened.update(self.record_episodes.get(arguments.get("evidence_id"), set()))
        elif name == "trajectory_action_evidence":
            eid = self.call_episodes.get(arguments.get("call_evidence_id"))
            if eid:
                self.opened.add(eid)
