"""Process-grounded task Skills, with downstream trials as the utility test.

Full evidence retrieval and episode ownership are mechanical requirements, not
claims that the resulting advice is true, causal, or useful.
"""
from rsagent.evolution.task_skill_extraction import (
    MAX_BYTES, TaskSkillExtractionProvider, check_task_skills,
)
from rsagent.evolution.tool_registry import canonical, safe_bytes, strict_json
from rsagent.runtime.jsonio import dump


class GroundedTaskSkillExtractionProvider(TaskSkillExtractionProvider):
    submission_schema = "remote_sensing_task_skills_v1"
    def __init__(self, workspace, output, trajectory, recorder=None, contrast=False):
        super().__init__(workspace, output, trajectory["records"], recorder)
        if type(contrast) is not bool:
            raise ValueError("contrast must be a boolean")
        self.trajectory = trajectory
        self.contrast = contrast
        self._read_intervals = {}
        self._read_empty = set()
        self._episodes = {episode["episode_id"]: episode for episode in trajectory["episodes"]}
        if len(self._episodes) != len(trajectory["episodes"]):
            raise ValueError("Episode IDs must be unique")
        self._triples = []
        for node in trajectory["nodes"]:
            episode_id = node["episode_id"]
            episode = self._episodes.get(episode_id)
            if episode is None:
                raise ValueError("Node refers to unknown episode")
            task_id = episode.get("task_evidence_id")
            for action in node["actions"]:
                ids = (task_id, action.get("call_evidence_id"), action.get("result_evidence_id"))
                # A missing tool result cannot constitute an inspected pair.
                if all(isinstance(eid, str) and eid in self.evidence for eid in ids):
                    self._triples.append((episode_id, ids))
        requirement = (
            "Each card must cite fully read L3 task, paired call and result from two different episodes, "
            "one ending.correct=true and one ending.correct=false. Unknown is not failure."
            if contrast else
            "Each card must cite a fully read L3 task and a paired call/result from that same episode."
        )
        for definition in self.definitions:
            definition["description"] += " " + requirement + " Process checks only, not semantic correctness."

    def note_evidence_read(self, evidence_id, offset=0, length=4000):
        """Record a successful trajectory_evidence retrieval, not a preview.

        The caller must invoke this only after the trajectory provider succeeds.
        Ranges use Python character offsets, matching TrajectoryView.raw.
        """
        if not isinstance(evidence_id, str) or evidence_id not in self.evidence:
            raise ValueError("Unknown or unauthorized evidence")
        if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 8000:
            raise ValueError("Use offset >= 0 and length 1..8000 characters")
        size = len(self.evidence[evidence_id]["text"])
        if size == 0 and offset == 0:
            self._read_empty.add(evidence_id)
        end = min(offset + length, size)
        if offset >= end:
            return
        intervals = sorted([*self._read_intervals.get(evidence_id, []), (offset, end)])
        merged = []
        for start, stop in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            else:
                merged.append((start, stop))
        self._read_intervals[evidence_id] = merged

    def _fully_read(self, evidence_id):
        size = len(self.evidence[evidence_id]["text"])
        if size == 0:
            return evidence_id in self._read_empty
        return self._read_intervals.get(evidence_id) == [(0, size)]

    def _repair_hints(self, cited, qualifying, outcomes=None, episode_ids=None):
        """Suggest one cheapest pair per already chosen, still-needed episode.

        These are alternatives, not extra mandatory pairs. Never select examples
        from an unreferenced episode or require more pairs from a satisfied side.
        """
        related = {eid for eid, ids in self._triples if cited.intersection(ids)}
        related.update(eid for eid, ep in self._episodes.items()
                       if ep.get("ending_evidence_id") in cited)
        if episode_ids is not None:
            related.intersection_update(episode_ids)
        hints = []
        for episode_id in sorted(related - set(qualifying)):
            outcome = self._episodes[episode_id].get("ending", {}).get("correct")
            if outcomes is not None and not any(outcome is needed for needed in outcomes):
                continue
            options = [ids for eid, ids in self._triples if eid == episode_id]
            if not options:
                continue
            ids = min(options, key=lambda ids: (
                sum(eid not in cited for eid in ids) + sum(not self._fully_read(eid) for eid in ids), ids))
            reads = []
            for eid in ids:
                size = len(self.evidence[eid]["text"])
                start = 0
                for left, right in [*self._read_intervals.get(eid, []), (size, size)]:
                    while start < left:
                        length = min(8000, left - start)
                        reads.append({"evidence_id": eid, "offset": start, "length": length})
                        start += length
                    start = max(start, right)
                if size == 0 and not self._fully_read(eid):
                    reads.append({"evidence_id": eid, "offset": 0, "length": 1})
            hints.append({"episode_id": episode_id, "observed_correct": outcome,
                          "alternative_only": True, "paired_evidence_ids": list(ids),
                          "missing_citations": [eid for eid in ids if eid not in cited],
                          "remaining_reads": reads})
        return hints[:4]

    def _check(self, document):
        result = check_task_skills(document, self.evidence)
        audit = {"process_only": True, "contrast_required": self.contrast,
                 "fully_read_evidence_ids": sorted(eid for eid in self.evidence if self._fully_read(eid)),
                 "cards": []}
        result["audit"] = audit
        result["scope"] = (
            "Format, authorized references, complete L3 reads and episode ownership only; "
            "no semantic or causal judgment. Utility is measured by downstream trials."
        )
        if not result["success"]:
            return result
        for index, card in enumerate(document["skills"]):
            cited = set(card["evidence_ids"])
            qualifying = sorted({episode_id for episode_id, ids in self._triples
                                 if set(ids) <= cited and all(self._fully_read(eid) for eid in ids)})
            positive = [eid for eid in qualifying if self._episodes[eid].get("ending", {}).get("correct") is True]
            negative = [eid for eid in qualifying if self._episodes[eid].get("ending", {}).get("correct") is False]
            valid = bool(positive and negative) if self.contrast else bool(qualifying)
            card_audit = {"index": index, "qualifying_episode_ids": qualifying,
                          "success_episode_ids": positive, "failure_episode_ids": negative,
                          "success": valid}
            audit["cards"].append(card_audit)
            if valid:
                continue
            requirement = ("two different episodes (one ending.correct=true, one false)" if self.contrast
                           else "at least one episode")
            result["errors"].append(
                f"skills[{index}]: cite and fully read via trajectory_evidence or trajectory_action_evidence the task_evidence_id "
                f"and a paired action call_evidence_id/result_evidence_id from {requirement}. "
                "Overview/expand previews do not count as L3 reads."
            )
            unread = sorted(eid for eid in cited if not self._fully_read(eid))
            if unread:
                result["errors"].append(f"skills[{index}]: cited evidence not fully read: " + ", ".join(unread))
            needed = ([True] if not positive else []) + ([False] if not negative else [])
            card_audit["missing_outcomes"] = needed if self.contrast else []
            card_audit["repair_alternatives"] = self._repair_hints(
                cited, qualifying, needed if self.contrast else None)
            card_audit["repair_note"] = (
                "Complete one pair per missing side, not every listed alternative. "
                "Choose other episodes yourself if needed. You may remove this card; empty skills is valid.")
        result["success"] = not result["errors"]
        return result

    def execute(self, name, arguments):
        # Keep the parent's source confinement and one-shot publication contract,
        # while checking grounding BEFORE any submission is written.
        if (name not in self.names or not isinstance(arguments, dict) or set(arguments) != {"source"}
                or not isinstance(arguments["source"], str)):
            raise ValueError("Expected task_skill_check/submit with source file")
        path = (self.workspace / arguments["source"]).resolve(strict=True)
        if not path.is_relative_to(self.workspace):
            raise ValueError("Source must remain inside workspace")
        document = strict_json(safe_bytes(path, MAX_BYTES).decode())
        result = self._check(document)
        if name == "task_skill_submit" and result["success"]:
            if (self.output / "submission.json").exists():
                raise ValueError("Already submitted; refusing overwrite")
            dump(self.output / "submission.json", {
                "schema": self.submission_schema, "document": document, "check": result,
                "status": "candidate_pending_behavior_evaluation", "effectiveness_verified": False,
            })
        if self.recorder:
            self.recorder.log(name, document=document, **result)
        return canonical(result)
