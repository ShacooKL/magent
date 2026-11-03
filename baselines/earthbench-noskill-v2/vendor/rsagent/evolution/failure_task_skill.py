"""Small task-Skill candidates anchored in observed failures, not host diagnoses."""
from rsagent.evolution.grounded_task_skill import GroundedTaskSkillExtractionProvider


class FailureTaskSkillExtractionProvider(GroundedTaskSkillExtractionProvider):
    def __init__(self, workspace, output, trajectory, failure_episode_ids, recorder=None):
        super().__init__(workspace, output, trajectory, recorder)
        self.failure_episode_ids = set(failure_episode_ids)
        if not self.failure_episode_ids or any(
            eid not in self._episodes or self._episodes[eid].get("ending", {}).get("correct") is not False
            for eid in self.failure_episode_ids
        ):
            raise ValueError("Require authorized episodes with explicit incorrect outcomes")
        for definition in self.definitions:
            definition["description"] = definition["description"].replace("0..8", "0..3")
            definition["description"] += (
                " At most three cards. Each must include a fully read task/call/result triple "
                "from a focus failure episode in the supplied index. This anchors investigation, "
                "not proof that the proposed decision fixes the failure. Empty skills is valid.")

    def inspection_progress(self):
        """Mechanical read coverage, independent of drafted advice or citations."""
        inspected = {episode_id for episode_id, ids in self._triples
                     if episode_id in self.failure_episode_ids and all(self._fully_read(eid) for eid in ids)}
        return {"fully_inspected_focus_episode_ids": sorted(inspected),
                "remaining_focus_episode_ids": sorted(self.failure_episode_ids - inspected),
                "scope": "Full task/call/result returned; not understanding, relevance, or utility."}

    def _check(self, document):
        result = super()._check(document)
        if not isinstance(document, dict) or not isinstance(document.get("skills"), list):
            return result
        if len(document["skills"]) > 3:
            result["errors"].append("Keep at most 3 task Skills; select the most useful candidates yourself.")
        for card in result["audit"]["cards"]:
            matched = sorted(set(card["qualifying_episode_ids"]) & self.failure_episode_ids)
            card["focus_failure_episode_ids"] = matched
            if not matched:
                card["success"] = False
                cited = set(document["skills"][card["index"]]["evidence_ids"])
                card["repair_alternatives"] = self._repair_hints(
                    cited, card["qualifying_episode_ids"], [False], self.failure_episode_ids)
                card["repair_note"] = (
                    "Complete one fully read task/call/result triple from a focus failure. "
                    "Other historical episodes, including historical failures, cannot substitute; other pairs are optional. "
                    "You may remove this card or submit an empty bundle.")
                result["errors"].append(
                    f"skills[{card['index']}]: fully read and cite a task/call/result triple from a "
                    "designated focus failure. Other historical episodes, even failures, do not meet this source requirement. "
                    "Choose the failure yourself, or omit this card.")
        result["success"] = not result["errors"]
        return result
