"""Session-local, bounded delivery of agent-selected advisory task Skills."""
from copy import deepcopy

from rsagent.evolution.tool_registry import canonical


class TaskSkillDeliveryProvider:
    """Expose a small catalog first; reveal only explicitly selected card bodies."""

    name = "task_skill_read"
    _fields = ("id", "title", "when", "guidance", "limitations")

    def __init__(self, cards, recorder=None, max_cards=2):
        if type(max_cards) is not int or max_cards <= 0:
            raise ValueError("max_cards must be a positive integer")
        if not isinstance(cards, list):
            raise ValueError("cards must be a list of advisory cards")
        self._cards = {}
        for card in cards:
            if not isinstance(card, dict) or not set(self._fields).issubset(card):
                raise ValueError("Each advisory card requires id, title, when, guidance, limitations")
            for field in ("id", "title", "when"):
                if not isinstance(card[field], str) or not card[field].strip():
                    raise ValueError(f"Card {field} must be a nonempty string")
            for field in ("guidance", "limitations"):
                values = card[field]
                if not isinstance(values, list) or not values or not all(
                        isinstance(value, str) and value.strip() for value in values):
                    raise ValueError(f"Card {field} must be a nonempty list of nonempty strings")
            if card["id"] in self._cards:
                raise ValueError("Card IDs must be unique")
            # Whitelist fields even when a caller accidentally supplies provenance.
            self._cards[card["id"]] = deepcopy({key: card[key] for key in self._fields})
        self.max_cards, self.recorder = max_cards, recorder
        self._selected_ids = []
        self.names = {self.name}
        self.definitions = [{
            "name": self.name,
            "description": (
                "Read advisory remote-sensing task Skills selected from the catalog. "
                f"At most {max_cards} distinct cards may be opened during this session. "
                "You may choose none and may reread already selected cards. "
                "Skills are optional decision references, not mandatory instructions or ground truth."
            ),
            "input_schema": {"type": "object", "properties": {
                "ids": {"type": "array", "items": {"type": "string"},
                        "minItems": 0, "maxItems": max_cards, "uniqueItems": True}},
                "required": ["ids"], "additionalProperties": False},
        }]

    @property
    def selected_ids(self):
        return list(self._selected_ids)

    def catalog(self):
        """Return fresh metadata only, never card bodies or historical provenance."""
        return [{key: card[key] for key in ("id", "title", "when")}
                for card in self._cards.values()]

    def execute(self, name, arguments):
        if name != self.name or not isinstance(arguments, dict) or set(arguments) != {"ids"}:
            raise ValueError("Expected task_skill_read with only ids")
        ids = arguments["ids"]
        if not isinstance(ids, list) or len(ids) > self.max_cards or not all(
                isinstance(value, str) and value.strip() for value in ids):
            raise ValueError(f"ids must contain 0..{self.max_cards} nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must not contain duplicates within one request")
        if any(value not in self._cards for value in ids):
            raise ValueError("Every requested ID must occur in the Skill catalog")
        new_ids = [value for value in ids if value not in self._selected_ids]
        if len(self._selected_ids) + len(new_ids) > self.max_cards:
            raise ValueError("Session Skill selection limit exceeded; no cards were opened")
        # Validate the entire request before changing selection state or revealing any card.
        self._selected_ids.extend(new_ids)
        result = {"success": True, "skills": [deepcopy(self._cards[value]) for value in ids],
                  "selected_ids": self.selected_ids,
                  "remaining_selections": self.max_cards - len(self._selected_ids)}
        if self.recorder:
            self.recorder.log("task_skill_read", requested_ids=list(ids),
                              selected_ids=self.selected_ids,
                              remaining_selections=result["remaining_selections"])
        return canonical(result)
