"""Small read-only skill catalog; deterministic lexical search, no extra model."""
import json
import re
from pathlib import Path
from magents.submissions import tool, TEXT


def tokens(text):
    return set(re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]', text.lower()))


class SkillLibrary:
    def __init__(self, directory):
        self.cards = {}
        for path in sorted(Path(directory).glob('*.json')):
            card = json.loads(path.read_text())
            if card['id'] in self.cards:
                raise ValueError('Duplicate skill ID: ' + card['id'])
            self.cards[card['id']] = card
        self.definitions = [
            tool('skill_search', 'Search advisory skills by task, statistic or tool keywords. Returns summaries, not instructions. Empty query lists the catalog.',
                 {'query': TEXT}),
            tool('skill_read', 'Read a matching skill and its validation limits before deciding whether to apply it.',
                 {'skill_id': TEXT})]

    def execute(self, name, arguments):
        if name == 'skill_read':
            card = self.cards[arguments['skill_id']]
            return json.dumps({k: v for k, v in card.items() if k != 'provenance'}, ensure_ascii=False)
        query = tokens(arguments['query'])
        matches = []
        for card in self.cards.values():
            searchable = ' '.join([card['name'], card['when'], *card['keywords'], *card['tools']])
            score = len(query & tokens(searchable))
            if score or not query:
                matches.append((score, card))
        matches.sort(key=lambda item: (-item[0], item[1]['id']))
        return json.dumps([dict({k: card[k] for k in ('id', 'name', 'when', 'tools', 'status')}, score=score)
                           for score, card in matches[:5]], ensure_ascii=False)
