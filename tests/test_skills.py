import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from magents.skills import SkillLibrary
from magents.adapters import WorkspaceTools
from magents.engine import Store, Team

LIBRARY = Path(__file__).resolve().parents[1] / 'library/skills'


class SkillTests(TestCase):
    def test_search_read_and_no_match(self):
        library = SkillLibrary(LIBRARY)
        for query in ('NDVI mean LST', '植被平均温度', 'calculate_mean_lst_by_ndvi'):
            hits = json.loads(library.execute('skill_search', {'query': query}))
            self.assertEqual(hits[0]['id'], 'ndvi-conditioned-mean-lst')
            self.assertNotIn('content', hits[0])
        self.assertEqual(json.loads(library.execute('skill_search', {'query': 'precipitation rainfall'})), [])
        card = json.loads(library.execute('skill_read', {'skill_id': 'ndvi-conditioned-mean-lst'}))
        self.assertEqual(card['status'], 'provisional')
        self.assertIn('calculate_mean_lst_by_ndvi', card['content'])
        self.assertNotIn('provenance', card)
        with self.assertRaises(KeyError):
            library.execute('skill_read', {'skill_id': '../private'})

    def test_official_only_can_retrieve_but_not_execute_code(self):
        with patch('magents.adapters.NamespaceRunner'), patch('magents.adapters.ToolBox') as box:
            box.return_value.definitions = []
            tools = WorkspaceTools(None, [], {}, official_only=True, skill_dir=LIBRARY)
            self.assertIn('provisional', tools.execute('skill_search', {'query': 'NDVI'}))
            with self.assertRaises(ValueError):
                tools.execute('run_command', {'command': 'true'})

    def test_coordinator_planning_retrieves_in_continuous_context(self):
        library = SkillLibrary(LIBRARY)
        class Client:
            count = 0
            def create_message(inner, system, messages, tools, max_tokens):
                inner.count += 1
                self.assertEqual({t['name'] for t in tools}, {'skill_search', 'skill_read', 'submit_plan'})
                if inner.count == 1:
                    inner.prefix = json.loads(json.dumps(messages))
                    return {'content': [{'type': 'tool_use', 'id': 's', 'name': 'skill_search', 'input': {'query': 'mean LST'}}]}
                self.assertEqual(messages[:len(inner.prefix)], inner.prefix)
                if inner.count == 2:
                    return {'content': [{'type': 'tool_use', 'id': 'r', 'name': 'skill_read', 'input': {'skill_id': 'ndvi-conditioned-mean-lst'}}]}
                return {'content': [{'type': 'tool_use', 'id': 'p', 'name': 'submit_plan',
                                     'input': {'plan': 'Adopt conditional mean guidance; verify task applicability.'}}]}
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'test', {})
            config = {'max_requests': 10, 'max_role_requests': 4, 'max_tokens': 100,
                      'tool_output_chars': 16000, 'skill_retrieval': True}
            team = Team(Client(), library, store, config)
            self.assertIn('Adopt', team.plan())
            self.assertEqual(store.state['requests'], 3)
            rows = [json.loads(line) for line in (store.root / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([r['call']['name'] for r in rows if r['kind'] == 'tool_result'], ['skill_search', 'skill_read', 'submit_plan'])

    def test_repeated_planner_errors_fall_back_and_execution_finishes(self):
        library = SkillLibrary(LIBRARY)
        class Client:
            planning_calls = 0
            def create_message(inner, system, messages, tools, max_tokens):
                names = {t['name'] for t in tools}
                if 'submit_plan' in names:
                    inner.planning_calls += 1
                    if inner.planning_calls == 4:
                        self.assertEqual(names, {'submit_plan'})
                    if inner.planning_calls == 2:
                        name, value = 'skill_search', {'query': 'NDVI mean'}
                    else:
                        # Reproduce illegal calls even after schema withdrawal.
                        name, value = 'list_files', {'path': '/workspace'}
                elif 'submit_progress' in names:
                    name, value = 'submit_progress', {'status': 'complete', 'completed': [],
                        'pending': [], 'issues': [], 'artifacts': [], 'evidence_ids': [], 'next_step': 'final'}
                else:
                    name, value = 'submit_final', {'status': 'incomplete', 'answer': '',
                        'reason': 'Offline execution reached final handoff', 'evidence_ids': []}
                return {'content': [{'type': 'tool_use', 'id': 'call', 'name': name, 'input': value}]}
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'test', {})
            config = {'max_requests': 10, 'max_role_requests': 2, 'max_plan_requests': 4,
                      'max_tokens': 100, 'tool_output_chars': 16000, 'skill_retrieval': True,
                      'review_rounds': 0}
            state = Team(Client(), library, store, config).run()
            self.assertEqual(state['plan_status'], 'fallback')
            self.assertEqual(state['requests'], 6)
            self.assertIn('Evidence ID:', state['plan'])
            self.assertEqual(state['final_submission']['reason'], 'Offline execution reached final handoff')

    def test_final_planning_slot_closes_retrieval(self):
        library = SkillLibrary(LIBRARY)
        class Client:
            count = 0
            def create_message(inner, system, messages, tools, max_tokens):
                inner.count += 1
                if inner.count < 4:
                    name, value = 'skill_search', {'query': 'NDVI'}
                else:
                    self.assertEqual([t['name'] for t in tools], ['submit_plan'])
                    name, value = 'submit_plan', {'plan': 'Delegate official computation; inspect applicability.'}
                return {'content': [{'type': 'tool_use', 'id': 'call', 'name': name, 'input': value}]}
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'test', {})
            config = {'max_requests': 40, 'max_role_requests': 12, 'max_tokens': 100,
                      'tool_output_chars': 16000, 'skill_retrieval': True}
            team = Team(Client(), library, store, config)
            self.assertIn('Delegate', team.plan())
            self.assertEqual(store.state['plan_status'], 'submitted')
            self.assertEqual(store.state['requests'], 4)
