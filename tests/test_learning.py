import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from magents.learning import LearningTools, trace_view, learning_prompt
from magents.engine import Store, Team
from magents.submissions import validate
from experiments.learn_q10 import source_runs


class LearningTests(TestCase):
    def test_diagnosis_mode_disables_replay_and_requires_hypothesis_checks(self):
        prompt = learning_prompt(True, True)
        self.assertIn('hypothesis_conclude for the central causal claim', prompt)
        self.assertNotIn('Call replay_task with the candidate', prompt)
        self.assertIn('single-source mode preserves original paths', prompt)
        tools = LearningTools(MagicMock(definitions=[]), {}, Path('/tmp'), 93,
                              Path('/tmp/unused.py'), allow_replay=False)
        self.assertNotIn('replay_task', {t['name'] for t in tools.definitions})
        with self.assertRaisesRegex(ValueError, 'disabled'):
            tools.execute('replay_task', {'skill': 'not executed'})

    def test_qualified_references_resolve_only_in_their_source(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            (source / 'events.jsonl').write_text(json.dumps({'id': 100, 'kind': 'test'}) + '\n')
            store = Store(root / 'learning', 'test', {})
            store.event('test')
            tools = LearningTools(MagicMock(definitions=[]), {'source-1': source, 'round-1/with-skill': source}, root, 93, root / 'runner.py')
            value = {'status': 'completed', 'answer': 'Diagnosis', 'reason': '',
                     'evidence_refs': ['source-1:100', 'learning:1', 'round-1/with-skill:100']}
            validate('submit_learning', value, store, tools.resolve_reference)
            for ref, error in [('100', 'qualified'), ('source-1:1', "source 'source-1'"),
                               ('learning:100', 'learning run'), ('missing:100', 'Available sources')]:
                with self.assertRaisesRegex(ValueError, error):
                    tools.resolve_reference(ref, store)
            with self.assertRaises(ValueError):
                validate('submit_learning', {**value, 'evidence_refs': [100]}, store, tools.resolve_reference)
            # Existing solving contract still uses local integer IDs.
            validate('submit_final', {'status': 'completed', 'answer': 'A', 'reason': '', 'evidence_ids': [1]}, store)
            with self.assertRaises(ValueError):
                validate('submit_final', {'status': 'completed', 'answer': 'A', 'reason': '', 'evidence_ids': ['source-1:100']}, store)

    def test_read_then_submit_source_and_local_evidence_end_to_end(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            (source / 'events.jsonl').write_text(json.dumps({'id': 100, 'kind': 'tool_result',
                'call': {'name': 'mean', 'input': {'threshold': 0.7}}, 'output': 'NaN'}) + '\n')
            store = Store(root / 'learning', 'test', {})
            tools = LearningTools(MagicMock(definitions=[]), {'source-1': source}, root, 93, root / 'runner.py')
            class Client:
                count = 0
                def create_message(inner, system, messages, definitions, max_tokens):
                    inner.count += 1
                    if inner.count == 1:
                        name, value = 'trace_read', {'run': 'source-1', 'event_id': 100}
                    else:
                        returned = messages[-1]['content'][0]['content']
                        self.assertIn('Evidence ref: learning:4', returned)
                        self.assertIn('"evidence_ref": "source-1:100"', returned)
                        name, value = 'submit_learning', {'status': 'completed', 'answer': 'Read actual parameter types.',
                            'reason': 'Offline interface test, not a real diagnosis.', 'evidence_refs': ['source-1:100', 'learning:4']}
                    return {'content': [{'type': 'tool_use', 'id': 'call', 'name': name, 'input': value}]}
            team = Team(Client(), tools, store, {'max_requests': 3, 'max_role_requests': 3, 'max_tokens': 100, 'tool_output_chars': 16000})
            team.contexts['learner'] = []
            team.prompts['learner'] = 'test'
            team.ask('learner', 'Read then submit', submission='submit_learning')
            self.assertEqual(store.state['requests'], 2)
            self.assertEqual(store.state['learning_submission']['evidence_refs'], ['source-1:100', 'learning:4'])
            self.assertNotIn('final_submission', store.state)

    def test_source_task_identity(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'run.json').write_text(json.dumps({'task_id': '93'}))
            self.assertEqual(source_runs(93, [root]), {'source-1': root.resolve()})
            with self.assertRaisesRegex(ValueError, 'does not match'):
                source_runs(10, [root])
            with self.assertRaisesRegex(ValueError, 'source-run'):
                source_runs(93, None)

    def test_trace_does_not_return_gt_key(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'events.jsonl').write_text(json.dumps({'id': 1, 'kind': 'host_score',
                'correct': True, 'expected': ['D']}) + '\n')
            base = MagicMock(definitions=[])
            tools = LearningTools(base, {'source': root}, root, 10, root / 'runner.py')
            result = tools.execute('trace_read', {'run': 'source', 'event_id': 1})
            self.assertNotIn('expected', result)
            self.assertIn('correct', result)
            expanded = tools.execute('trace_expand', {'run': 'source', 'event_id': 1, 'section': 'event'})
            self.assertNotIn('expected', expanded)
            self.assertEqual(json.loads(expanded)['evidence_ref'], 'source:1')

    def test_argument_types_are_preserved_in_default_view(self):
        row = {'id': 100, 'kind': 'tool_result', 'role': 'reviewer',
               'call': {'name': 'example', 'input': {
                   'paths': '["one.tif", "two.tif"]', 'actual_paths': ['one.tif', 'two.tif'],
                   'threshold': 0.7, 'text_threshold': '0.7', 'verbose': True}},
               'output': 'NaN', 'dispatch_error': False}
        view = trace_view('source-1', row)
        self.assertEqual(view['arguments']['paths']['json_type'], 'string')
        self.assertEqual(view['arguments']['actual_paths']['json_type'], 'array')
        self.assertEqual(view['arguments']['threshold']['json_type'], 'number')
        self.assertEqual(view['arguments']['text_threshold']['json_type'], 'string')
        self.assertEqual(view['arguments']['verbose']['json_type'], 'boolean')
        self.assertEqual(view['output']['value'], 'NaN')
        self.assertFalse(view['dispatch_error'])

    def test_long_arguments_do_not_hide_output_and_expand_losslessly(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {'paths': json.dumps(['目录/' + 'x' * 100 + '.tif'] * 100, ensure_ascii=False), 'threshold': 0.7}
            row = {'id': 100, 'kind': 'tool_result', 'call': {'name': 'mean', 'input': payload}, 'output': 'NaN'}
            (root / 'events.jsonl').write_text(json.dumps(row) + '\n')
            tools = LearningTools(MagicMock(definitions=[]), {'source': root}, root, 93, root / 'runner.py')
            view = json.loads(tools.execute('trace_read', {'run': 'source', 'event_id': 100}))
            self.assertTrue(view['arguments']['paths']['truncated'])
            self.assertEqual(view['arguments']['paths']['json_type'], 'string')
            self.assertEqual(view['arguments']['threshold']['value'], 0.7)
            self.assertEqual(view['output']['value'], 'NaN')
            self.assertLess(len(json.dumps(view)), 3000)
            chunks, offset = [], 0
            while offset is not None:
                page = json.loads(tools.execute('trace_expand', {'run': 'source', 'event_id': 100,
                                                               'section': 'arguments', 'offset_chars': offset}))
                self.assertEqual(page['returned_chars'], len(page['text']))
                chunks.append(page['text'])
                offset = page['next_offset_chars']
            self.assertEqual(json.loads(''.join(chunks)), payload)
            with self.assertRaisesRegex(ValueError, 'characters'):
                tools.execute('trace_expand', {'run': 'source', 'event_id': 100,
                                              'section': 'arguments', 'limit_chars': 20})
            schema = next(t['input_schema'] for t in tools.definitions if t['name'] == 'trace_read')
            self.assertEqual(schema['required'], ['run', 'event_id'])

    def test_replay_is_frozen_paired_and_bounded(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            base = MagicMock(definitions=[])
            tools = LearningTools(base, {}, root, 10, root / 'runner.py')
            with patch('magents.learning.subprocess.run', return_value=MagicMock(returncode=0)) as run:
                for _ in range(2):
                    tools.execute('replay_task', {'skill': 'Inspect applicability before selecting a tool.'})
                self.assertEqual(run.call_count, 4)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertTrue(all('--official-only' in cmd for cmd in commands))
                self.assertTrue(all('--no-skill-retrieval' in cmd for cmd in commands))
                self.assertEqual(sum('--skill-file' in cmd for cmd in commands), 2)
                self.assertTrue(all('--data-dir' not in cmd for cmd in commands))
                self.assertEqual((root / 'round-1/candidate.txt').read_text(),
                                 'Inspect applicability before selecting a tool.')
                with self.assertRaisesRegex(ValueError, 'Two paired'):
                    tools.execute('replay_task', {'skill': 'Another skill'})
                self.assertEqual(run.call_count, 4)
