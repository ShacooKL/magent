from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from magents.engine import Store, Team
from magents.submissions import validate
from unittest.mock import MagicMock
import json


class VerificationSubmissionTests(TestCase):
    def test_test_ids_normalize_without_mutating_raw_model_payload(self):
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'task', {})
            event_id = store.event('tool_result', call={'name': 'hypothesis_test'},
                                   output=json.dumps({'test_id': 'h1:t1', 'test_executed': True, 'output': 'NaN'}))
            item = dict(candidate='Argument formatting', verdict='supported', confidence='high',
                        source_refs=['source-1:87'], test_refs=['h1:t1'], observations='Different errors',
                        control_audit='Only type changed', prediction_audit='Needs review', scope='One call',
                        not_established='Generalization', remaining_objections=[])
            report = dict(status='completed', candidates=[item], reason='Limited audit')
            resolver = MagicMock()
            accepted = validate('submit_verification', report, store, resolver)
            self.assertEqual(accepted['candidates'][0]['test_refs'], [f'learning:{event_id}'])
            self.assertEqual(report['candidates'][0]['test_refs'], ['h1:t1'])
            resolver.assert_any_call(f'learning:{event_id}', store)
            class Client:
                def create_message(self, *args, **kwargs):
                    return {'content': [{'type': 'tool_use', 'id': 'report', 'name': 'submit_verification', 'input': report}]}
            team = Team(Client(), MagicMock(definitions=[]), store, dict(max_requests=1,
                        max_role_requests=1, max_tokens=500, tool_output_chars=1000))
            team.contexts['learner'], team.prompts['learner'] = [], 'Audit only'
            team.ask('learner', 'Submit', submission='submit_verification')
            self.assertEqual(store.state['verification_submission'], accepted)
            rows = [json.loads(line) for line in (store.root / 'events.jsonl').read_text().splitlines()]
            response = next(row for row in rows if row['kind'] == 'response')
            self.assertEqual(response['response']['content'][0]['input']['candidates'][0]['test_refs'], ['h1:t1'])
            for bad in ('h9:t9', 'h1', 'source-1:h1:t1'):
                item['test_refs'] = [bad]
                with self.assertRaises(ValueError):
                    validate('submit_verification', report, store, resolver)
            item['test_refs'] = ['h1:t1']
            store.event('tool_result', call={'name': 'hypothesis_test'},
                        output=json.dumps({'test_id': 'h1:t1'}))
            with self.assertRaisesRegex(ValueError, '2 matches'):
                validate('submit_verification', report, store, resolver)

    def test_candidate_contract_and_local_test_references(self):
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'task', {})
            test_id = store.event('tool_result', call={'name': 'hypothesis_test'}, output='NaN')
            prose_id = store.event('response', response={'content': 'Claimed success'})
            item = dict(candidate='Argument formatting', verdict='supported', confidence='high',
                        source_refs=['source-1:87'], test_refs=[f'learning:{test_id}'],
                        observations='Different error outputs', control_audit='Only type changed',
                        prediction_audit='Original predictions reversed', scope='One call',
                        not_established='Dataset-wide cause', remaining_objections=[])
            report = dict(status='completed', candidates=[item], reason='Audit finished; scope limited')
            resolver = MagicMock()
            validate('submit_verification', report, store, resolver)
            with self.assertRaises(ValueError):
                validate('submit_verification', dict(status='completed', answer='C', reason='closest'), store, resolver)
            item['test_refs'] = [f'learning:{prose_id}']
            with self.assertRaisesRegex(ValueError, 'hypothesis_test tool results'):
                validate('submit_verification', report, store, resolver)
            item['test_refs'] = []
            with self.assertRaisesRegex(ValueError, 'actual test references'):
                validate('submit_verification', report, store, resolver)
            item['verdict'] = 'untested'
            validate('submit_verification', report, store, resolver)

            class Client:
                def create_message(self, *args, **kwargs):
                    return {'content': [{'type': 'tool_use', 'id': 'report',
                                         'name': 'submit_verification', 'input': report}]}
            tools = MagicMock(definitions=[])
            team = Team(Client(), tools, store, dict(max_requests=1, max_role_requests=1,
                        max_tokens=500, tool_output_chars=1000))
            team.contexts['learner'], team.prompts['learner'] = [], 'Audit only'
            team.ask('learner', 'Submit report', submission='submit_verification')
            self.assertEqual(store.state['verification_submission'], report)
            self.assertNotIn('learning_submission', store.state)
