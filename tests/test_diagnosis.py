from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from magents.engine import Store, StageYield
from magents.diagnosis import diagnose_simple, PROMPT, SINGLE_PROMPT


class SimpleDiagnosisTests(TestCase):
    def test_single_and_review_share_total_budget_with_private_contexts(self):
        for review in (False, True):
            with self.subTest(review=review), TemporaryDirectory() as temp:
                agents, calls = [], []
                class FakeTeam:
                    def __init__(self, client, tools, store, config):
                        self.contexts, self.prompts = {}, {}
                        agents.append(self)
                    def ask(self, role, instruction, **kwargs):
                        calls.append((instruction, kwargs))
                        store.state['requests'] += kwargs['request_limit']
                        store.state['learning_submission'] = {'answer': 'diagnosis', 'status': 'completed'}
                store = Store(Path(temp) / 'run', 'task', {})
                with patch('magents.diagnosis.Team', FakeTeam):
                    diagnose_simple(None, None, store, {'max_requests': 40}, 'source', review=review)
                self.assertEqual(store.state['requests'], 40)
                self.assertEqual(len(agents), 2 if review else 1)
                self.assertEqual(agents[0].prompts['learner'], PROMPT if review else SINGLE_PROMPT)
                self.assertEqual([kw['request_limit'] for _, kw in calls], [27, 13] if review else [40])
                if review:
                    self.assertIsNot(agents[0].contexts['learner'], agents[1].contexts['learner'])
                    self.assertIn('Unverified analyst report', calls[1][0])
                    self.assertEqual(store.state['diagnosis_draft']['answer'], 'diagnosis')

    def test_single_protocol_has_ordered_checks_without_new_submission_contract(self):
        markers = ['First locate', '1. Parameter format', '2. Parameter meaning',
                   '3. Tool selection', '4. Call sequence', 'Verify only what is uncertain']
        positions = [SINGLE_PROMPT.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('Finish with submit_learning', SINGLE_PROMPT)
        self.assertIn('identical NaN outputs need not have identical causes', SINGLE_PROMPT)

    def test_failed_review_does_not_leave_draft_as_final(self):
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'task', {})
            class FakeTeam:
                def __init__(self, *args):
                    self.contexts, self.prompts = {}, {}
                def ask(self, *args, **kwargs):
                    store.state['requests'] += 1
                    if store.state['requests'] == 1:
                        store.state['learning_submission'] = {'answer': 'draft', 'status': 'completed'}
                    else:
                        raise StageYield('No final review')
            with patch('magents.diagnosis.Team', FakeTeam), self.assertRaises(StageYield):
                diagnose_simple(None, None, store, {'max_requests': 40}, 'source', review=True)
            self.assertNotIn('learning_submission', store.state)
            self.assertEqual(store.state['diagnosis_draft']['answer'], 'draft')
