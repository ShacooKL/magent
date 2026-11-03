import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from magents.engine import Store, StageYield
from magents.debate import diagnose_debate, extract_after_debate, executed_tests
from magents.hypotheses import Hypotheses


class DebateTests(TestCase):
    def test_supervised_verification_preserves_roles_and_40_request_budget(self):
        from unittest.mock import MagicMock
        tools = MagicMock()
        tools.base.hypotheses = MagicMock(spec=Hypotheses)
        tools.base.hypotheses.snapshot.return_value = {}
        calls, teams = [], []
        class FakeTeam:
            def __init__(self, client, tools, store, config):
                self.contexts, self.prompts, self.store = {}, {}, store
                self.index = len(teams)
                teams.append(self)
            def ask(self, role, instruction, **kwargs):
                calls.append((self.index, instruction, kwargs))
                self.store.state['requests'] += kwargs['request_limit']
                if self.store.state['debate_phase']['phase'] == 'verify' and not kwargs.get('text_only'):
                    tools.base.hypotheses.snapshot.return_value = {'h1': {'tests': [
                        {'test_id': 'h1:t1', 'test_executed': True}]}}
                if kwargs.get('submission'):
                    self.store.state['verification_submission'] = {'candidates': [{'candidate': 'Audit with uncertainty'}], 'status': 'incomplete'}
                return 'Unverified report'
        with TemporaryDirectory() as temp, patch('magents.debate.Team', FakeTeam):
            store = Store(Path(temp) / 'run', 'task', {})
            result = diagnose_debate(None, tools, store,
                {'max_requests': 40, 'supervised_verification': True}, 'prompt', 'source and candidates')
            self.assertIs(result, teams[1])
            self.assertEqual(store.state['requests'], 40)
            self.assertEqual([kw['request_limit'] for _, _, kw in calls], [5, 1, 3, 1, 11, 1, 5, 1, 5, 1, 6])
            self.assertEqual([i for i, _, _ in calls], [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
            for i, _, kw in calls:
                if i == 1:
                    self.assertIsNotNone(kw['allowed_tools'])
                    self.assertNotIn('run_command', kw['allowed_tools'])
                    self.assertNotIn('hypothesis_test', kw['allowed_tools'])
            self.assertIsNone(calls[4][2]['allowed_tools'])
            self.assertIn('evidence_read', calls[-1][2]['allowed_tools'])
            self.assertIn('confidence', calls[0][1])
            self.assertEqual(store.state['debate_phase'], {'analyst': 'B', 'phase': 'synthesis'})
            self.assertEqual(store.state['verification_gate']['status'], 'executed_pending_audit')
            self.assertEqual(calls[-1][2]['submission'], 'submit_verification')

    def test_execution_gate_rejects_missing_inputs_unknown_execution_and_old_tests(self):
        record = {'h1': {'tests': [
            {'test_id': 'missing', 'test_executed': False},
            {'test_id': 'unknown', 'test_executed': None},
            {'test_id': 'ran_nan', 'test_executed': True, 'output': 'NaN'}]}}
        self.assertEqual(executed_tests(record), {('h1', 'ran_nan')})
        self.assertFalse(executed_tests(record) - executed_tests(record))

    def test_prose_only_verifier_retries_within_budget_then_stops_before_audit(self):
        from unittest.mock import MagicMock
        calls = []
        class Client:
            def create_message(self, system, messages, definitions, max_tokens):
                calls.append((messages[-1]['content'], {t['name'] for t in definitions}))
                return {'content': [{'type': 'text', 'text': 'I think the data is sparse; no test performed.'}]}
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'task', {})
            tools = MagicMock(definitions=[{'name': 'hypothesis_test'}, {'name': 'run_command'}, {'name': 'trace_read'}])
            with self.assertRaisesRegex(StageYield, 'no target execution'):
                diagnose_debate(Client(), tools, store,
                    {'max_requests': 40, 'max_role_requests': 40, 'max_tokens': 100,
                     'tool_output_chars': 1000, 'supervised_verification': True}, 'prompt', 'source')
            self.assertEqual(store.state['requests'], 22)
            self.assertEqual(store.state['debate_phase']['phase'], 'verify')
            self.assertEqual(store.state['verification_gate']['status'], 'no_executed_test')
            self.assertIsNone(store.state['learning_submission'])
            retry_calls = [(message, names) for message, names in calls if 'NO new executed' in message]
            self.assertTrue(retry_calls)
            self.assertTrue(all('hypothesis_test' in names and 'run_command' in names for _, names in retry_calls))
            self.assertIn('CURRENT RESPONSE ONLY', calls[-1][0])
            tools.execute.assert_not_called()

    def test_candidate_mode_reserves_40_and_never_allows_execution(self):
        calls = []
        class FakeTeam:
            def __init__(self, client, tools, store, config):
                self.contexts, self.prompts, self.store = {}, {}, store
            def ask(self, role, instruction, **kwargs):
                calls.append((instruction, kwargs))
                self.store.state['requests'] += kwargs['request_limit']
                if kwargs.get('submission'):
                    self.store.state['learning_submission'] = {'answer': 'Candidate, not verified', 'status': 'completed'}
                return 'Candidate, not verified'
        with TemporaryDirectory() as temp, patch('magents.debate.Team', FakeTeam):
            store = Store(Path(temp) / 'run', 'task', {})
            diagnose_debate(None, None, store, {'max_requests': 40, 'debate_candidates_only': True},
                            'obsolete validation prompt', 'source')
            self.assertEqual([kw['request_limit'] for _, kw in calls], [7, 1, 7, 1, 8, 1, 8, 1, 6])
            self.assertEqual(store.state['requests'], 40)
            for _, kwargs in calls:
                self.assertIsNotNone(kwargs['allowed_tools'])
                self.assertFalse({'run_command', 'hypothesis_test', 'replay_task', 'lst_single_channel'}
                                 & kwargs['allowed_tools'])
            self.assertIn('Peer revision', calls[6][0])
            self.assertEqual(store.state['debate_phase']['phase'], 'synthesis')

    def test_extraction_continues_context_and_saves_separate_candidate(self):
        from magents.engine import Team
        from unittest.mock import MagicMock
        candidate = dict(status='candidate', title='Typed inputs', when='Multi-file calls fail',
            steps=['Inspect actual JSON types'], boundaries=['Follow the specific tool schema'],
            verified_findings=['One probe read files'], unverified_claims=['Task accuracy gain'],
            evidence_refs=['learning:1'], reason='Candidate only')
        class Client:
            def create_message(self, system, messages, definitions, max_tokens):
                self.assertion = messages[0]['content']
                return {'content': [{'type': 'tool_use', 'id': 'skill', 'name': 'submit_skill', 'input': candidate}]}
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'task', {})
            store.state['requests'] = 24
            store.state['learning_submission'] = {'answer': 'diagnosis preserved'}
            tools = MagicMock(definitions=[])
            client = Client()
            team = Team(client, tools, store, {})
            team.contexts['learner'] = [{'role': 'user', 'content': 'prior debate'}]
            team.prompts['learner'] = 'test'
            extract_after_debate(team, {'max_requests': 32, 'max_tokens': 100, 'max_role_requests': 24,
                                       'tool_output_chars': 1000})
            self.assertEqual(client.assertion, 'prior debate')
            self.assertEqual(store.state['requests'], 25)
            self.assertEqual(store.state['skill_submission'], candidate)
            self.assertEqual(store.state['learning_submission']['answer'], 'diagnosis preserved')

    def test_plain_text_interim_reports_reach_peer_without_submit_tool(self):
        from unittest.mock import MagicMock
        calls = []
        class Client:
            def create_message(self, system, messages, definitions, max_tokens):
                names = {t['name'] for t in definitions}
                calls.append((names, messages[-1]['content']))
                if 'submit_learning' in names:
                    return {'content': [{'type': 'tool_use', 'id': 'final', 'name': 'submit_learning',
                        'input': {'status': 'incomplete', 'answer': 'No controlled validation yet',
                                  'reason': 'Diagnostic test fixture', 'evidence_refs': []}}]}
                return {'content': [{'type': 'text', 'text': f'Unverified opinion {len(calls)}; needs testing.'}]}
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'task', {})
            tools = MagicMock(definitions=[{'name': 'trace_read'}, {'name': 'run_command'}])
            diagnose_debate(Client(), tools, store, {'max_requests': 24, 'max_role_requests': 24,
                'max_tokens': 200, 'tool_output_chars': 1000}, 'prompt', 'source')
            self.assertEqual(len(calls), 9)
            for i in (1, 3, 5, 7):
                self.assertEqual(calls[i][0], set())
            self.assertIn('Unverified opinion 4', calls[4][1])
            self.assertIn('Unverified opinion 2', calls[6][1])
            self.assertNotIn('submit_learning', set.union(*(c[0] for c in calls[:8])))
            self.assertEqual(store.state['learning_submission']['status'], 'incomplete')
            rows = [json.loads(line) for line in (store.root / 'events.jsonl').read_text().splitlines()]
            reports = [r for r in rows if r['kind'] == 'debate_report' and r['phase'] != 'synthesis']
            self.assertEqual(len(reports), 4)
            self.assertTrue(all(r['verified'] is False for r in reports))

    def test_private_contexts_and_delayed_exchange(self):
        calls = []
        teams = []
        class FakeTeam:
            def __init__(self, client, tools, store, config):
                self.contexts, self.prompts, self.store = {}, {}, store
                self.index = len(teams)
                teams.append(self)
            def ask(self, role, message, **kwargs):
                calls.append((self.index, message, kwargs['request_limit']))
                self.store.state['requests'] += 1
                result = {'status': 'completed', 'answer': f'report-{len(calls)}'}
                self.store.state['learning_submission'] = result
                return json.dumps(result)
        with TemporaryDirectory() as temp, patch('magents.debate.Team', FakeTeam):
            store = Store(Path(temp) / 'run', 'task', {})
            diagnose_debate(None, None, store, {'max_requests': 24}, 'prompt', 'source')
            self.assertEqual([c[0] for c in calls], [0, 0, 1, 1, 0, 0, 1, 1, 0])
            self.assertEqual([c[2] for c in calls], [2, 1, 2, 1, 8, 1, 13, 1, 16])
            self.assertNotIn('report-2', calls[2][1])
            self.assertIn('report-4', calls[4][1])
            self.assertIn('report-2', calls[6][1])
            self.assertIn('report-8', calls[8][1])
            self.assertIsNot(teams[0].contexts, teams[1].contexts)
            self.assertEqual(store.state['requests'], 9)

    def test_blocked_report_calls_preserve_text_and_advance_without_extra_requests(self):
        from unittest.mock import MagicMock
        for include_text in (True, False):
            with self.subTest(include_text=include_text), TemporaryDirectory() as temp:
                calls = []
                class Client:
                    def create_message(self, system, messages, definitions, max_tokens):
                        calls.append(messages[-1]['content'])
                        if len(calls) == 6:  # A challenge report, after both independent reports
                            blocks = [{'type': 'text', 'text': 'Partial opinion needs audit.'}] if include_text else []
                            return {'content': blocks + [{'type': 'tool_use', 'id': 'blocked',
                                'name': 'run_command', 'input': {'command': 'must not run'}}]}
                        if any(t['name'] == 'submit_learning' for t in definitions):
                            return {'content': [{'type': 'tool_use', 'id': 'final', 'name': 'submit_learning',
                                'input': {'status': 'incomplete', 'answer': 'Still unverified',
                                          'reason': 'Fixture', 'evidence_refs': []}}]}
                        return {'content': [{'type': 'text', 'text': 'Earlier evidence-window opinion.'}]}
                store = Store(Path(temp) / 'run', 'task', {})
                tools = MagicMock(definitions=[{'name': 'trace_read'}, {'name': 'run_command'}])
                diagnose_debate(Client(), tools, store, {'max_requests': 24, 'max_role_requests': 24,
                    'max_tokens': 200, 'tool_output_chars': 1000}, 'prompt', 'source')
                tools.execute.assert_not_called()
                self.assertEqual(store.state['requests'], 9)
                self.assertEqual(store.state['learning_submission']['status'], 'incomplete')
                peer_input = calls[6]
                if include_text:
                    self.assertIn('Partial opinion needs audit.', peer_input)
                    self.assertIn('blocked and did not execute', peer_input)
                else:
                    self.assertIn('no usable model-authored interim report', peer_input)
                rows = [json.loads(line) for line in (store.root / 'events.jsonl').read_text().splitlines()]
                handoff = next(r for r in rows if r['kind'] == 'debate_report_handoff')
                self.assertEqual(handoff['status'], 'recovered_text' if include_text else 'missing_report')
                self.assertTrue(any(r['kind'] == 'debate_report' and r['analyst'] == 'B'
                                    and r['phase'] == 'challenge_and_test' for r in rows))

    def test_read_phase_blocks_hidden_compute_and_only_exposes_submission_at_end(self):
        from magents.engine import Team
        from unittest.mock import MagicMock
        schemas = []
        class Client:
            def create_message(self, system, messages, definitions, max_tokens):
                schemas.append({t['name'] for t in definitions})
                if len(schemas) == 1:
                    return {'content': [{'type': 'tool_use', 'id': 'bad', 'name': 'run_command',
                                         'input': {'command': 'true'}}]}
                return {'content': [{'type': 'tool_use', 'id': 'done', 'name': 'submit_learning',
                    'input': {'status': 'incomplete', 'answer': 'Needs verification',
                              'reason': 'Preliminary only', 'evidence_refs': []}}]}
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'task', {})
            tools = MagicMock(definitions=[{'name': 'trace_read'}, {'name': 'run_command'}])
            team = Team(Client(), tools, store, {'max_requests': 24, 'max_tokens': 100,
                                               'max_role_requests': 24, 'tool_output_chars': 1000})
            team.contexts['learner'], team.prompts['learner'] = [], 'test'
            team.ask('learner', 'read only', request_limit=2, reserve=18,
                     allowed_tools={'trace_read'}, submission_only_at_end=True, submission='submit_learning')
            self.assertNotIn('run_command', schemas[0])
            self.assertEqual(schemas[1], {'submit_learning'})
            tools.execute.assert_not_called()

    def test_full_phase_consumption_preserves_peer_and_synthesis_budgets(self):
        limits = []
        class FakeTeam:
            def __init__(self, client, tools, store, config):
                self.contexts, self.prompts, self.store = {}, {}, store
            def ask(self, role, instruction, **kwargs):
                limits.append(kwargs['request_limit'])
                self.store.state['requests'] += kwargs['request_limit']
                self.store.state['learning_submission'] = {'answer': 'Unverified hypothesis', 'status': 'incomplete'}
                return json.dumps(self.store.state['learning_submission'])
        with TemporaryDirectory() as temp, patch('magents.debate.Team', FakeTeam):
            store = Store(Path(temp) / 'run', 'task', {})
            diagnose_debate(None, None, store, {'max_requests': 24}, 'prompt', 'source')
            self.assertEqual(limits, [2, 1, 2, 1, 6, 1, 6, 1, 4])
            self.assertEqual(store.state['requests'], 24)
