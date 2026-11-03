import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch
from magents.engine import Store, StageYield
from magents.skill_replay import extract_and_replay
from magents.learning import LearningTools


class SkillReplayTests(TestCase):
    def test_paired_solver_settings_are_identical_except_skill(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            tools = LearningTools(MagicMock(definitions=[]), {}, root, 93, root / 'runner.py',
                                  replay_profile='dashscope-qwen35-flash', replay_max_requests=40,
                                  replay_official_only=False)
            with patch('magents.learning.subprocess.run', return_value=MagicMock(returncode=0)) as run:
                tools.execute('replay_task', {'skill': 'A conditional advisory skill'})
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertEqual(sum('--skill-file' in cmd for cmd in commands), 1)
            for cmd in commands:
                self.assertEqual(cmd[cmd.index('--profile') + 1], 'dashscope-qwen35-flash')
                self.assertEqual(cmd[cmd.index('--max-requests') + 1], '40')
                self.assertNotIn('--official-only', cmd)
                self.assertIn('--no-skill-retrieval', cmd)

    def test_candidate_pair_and_abstention(self):
        for status in ('candidate', 'abstain'):
            with self.subTest(status=status), TemporaryDirectory() as temp:
                store = Store(Path(temp) / 'run', 'task', {})
                store.state.update(requests=10, learning_submission={'status': 'incomplete', 'answer': 'Finding with limits'})
                candidate = dict(status=status, title='Use correct input types', when='Multi-file call',
                                 steps=['Check actual types'], boundaries=['Follow tool contract'],
                                 evidence_refs=['source-1:87'], verified_findings=['example'],
                                 unverified_claims=['benefit'], reason='not for solver')
                team = MagicMock(store=store, config={'max_requests': 40}, prompts={})
                team.contexts = {'learner': [{'role': 'user', 'content': 'old context'}]}
                def extract(*args, **kwargs):
                    self.assertEqual(kwargs['submission'], 'submit_skill')
                    self.assertEqual(kwargs['request_limit'], 8)
                    store.state['skill_submission'] = candidate
                team.ask.side_effect = extract
                with patch('magents.skill_replay.LearningTools') as replay:
                    replay.return_value.execute.return_value = json.dumps({'round': 1, 'results': {}})
                    extract_and_replay(team, task_id=93, runner_script=Path('runner.py'),
                                       profile='qwen-test', official_only=False)
                    self.assertEqual(team.config['max_requests'], 18)
                    self.assertEqual(team.contexts['learner'][0]['content'], 'old context')
                    self.assertEqual(store.state['learning_submission']['status'], 'incomplete')
                    if status == 'candidate':
                        sent = json.loads(replay.return_value.execute.call_args.args[1]['skill'])
                        self.assertEqual(set(sent), {'title', 'when', 'steps', 'boundaries'})
                        self.assertEqual(replay.call_args.kwargs['replay_profile'], 'qwen-test')
                        self.assertFalse(replay.call_args.kwargs['replay_official_only'])
                        self.assertEqual(store.state['status'], 'skill_replay_finished_pending_audit')
                    else:
                        replay.assert_not_called()
                        self.assertEqual(store.state['status'], 'skill_abstained')

    def test_extraction_failure_never_starts_replay(self):
        with TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'run', 'task', {})
            store.state['learning_submission'] = {'answer': 'diagnosis'}
            team = MagicMock(store=store, config={}, prompts={})
            team.ask.side_effect = StageYield('No candidate')
            with patch('magents.skill_replay.LearningTools') as replay, self.assertRaises(StageYield):
                extract_and_replay(team, task_id=93, runner_script=Path('runner.py'), profile='qwen-test')
            replay.assert_not_called()
