import json
import unittest
from unittest.mock import MagicMock, patch
from magents.hypotheses import Hypotheses
from magents.adapters import WorkspaceTools
from magents.engine import Team


class HypothesisTests(unittest.TestCase):
    def proposal(self, module):
        return json.loads(module.call('hypothesis_propose', dict(
            hypothesis='path missing', alternative='encoded list', test_plan='inspect input',
            supports_if='actual file missing', refutes_if='file exists')))['hypothesis_id']

    def test_linked_cycle_without_truth_certification(self):
        execute = MagicMock(return_value='file exists')
        module = Hypotheses(execute)
        h = self.proposal(module)
        test = json.loads(module.call('hypothesis_test', dict(
            hypothesis_id=h, tool_name='run_command', arguments={'command': 'test -f input.tif'})))
        conclusion = dict(hypothesis_id=h, test_ids=[test['test_id']], verdict='not_supported',
                          observation='file exists', reason='missing-file explanation not supported',
                          next_action='inspect actual argument type')
        result = json.loads(module.call('hypothesis_conclude', conclusion))
        self.assertFalse(result['semantic_validation'])
        conclusion['hypothesis_id'] = self.proposal(module)
        with self.assertRaises(ValueError):
            module.call('hypothesis_conclude', conclusion)

    def test_failed_probe_is_evidence_not_success(self):
        module = Hypotheses(MagicMock(side_effect=RuntimeError('bad input')))
        h = self.proposal(module)
        result = json.loads(module.call('hypothesis_test', dict(
            hypothesis_id=h, tool_name='official_tool', arguments={})))
        self.assertTrue(result['dispatch_error'])
        self.assertIsNone(result['test_executed'])
        self.assertEqual(result['status'], 'call_error_execution_unknown')
        self.assertIn('bad input', result['output'])
        self.assertNotIn('conclusion', module.records[h])

    def test_missing_prerequisites_do_not_run_target_or_support_a_claim(self):
        execute = MagicMock(return_value=json.dumps({'stdout': '["/workspace/missing.tif"]',
                                                     'exit_code': 0, 'timed_out': False}))
        module = Hypotheses(execute)
        h = self.proposal(module)
        result = json.loads(module.call('hypothesis_test', dict(hypothesis_id=h,
            tool_name='calculate_mean_lst_by_ndvi', arguments={},
            required_inputs=['/workspace/missing.tif'])))
        self.assertFalse(result['test_executed'])
        self.assertEqual(module.snapshot()[h]['tests'][0]['status'], 'missing_inputs')
        self.assertEqual(execute.call_count, 1)
        with self.assertRaisesRegex(ValueError, 'failed to execute'):
            module.call('hypothesis_conclude', dict(hypothesis_id=h, test_ids=[result['test_id']],
                verdict='supported', observation='missing file', reason='test did not run', next_action='recreate'))

    def test_existing_prerequisites_dispatch_original_arguments(self):
        execute = MagicMock(side_effect=[json.dumps({'stdout': '[]', 'exit_code': 0,
                                                    'timed_out': False}), '290'])
        module = Hypotheses(execute)
        h = self.proposal(module)
        arguments = {'red_paths': ['a']}
        result = json.loads(module.call('hypothesis_test', dict(hypothesis_id=h,
            tool_name='official', arguments=arguments, required_inputs=['a'])))
        self.assertEqual(result['output'], '290')
        self.assertTrue(result['test_executed'])
        self.assertIsNone(result['computation_success'])
        execute.assert_called_with('official', arguments)

    def test_bad_hypothesis_id_explains_namespace(self):
        module = Hypotheses(MagicMock())
        with self.assertRaisesRegex(ValueError, 'successful hypothesis_propose'):
            module.call('hypothesis_test', {'hypothesis_id': 'learning:14'})

    def test_diagnostics_use_official_environment_and_no_recursive_dispatch(self):
        with patch('magents.adapters.NamespaceRunner'), patch('magents.adapters.ToolBox') as box:
            box.return_value.definitions = []
            tools = WorkspaceTools(None, [], {}, official_only=True, hypothesis_verification=True)
            tools.earth = MagicMock()
            tools.earth.names = {'calculate_mean_lst_by_ndvi'}
            tools.earth.runner.run.return_value = MagicMock(stdout='ok', exit_code=0, timed_out=False)
            tools.diagnostic_execute('run_command', {'command': 'pwd'})
            tools.earth.runner.run.assert_called_once()
            tools.runner.run.assert_not_called()
            with self.assertRaises(ValueError):
                tools.execute('run_command', {'command': 'pwd'})
            with self.assertRaises(ValueError):
                tools.diagnostic_execute('hypothesis_test', {})
            args = {'red_paths': ['a', 'b']}
            tools.diagnostic_execute('calculate_mean_lst_by_ndvi', args)
            tools.earth.execute.assert_called_once_with('calculate_mean_lst_by_ndvi', args)

    def test_prompt_has_explicit_permission_boundary(self):
        team = Team(None, None, MagicMock(), {'official_only': True, 'hypothesis_verification': True})
        for prompt in team.prompts.values():
            self.assertIn('final calculations must use official tools', prompt)
            self.assertNotIn('Do not write or execute auxiliary code', prompt)


if __name__ == '__main__':
    unittest.main()
