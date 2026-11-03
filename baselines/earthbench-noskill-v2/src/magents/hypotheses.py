"""Small evidence-linked hypothesis checks; conclusions remain agent judgments."""
import json
import shlex
from magents.submissions import tool, TEXT

PROMPT = """
Hypothesis verification is available. Before attributing repeated failure to data,
environment or a tool defect, or changing task coverage/statistics because of it,
use hypothesis_propose to state the hypothesis, alternative explanation, minimal
test and predicted supporting/refuting observations. Use hypothesis_test to execute
the test, then hypothesis_conclude with its returned test IDs. A successful command
does not establish the hypothesis: distinguish observations from interpretations;
inconclusive is valid. Compare actual submitted arguments, not intended arguments.
Diagnostic run_command may inspect paths, types, source and make temporary probes;
it must not replace the final official-tool computation. Diagnostic commands are
not technically restricted to this purpose, so compliance requires trace review.
Do not overwrite original inputs. Keep checks small and preserve task coverage.
For hypothesis_test list actual prerequisite files in required_inputs (not output
paths). They are checked in the current sandbox before dispatch; a failed preflight
is not a test of the causal claim. Follow artifact_access: in single-source mode old
/workspace paths are read-only and new outputs belong under /diagnostics. Other modes
provide explicit source mappings. Recreate only missing artifacts. Use [] for tests needing no
files or when file absence is itself the observation being tested. Use returned
hypothesis IDs such as h1, not learning event references, to address hypotheses.
For a causal comparison keep inputs, tool, threshold and other settings unchanged
except the factor being tested. A threshold change is a separate diagnostic, never
a replacement for the task's required threshold. NaN alone does not identify a cause.
test_executed=false means no target call occurred, not that the target tool failed.
"""


class Hypotheses:
    def __init__(self, execute):
        self.execute = execute
        self.records = {}
        self.definitions = [
            tool('hypothesis_propose', 'Register a falsifiable hypothesis before testing; no automatic truth judgment.',
                 {key: TEXT for key in ('hypothesis', 'alternative', 'test_plan', 'supports_if', 'refutes_if')}),
            tool('hypothesis_test', 'Run a diagnostic command or official tool for a registered hypothesis. Output is evidence, not proof.',
                 {'hypothesis_id': TEXT, 'tool_name': TEXT,
                  'arguments': {'type': 'object', 'additionalProperties': True},
                  'required_inputs': {'type': 'array', 'items': TEXT,
                                      'description': 'Actual prerequisite file paths in current sandbox; exclude outputs. Empty only when no files are needed or testing absence itself.'}}),
            tool('hypothesis_conclude', 'Record an agent interpretation tied to tests. Does not certify causality or correctness.',
                 {'hypothesis_id': TEXT, 'test_ids': {'type': 'array', 'items': TEXT, 'minItems': 1},
                  'verdict': {'type': 'string', 'enum': ['supported', 'not_supported', 'inconclusive']},
                  'observation': TEXT, 'reason': TEXT, 'next_action': TEXT})]

    def snapshot(self):
        return {key: {'proposal': value['proposal'], 'tests': list(value['tests'].values()),
                      'agent_conclusion': value.get('conclusion'),
                      'semantic_validation': False} for key, value in self.records.items()}

    def call(self, name, arguments):
        if name == 'hypothesis_propose':
            for key in ('hypothesis', 'alternative', 'test_plan', 'supports_if', 'refutes_if'):
                if not isinstance(arguments.get(key), str) or not arguments[key].strip():
                    raise ValueError(f'{key} must be nonempty')
            identifier = f'h{len(self.records) + 1}'
            self.records[identifier] = {'proposal': dict(arguments), 'tests': {}}
            return json.dumps({'hypothesis_id': identifier, 'status': 'untested'})
        if arguments['hypothesis_id'] not in self.records:
            raise ValueError('Unknown hypothesis ID. Use an ID returned by a successful hypothesis_propose, '
                             f'not an evidence reference. Available IDs: {list(self.records)}')
        record = self.records[arguments['hypothesis_id']]
        if name == 'hypothesis_test':
            identifier = f"{arguments['hypothesis_id']}:t{len(record['tests']) + 1}"
            attempted = False
            try:
                paths = arguments.get('required_inputs', [])
                if paths:
                    command = shlex.join(['python', '-c',
                        'import json,os,sys; print(json.dumps([p for p in json.loads(sys.argv[1]) if not (os.path.isfile(p) and os.access(p,os.R_OK))]))',
                        json.dumps(paths)])
                    check = json.loads(self.execute('run_command', {'command': command}))
                    if check['exit_code'] != 0 or check['timed_out']:
                        raise ValueError('Input preflight failed: ' + check['stdout'])
                    missing = json.loads(check['stdout'])
                    if missing:
                        result = {'test_id': identifier, 'tool_name': arguments['tool_name'],
                                  'arguments': arguments['arguments'], 'required_inputs': paths,
                                  'call_attempted': False, 'computation_success': None, 'dispatch_error': False,
                                  'test_executed': False, 'status': 'missing_inputs',
                                  'missing_inputs': missing,
                                  'output': 'Recreate prerequisites in this workspace. This does not test the causal hypothesis.'}
                        record['tests'][identifier] = result
                        return json.dumps(result)
                attempted = True
                output = self.execute(arguments['tool_name'], arguments['arguments'])
                failed = False
            except Exception as error:
                output, failed = f'{type(error).__name__}: {error}', True
            result = {'test_id': identifier, 'tool_name': arguments['tool_name'],
                      'arguments': arguments['arguments'], 'required_inputs': arguments.get('required_inputs', []),
                      'dispatch_error': failed, 'output': output,
                      'call_attempted': attempted,
                      'test_executed': (None if failed else True) if attempted else False,
                      'status': ('call_error_execution_unknown' if attempted else 'preflight_error') if failed else 'returned',
                      'computation_success': None}
            record['tests'][identifier] = result
            return json.dumps(result, ensure_ascii=False)
        if name == 'hypothesis_conclude':
            refs = arguments['test_ids']
            if not refs or any(ref not in record['tests'] for ref in refs):
                raise ValueError('Cite existing tests belonging to this hypothesis')
            if arguments['verdict'] not in {'supported', 'not_supported', 'inconclusive'}:
                raise ValueError('Invalid verdict')
            if arguments['verdict'] != 'inconclusive' and all(
                    record['tests'][ref].get('test_executed') is False or record['tests'][ref]['dispatch_error'] for ref in refs):
                raise ValueError('All cited tests failed to execute; use inconclusive or run a valid test')
            for key in ('observation', 'reason', 'next_action'):
                if not isinstance(arguments.get(key), str) or not arguments[key].strip():
                    raise ValueError(f'{key} must be nonempty')
            record['conclusion'] = dict(arguments)
            return json.dumps({'recorded': True, 'semantic_validation': False,
                               'conclusion': arguments}, ensure_ascii=False)
        raise ValueError(f'Unknown hypothesis action: {name}')
