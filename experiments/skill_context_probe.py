"""Three-condition micro experiment. Generate calls only; no raster recomputation."""
import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'python-agent/src')]
from magents.adapters import api_settings, client_for
from magents.engine import Team, EVIDENCE_TOOL
from rsagent.billing.qwen import policy_for, quote

TARGET = 'calculate_mean_lst_by_ndvi'
CONDITIONS = ['short_skill', 'long_skill', 'long_skill_reminder']


def reconstruct(rows, before=83, chars=16000):
    """Reconstruct executor-visible history, preserving original tool-result groups."""
    messages, pending = [], []
    for row in rows:
        if row['id'] >= before:
            break
        if row.get('role') != 'executor':
            continue
        if row['kind'] == 'tool_result':
            output = row['output']
            visible = output[:chars]
            if len(output) > len(visible):
                visible += '\n[Truncated display; use evidence_read for full event.]'
            pending.append({'type': 'tool_result', 'tool_use_id': row['call']['id'],
                            'content': f"Evidence ID: {row['id']}\n{visible}",
                            'is_error': row.get('dispatch_error', False)})
            continue
        if pending:
            messages.append({'role': 'user', 'content': pending})
            pending = []
        if row['kind'] == 'handoff':
            messages.append({'role': 'user', 'content': row['instruction']})
        elif row['kind'] == 'response':
            if row['response'].get('stop_reason') in ('length', 'max_tokens'):
                messages.append({'role': 'user', 'content': 'Previous output was truncated and no tools ran. Produce a shorter response.'})
            else:
                messages.append({'role': 'assistant', 'content': row['response']['content']})
    if pending:
        messages.append({'role': 'user', 'content': pending})
    return messages


def classify(response, expected):
    calls = [b for b in response.get('content', []) if b['type'] == 'tool_use']
    valid = len(calls) == 1 and calls[0]['name'] == TARGET
    args = calls[0].get('input', {}) if valid else {}
    arrays = valid and all(isinstance(args.get(k), list) for k in ('red_paths', 'nir_paths', 'lst_paths'))
    return {'one_target_call': valid, 'native_arrays': arrays,
            'exact_inputs': arrays and all(args.get(k) == v for k, v in expected.items()),
            'truncated': response.get('stop_reason') in ('length', 'max_tokens'),
            'argument_types': {k: type(v).__name__ for k, v in args.items()}}


def prepare(source):
    rows = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines()]
    manifest = json.loads((source / 'run.json').read_text())
    state = json.loads((source / 'state.json').read_text())
    original = next(r for r in rows if r['id'] == 84)['call']['input']
    expected = {k: json.loads(original[k]) for k in ('red_paths', 'nir_paths', 'lst_paths')}
    expected.update(ndvi_threshold=0.7, mode='above')
    for paths in (expected[k] for k in ('red_paths', 'nir_paths', 'lst_paths')):
        for path in paths:
            host = source / 'workspace' / path[len('/workspace/'):] if path.startswith('/workspace/') else Path(path)
            if not host.is_file():
                raise ValueError(f'Missing existing input: {host}')
    roster = '\n'.join(f'{i}: red={r}; nir={n}; lst={l}' for i, (r,n,l) in
                       enumerate(zip(expected['red_paths'], expected['nir_paths'], expected['lst_paths']), 1))
    instruction = ('Micro-test: all LST rasters below already exist and match their source bands. '
                   'Do not regenerate or inspect files. Emit exactly one calculate_mean_lst_by_ndvi '
                   'tool call for all listed scenes, ndvi_threshold=0.7 and mode=above. '
                   'This call is recorded for inspection only and will not be executed.\n' + roster)
    long = reconstruct(rows, chars=manifest['config']['tool_output_chars'])
    # Same original task/skill, same complete catalog, same current request in all arms.
    short = [{'role': 'user', 'content': 'Task:\n' + state['task']}]
    contexts = {CONDITIONS[0]: short, CONDITIONS[1]: long, CONDITIONS[2]: deepcopy(long)}
    for name, history in contexts.items():
        content = instruction
        if name == 'long_skill_reminder':
            content += '\nRepeat of the SAME original advisory skill, unchanged:\n' + manifest['skill']
        history.append({'role': 'user', 'content': content})
    definitions = [EVIDENCE_TOOL] + next(r for r in rows if r['kind'] == 'tool_catalog')['definitions']
    system = Team(None, None, None, manifest['config']).prompts['executor']
    failed = next(r for r in rows if r['id'] == 93)
    feedback = ('Separate error-interpretation case, NOT the outcome of your preceding call. '
                'The preceding call was not executed. Inspect this actual historical invocation '
                'and its complete returned diagnostic. In concise prose explain the most likely '
                'cause, specific correction and remaining uncertainty. Do not call tools or solve '
                'the task.\n' + json.dumps({'call': failed['call'], 'output': failed['output']}, ensure_ascii=False))
    return contexts, definitions, system, feedback, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'var/runs/q93-single-skill-replay-01/round-1/with-skill')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    contexts, definitions, system, feedback, expected = prepare(args.source.resolve())
    if args.dry_run:
        print(json.dumps({'conditions': {k: {'messages': len(v), 'chars': len(json.dumps(v))} for k,v in contexts.items()},
                          'scenes': len(expected['lst_paths']), 'max_requests': 18}))
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    api_path = ROOT.parent / 'config/py-agent/config.json'
    api = api_settings(api_path, 'dashscope-qwen35-flash')
    client, policy = client_for(api_path, api), policy_for(api)
    (root / 'run.json').write_text(json.dumps({'source': str(args.source.resolve()), 'conditions': CONDITIONS,
        'repeats': 3, 'max_requests': 18, 'max_output_tokens': 4096, 'model': api['model'],
        'temperature': api.get('temperature'), 'enable_thinking': api.get('enable_thinking'),
        'system': system, 'definitions': definitions, 'contexts': contexts, 'feedback': feedback,
        'expected': expected, 'note': 'Rotated condition order. No seed pairing, no tool execution. Not an accuracy/skill-gain experiment.'}, ensure_ascii=False, indent=2))
    results, costs = [], []
    for repeat in range(3):
        for condition in CONDITIONS[repeat:] + CONDITIONS[:repeat]:
            row = {'repeat': repeat + 1, 'condition': condition}
            try:
                history = deepcopy(contexts[condition])
                for phase in ('call', 'explanation'):
                    response = client.create_message(system, history, definitions if phase == 'call' else [], max_tokens=4096)
                    with (root / 'events.jsonl').open('a') as log:
                        log.write(json.dumps({**row, 'phase': phase, 'response': response}, ensure_ascii=False) + '\n')
                    price = quote(response.get('raw_usage', response.get('usage')), policy)
                    if price.get('estimated_cost') is not None:
                        costs.append(Decimal(price['estimated_cost']))
                    row[phase + '_usage'] = response.get('usage')
                    if phase == 'call':
                        row.update(classify(response, expected))
                        # Keep the response and explicitly close unexecuted calls.
                        if response.get('stop_reason') not in ('length', 'max_tokens'):
                            history.append({'role': 'assistant', 'content': response['content']})
                            calls = [b for b in response['content'] if b['type'] == 'tool_use']
                            if calls:
                                history.append({'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': b['id'],
                                    'content': 'Not executed: call-generation micro-test only.'} for b in calls]})
                        history.append({'role': 'user', 'content': feedback})
                    else:
                        row['explanation'] = '\n'.join(b.get('text','') for b in response['content'] if b['type'] == 'text')
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            results.append(row)
            (root / 'summary.json').write_text(json.dumps({'runs': results, 'estimated_cny': str(sum(costs, Decimal(0)))}, ensure_ascii=False, indent=2))
            print(json.dumps({k: v for k,v in row.items() if k not in ('explanation', 'call_usage', 'explanation_usage')}), flush=True)


if __name__ == '__main__':
    main()
