"""Target-tool-only argument probe; no tool execution or task scoring."""
import argparse
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

from skill_context_probe import ROOT, TARGET, prepare, classify
from magents.adapters import api_settings, client_for
from rsagent.billing.qwen import policy_for, quote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--schema-comparison', action='store_true',
                        help='Compare union versus array-only schema with identical plain task')
    args = parser.parse_args()
    source = ROOT / 'var/runs/q93-single-skill-replay-01/round-1/with-skill'
    _, definitions, _, _, expected = prepare(source)
    definitions = [d for d in definitions if d['name'] == TARGET]
    assert len(definitions) == 1
    skill = json.loads((source / 'run.json').read_text())['skill']
    roster = '\n'.join(f'{i}: red={r}; nir={n}; lst={l}' for i, (r, n, l) in enumerate(
        zip(expected['red_paths'], expected['nir_paths'], expected['lst_paths']), 1))
    task = ('All files listed below exist. Generate exactly one calculate_mean_lst_by_ndvi call '
            'covering all 36 listed scenes, ndvi_threshold=0.7 and mode=above. '
            'Do not inspect or regenerate files. The call will only be recorded, not executed.\n' + roster)
    example = ('Parameter-format example only; use the actual paths above, not these placeholders.\n'
               'Correct: {"red_paths": ["a.tif", "b.tif"]}\n'
               'Incorrect: {"red_paths": "[\\"a.tif\\", \\"b.tif\\"]"}\n'
               'Apply the same native-array format to red_paths, nir_paths and lst_paths.')
    conditions = {
        'schema_only': task,
        'original_skill': task + '\nAdvisory skill:\n' + skill,
        'skill_plus_example': task + '\nAdvisory skill:\n' + skill + '\n' + example,
    }
    schemas = {name: definitions for name in conditions}
    if args.schema_comparison:
        arrays = deepcopy(definitions)
        for key in ('red_paths', 'nir_paths', 'lst_paths'):
            prop = arrays[0]['input_schema']['properties'][key]
            del prop['anyOf']
            prop.update(type='array', items={'type': 'string'})
        conditions = {'union_schema': task, 'array_schema': task}
        schemas = {'union_schema': definitions, 'array_schema': arrays}
    max_requests = len(conditions) * 3
    system = 'Generate the requested tool call using the supplied tool schema and task data.'
    api_path = ROOT.parent / 'config/py-agent/config.json'
    api = api_settings(api_path, 'dashscope-qwen35-flash')
    policy = policy_for(api)
    # Conservative byte-count reservation for textual prompt + generous envelope.
    prompt_bound = max(len(json.dumps({'system': system, 'task': t, 'tools': definitions}).encode())
                       for t in conditions.values()) + 8192
    per_call = Decimal(quote({'prompt_tokens': prompt_bound, 'completion_tokens': 4096},
                             policy, reservation=True)['estimated_cost'])
    budget = Decimal('1')
    assert max_requests * per_call <= budget, 'Preflight exceeds this experiment budget'
    manifest = {'conditions': conditions, 'definitions': definitions, 'schemas': schemas, 'system': system,
                'expected': expected, 'source': str(source), 'repeats': 3, 'max_requests': max_requests,
                'model': api['model'], 'temperature': api.get('temperature'),
                'enable_thinking': api.get('enable_thinking'), 'max_tokens': 4096,
                'budget_cny': str(budget), 'reserved_total_cny': str(max_requests * per_call),
                'note': 'Independent calls, rotated order, no paired seeds. Only target tool; no original task history. Not comparable to task accuracy.'}
    if args.dry_run:
        print(json.dumps({k: manifest[k] for k in ('max_requests', 'budget_cny', 'reserved_total_cny')}))
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    client = client_for(api_path, api)
    results, cost = [], Decimal(0)
    names = list(conditions)
    for repeat in range(3):
        offset = repeat % len(names)
        for name in names[offset:] + names[:offset]:
            if cost + per_call > budget:
                raise RuntimeError('Budget exhausted')
            response = client.create_message(system, [{'role': 'user', 'content': conditions[name]}],
                                             schemas[name], max_tokens=4096)
            with (root / 'events.jsonl').open('a') as log:
                log.write(json.dumps({'repeat': repeat + 1, 'condition': name, 'response': response}, ensure_ascii=False) + '\n')
            billing = quote(response.get('raw_usage', response.get('usage')), policy)
            if billing.get('estimated_cost') is None:
                raise RuntimeError('Missing billable usage; stopping')
            cost += Decimal(billing['estimated_cost'])
            row = {'repeat': repeat + 1, 'condition': name, **classify(response, expected),
                   'usage': response.get('usage'), 'estimated_cny': billing['estimated_cost']}
            results.append(row)
            (root / 'summary.json').write_text(json.dumps({'runs': results, 'estimated_cny': str(cost)}, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
