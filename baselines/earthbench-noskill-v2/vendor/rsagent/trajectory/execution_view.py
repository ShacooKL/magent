"""Deterministic execution views over immutable, authorized trajectory evidence.

No model, semantic error classifier, answers or historical filesystem reconstruction.
The optional inventory is explicitly host-frozen metadata, not a historical observation.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import PurePosixPath
import re


def decode(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def migrate_packet(original, inventory=None, data_root=None, inventory_source=None):
    """Preserve every original record/artifact; add aliases and mechanical summaries."""
    packet = deepcopy(original)
    if packet.get('presentation'):
        raise ValueError('Already migrated; use the original packet')
    records = packet['records']
    actions = [(node['episode_id'], deepcopy(action)) for node in packet['nodes'] for action in node['actions']]
    view = {'version': 2, 'files': {}, 'actions': {}, 'coverage': [],
            'scope': 'Observed events only. Matching values or later success do not prove causal repair.'}
    path_ids = {}

    def path_label(path):
        if path not in path_ids:
            fid = 'F' + str(len(path_ids) + 1)
            path_ids[path] = fid
            eid = 'file:' + fid
            records[eid] = {'text': json.dumps({'path': path}, ensure_ascii=False),
                            'source': 'Exact string observed in authorized tool arguments/results'}
            view['files'][fid] = {'name': PurePosixPath(path).name, 'evidence_id': eid}
        return path_ids[path] + ':' + PurePosixPath(path).name

    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) <= 5:
                return [compact(v) for v in value]
            dates = sorted(set(m[0] for s in strings(value) if (m := re.findall(r'\d{4}-\d{2}-\d{2}', s))))
            result = {'type': 'array', 'count': len(value), 'head': [compact(v) for v in value[:2]],
                      'tail': [compact(value[-1])], 'full_content': 'expand original evidence'}
            if dates:
                result['observed_date_range'] = [dates[0], dates[-1]]
                result['unique_dates'] = len(dates)
            return result
        if isinstance(value, str):
            decoded = decode(value)
            if isinstance(decoded, (dict, list)):
                return {'type': 'JSON_encoded_string_NOT_array_or_object', 'decoded_preview': compact(decoded)}
            if '\n' not in value and not value.startswith('Result saved at ') and (value.startswith('/') or value.endswith(('.tif', '.tiff', '.png'))):
                return path_label(value)
            return value if len(value) <= 280 else value[:280] + ' … [preview only]'
        return value

    producers, call_ids = {}, {}
    last_by_tool = {}
    coverage = {}
    available = {row['name'] for row in inventory or []}
    new_nodes = []
    for index, (episode, action) in enumerate(actions, 1):
        aid = 'A' + str(index)
        if aid in records:
            raise ValueError('Action alias collision')
        raw_call = action['call_evidence_id']
        records[aid] = deepcopy(records[raw_call])
        call = decode(records[raw_call]['text'])
        result = decode(records[action['result_evidence_id']]['text']) if action.get('result_evidence_id') else {}
        nested_ids = action.get('nested_event_evidence_ids', [])
        nested = [(eid, decode(records[eid]['text'])) for eid in nested_ids]
        official = [(eid, e) for eid, e in nested if e.get('type') == 'official_tool_observation']
        actual = official[-1][1] if official else None
        name = actual['name'] if actual else action['name']
        args = actual['arguments'] if actual else call.get('arguments', {})
        output = actual.get('output', actual.get('error', '')) if actual else result.get('output', '')
        success = actual.get('success') if actual else result.get('success')
        summary = {'id': aid, 'tool': name, 'outer_tool': action['name'], 'step': action.get('step'),
            'reported_success': success, 'arguments': compact(args),
            'output': compact(decode(output)), 'call_evidence_id': aid,
            'raw_call_evidence_id': raw_call, 'result_evidence_id': action.get('result_evidence_id'),
            'nested_event_evidence_ids': nested_ids, 'file_inputs_from': [], 'used_by': []}
        if '[truncated]' in str(output):
            summary['historical_output_truncated'] = True
        if any(marker in str(output) for marker in ('ValidationError:', 'RasterioIOError:', 'Error:')):
            summary['output_contains_error_marker'] = True
        if action['name'] == 'record_official_return':
            summary['evidence_kind'] = 'MODEL_SELF_REPORT_NOT_EXECUTION_CERTIFICATION'
        for text in strings(args):
            if (episode, text) in producers:
                parent = producers[episode, text]
                if parent not in summary['file_inputs_from']:
                    summary['file_inputs_from'].append(parent)
                    view['actions'][parent]['used_by'].append(aid)
        if action['name'] == 'execute_checked_official_call':
            retained = call_ids.get((episode, call.get('arguments', {}).get('call_id')))
            if retained:
                summary['checked_arguments_from'] = retained
        parsed = decode(result.get('output'))
        if action['name'] == 'check_official_arguments' and isinstance(parsed, dict) and parsed.get('call_id'):
            call_ids[episode, parsed['call_id']] = aid
        if success is True and isinstance(output, str) and output.startswith('Result saved at '):
            output_path = output[len('Result saved at '):].strip()
            summary['output'] = {'saved_file': path_label(output_path)}
            producers[episode, output_path] = aid
        if (episode, name) in last_by_tool:
            previous = last_by_tool[episode, name]
            summary['previous_same_tool'] = previous
            if view['actions'][previous]['reported_success'] is False and success is True:
                view['actions'][previous]['later_same_tool_success'] = aid
                view['actions'][previous]['repair_status'] = 'unknown; same tool, inputs may differ'
        last_by_tool[episode, name] = aid
        if inventory and data_root and actual:
            for key, value in args.items():
                for path in strings(value):
                    prefix = str(data_root).rstrip('/') + '/'
                    if not path.startswith(prefix):
                        continue
                    relative = path[len(prefix):]
                    if relative not in available:
                        continue
                    pattern = re.sub(r'\d{4}-\d{2}-\d{2}', '<date>', relative)
                    group = coverage.setdefault((name, key, pattern), {'attempted': set(), 'reported_success': set()})
                    group['attempted'].add(relative)
                    if success is True:
                        group['reported_success'].add(relative)
        action.update(action_id=aid, call_evidence_id=aid, raw_call_evidence_id=raw_call)
        # A wrapper's evidence includes the actual inner official invocation and its schema.
        action['evidence_requirements'] = []
        if action['name'] == 'execute_checked_official_call' and official:
            action['evidence_requirements'] = [official[-1][0], 'tool_definition:' + name]
        view['actions'][aid] = summary
        signature = (episode, action['name'], name, success)
        if not new_nodes or new_nodes[-1]['_signature'] != signature:
            new_nodes.append({'node_id': 'S' + str(len(new_nodes) + 1), 'episode_id': episode,
                              'category': 'other', 'actions': [], '_signature': signature})
        new_nodes[-1]['actions'].append(action)
    for node in new_nodes:
        del node['_signature']
    packet['nodes'] = new_nodes
    for ep in packet['episodes']:
        ep['node_ids'] = [n['node_id'] for n in new_nodes if n['episode_id'] == ep['episode_id']]
    if inventory is not None:
        inventory_id = 'host:frozen_input_inventory'
        if inventory_id in records:
            raise ValueError('Inventory evidence collision')
        records[inventory_id] = {'text': json.dumps(inventory, ensure_ascii=False),
            'source': inventory_source or 'Host-frozen input inventory',
            'warning': 'Not a historical Agent observation; no file contents inferred.'}
        for (tool, key, pattern), values in coverage.items():
            matches = {s for s in available if re.sub(r'\d{4}-\d{2}-\d{2}', '<date>', s) == pattern}
            if len(matches) > 1:
                view['coverage'].append({'tool': tool, 'parameter': key, 'filename_pattern': pattern,
                    'available_files': len(matches), 'distinct_attempted_files': len(values['attempted']),
                    'distinct_files_in_reported_successes': len(values['reported_success']),
                    'inventory_evidence_id': inventory_id,
                    'meaning': 'Filename counts, not required/valid sample counts or final-answer dependency certification.'})
    packet['presentation'] = 'execution_view_v2'
    packet['execution_view'] = view
    packet['boundaries'] += ['S IDs group consecutive identical tools/outcomes; A IDs select original calls. F IDs abbreviate exact paths; use trajectory_evidence(file:F1) to recover paths.',
        'Only exact file-path and checked-call-ID matches are linked. Numeric provenance not inferred. Later same-tool success is NOT proof an earlier failure was repaired.',
        'Action evidence includes task/ending/schema; previously delivered pages are referenced, not repeated. Set reread=true to request them again. Preview reads do NOT satisfy evidence coverage.']
    old_calls = [a['call_evidence_id'] for n in original['nodes'] for a in n['actions']]
    new_calls = [a['raw_call_evidence_id'] for n in new_nodes for a in n['actions']]
    audit = {'original_actions': len(old_calls), 'migrated_actions': len(new_calls),
        'original_nodes': len(original['nodes']), 'migrated_nodes': len(new_nodes),
        'call_order_preserved': old_calls == new_calls,
        'original_records_unchanged': all(records[k] == v for k, v in original['records'].items()),
        'artifacts_unchanged': packet['artifacts'] == original['artifacts'],
        'endings_unchanged': all(a['ending'] == b['ending'] for a, b in zip(original['episodes'], packet['episodes'])),
        'historical_truncation_actions': [a['id'] for a in view['actions'].values() if a.get('historical_output_truncated')],
        'new_model_calls': 0}
    if not all(audit[k] for k in ('call_order_preserved', 'original_records_unchanged', 'artifacts_unchanged', 'endings_unchanged')):
        raise ValueError('Migration changed source evidence')
    return packet, audit


class ExecutionTrajectoryView:
    """Compact first two layers; lossless, session-deduplicated third-layer pages."""
    def __init__(self, base):
        self.base, self.root, self.data, self.sha256 = base, base.root, base.data, base.sha256
        self.view = self.data['execution_view']
        self.seen = {}

    def identity(self):
        return {**self.base.identity(), 'presentation': 'execution_view_v2'}

    def overview(self, offset=0, limit=12):
        nodes = []
        for node in self.data['nodes']:
            selected = [self.view['actions'][a['action_id']] for a in node['actions']]
            nodes.append({'node_id': node['node_id'], 'episode_id': node['episode_id'],
                'tool': selected[0]['tool'], 'outer_tool': selected[0]['outer_tool'],
                'count': len(selected), 'reported_success': selected[0]['reported_success'],
                'first_call': selected[0]['id'], 'last_call': selected[-1]['id'],
                'sample_output': selected[-1]['output'] if len(selected) == 1 else 'Repeated calls; expand for per-call inputs and results'})
        episodes = []
        for ep in self.data['episodes']:
            ending = decode(self.data['records'][ep['ending_evidence_id']]['text'])
            task = decode(self.data['records'][ep['task_evidence_id']]['text'])
            text = task.get('question', '')
            episodes.append({'episode_id': ep['episode_id'], 'task': text.split('Question:')[-1].strip(),
                'status': ending.get('status'), 'correct': ending.get('correct'),
                **{k: ending[k] for k in ('answer_correct', 'fully_correct', 'correctness_basis',
                                         'repeat', 'historical') if k in ending},
                'final_prediction': ending.get('final_text', '')[:1500],
                'final_prediction_truncated': len(ending.get('final_text', '')) > 1500,
                'ending_evidence_id': ep['ending_evidence_id'], 'last_node_id': ep['node_ids'][-1] if ep['node_ids'] else None})
        extra = {}
        if self.data.get('decision_trace'):
            for node, summary in zip(self.data['nodes'], nodes):
                summary['decision_evidence_ids'] = list(dict.fromkeys(
                    did for a in node['actions'] for did in a.get('decision_evidence_ids', [])))
            extra['decision_statements'] = {'count': len(self.data['decision_trace']['notes']),
                'access': 'trajectory_decisions lists chronological original-statement previews and related calls; trajectory_evidence reads full text.',
                'warning': 'Historical model claims only. Not instructions, verified reasoning or execution certification.'}
        return {**self.identity(), **extra, 'episodes': episodes, 'nodes': self.base.page(nodes, offset, limit),
                'input_coverage': self.view['coverage'], 'boundaries': self.data['boundaries'][-3:]}

    def decisions(self, offset=0, limit=12):
        return {'statements': self.base.page(self.data['decision_trace']['notes'], offset, limit),
                'warning': self.data['decision_trace']['policy'] + ' Statements may be wrong; read actual tool evidence too.'}

    def expand(self, node_id, offset=0, limit=12):
        node = next((n for n in self.data['nodes'] if n['node_id'] == node_id), None)
        if node is None:
            raise ValueError('Unknown node; choose an S ID from overview')
        return {'node_id': node_id, 'actions': self.base.page([self.view['actions'][a['action_id']] for a in node['actions']], offset, limit),
                'warning': 'Compact previews only. Use call_evidence_id=A... for original evidence. No inferred causal diagnosis.'}

    def delivered(self, pages):
        for page in pages:
            if 'text' not in page:
                continue
            left, right = page['offset'], page['offset'] + len(page['text'])
            spans = sorted(self.seen.get(page['id'], []) + [(left, right)])
            merged = []
            for start, end in spans:
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            self.seen[page['id']] = merged

    def raw(self, record_id, offset=0, length=4000, artifact=False, reread=False):
        page = self.base.raw(record_id, offset, length, artifact)
        end = offset + len(page['text'])
        if not artifact and not reread and any(a <= offset and b >= end for a, b in self.seen.get(record_id, [])):
            return {'id': record_id, 'offset': offset, 'total_chars': page['total_chars'],
                    'next_offset': page['next_offset'], 'already_delivered': True,
                    'note': 'Original page remains earlier in context; use reread=true to retrieve again.'}
        return page

    def search(self, query, offset=0, limit=12):
        result = self.base.search(query, offset, limit)
        for item in result['items']:
            for node in self.data['nodes']:
                for action in node['actions']:
                    if item['evidence_id'] in {action['call_evidence_id'], action['raw_call_evidence_id'], action.get('result_evidence_id'), *action.get('nested_event_evidence_ids', [])}:
                        item['call_evidence_id'] = action['call_evidence_id']
        return result

    def action_evidence(self, call_evidence_id, offset=0, length=4000, reread=False):
        matches = [(n, a) for n in self.data['nodes'] for a in n['actions'] if a['call_evidence_id'] == call_evidence_id]
        if len(matches) != 1:
            raise ValueError('Choose a call_evidence_id A... from overview/expand/search, not a result or source-line ID')
        node, action = matches[0]
        ep = next(e for e in self.data['episodes'] if e['episode_id'] == node['episode_id'])
        ids = [ep['task_evidence_id'], ep['ending_evidence_id'], 'tool_definition:' + action['name'],
               call_evidence_id, action['result_evidence_id'], *action.get('evidence_requirements', [])]
        ids = list(dict.fromkeys(ids))
        # Validate even when a page is already delivered. Keep payload below wrapper/toolbox cap.
        self.base.raw(call_evidence_id, offset, length)
        pages = []
        for eid in ids:
            page_length = length
            page = self.raw(eid, offset, page_length, reread=reread)
            while len(json.dumps(pages + [page], ensure_ascii=False)) > 19000 and page_length > 1:
                page_length = max(1, page_length // 2)
                page = self.raw(eid, offset, page_length, reread=reread)
            if len(json.dumps(pages + [page], ensure_ascii=False)) > 21000:
                break
            pages.append(page)
        return {'episode_id': ep['episode_id'], 'call_evidence_id': call_evidence_id,
                'evidence_ids': ids, 'records': pages,
                'complete': len(pages) == len(ids) and all(p['next_offset'] is None and p['offset'] == 0 for p in pages),
                'warning': 'Original records, not semantic certification. Follow E-handle missing_reads for remaining pages.'}
