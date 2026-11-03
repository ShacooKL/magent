"""Small handoff contracts; shape/reference checks are not truth checks."""
import json
import re
from copy import deepcopy

def tool(name, description, properties):
    return {"name": name, "description": description, "input_schema": {
        "type": "object", "properties": properties, "required": list(properties),
        "additionalProperties": False}}


TEXT = {"type": "string"}
TEXTS = {"type": "array", "items": TEXT}
IDS = {"type": "array", "items": {"type": "integer"}}
VERIFICATION_ITEM = tool('candidate', '', {
    'candidate': TEXT,
    'verdict': {'type': 'string', 'enum': ['supported', 'refuted', 'unresolved', 'untested']},
    'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
    'source_refs': TEXTS, 'test_refs': {'type': 'array', 'items': {
        'type': 'string', 'description': 'Local test ID such as h1:t1, or learning:event-id. The runner resolves test IDs automatically within this run.'}},
    'observations': TEXT, 'control_audit': TEXT, 'prediction_audit': TEXT,
    'scope': TEXT, 'not_established': TEXT, 'remaining_objections': TEXTS,
})['input_schema']
SUBMISSIONS = {
    'submit_verification': tool('submit_verification',
        'Submit per-candidate verification audit, never a task answer. Status completed means audit finished, not all candidates proved. Cite source events and actual local hypothesis_test events. No answer letters or option matching.', {
        'status': {'type': 'string', 'enum': ['completed', 'incomplete']},
        'candidates': {'type': 'array', 'items': VERIFICATION_ITEM, 'minItems': 1, 'maxItems': 3},
        'reason': TEXT}),
    'submit_skill': tool('submit_skill', 'Submit one advisory candidate, never a validated or published skill. Abstain if evidence is insufficient.', {
        'status': {'type': 'string', 'enum': ['candidate', 'abstain']},
        'title': TEXT, 'when': TEXT, 'steps': TEXTS, 'boundaries': TEXTS,
        'verified_findings': TEXTS, 'unverified_claims': TEXTS,
        'evidence_refs': TEXTS, 'reason': TEXT}),
    "submit_learning": tool("submit_learning", "Submit the learning diagnosis, not a task answer. Copy qualified evidence_refs from tool results, e.g. source-1:100, learning:4, round-1/with-skill:12. Bare numeric IDs are not accepted.", {
        "status": {"type": "string", "enum": ["completed", "incomplete"]},
        "answer": TEXT, "reason": TEXT,
        "evidence_refs": {"type": "array", "items": {"type": "string", "description": "Existing run-name:event-id reference; never guess its namespace."}}}),
    "submit_plan": tool("submit_plan", "Finish planning and delegate to executor. No calculation or evidence of completed work is required.", {
        "plan": TEXT}),
    "submit_progress": tool("submit_progress", "Hand off the current cumulative execution progress. This ends the slice, not necessarily the task.", {
        "status": {"type": "string", "enum": ["complete", "partial"]},
        "completed": TEXTS, "pending": TEXTS, "issues": TEXTS,
        "artifacts": TEXTS, "evidence_ids": IDS, "next_step": TEXT}),
    "submit_review": tool("submit_review", "Submit checks of existing execution evidence, not a new solution plan.", {
        "verdict": {"type": "string", "enum": ["PASS", "ISSUES", "INSUFFICIENT"]},
        "checked": TEXTS, "issues": TEXTS, "unchecked": TEXTS, "evidence_ids": IDS}),
    "submit_final": tool("submit_final", "Submit a supported answer or explicitly report an incomplete task. Ordinary text is not submission.", {
        "status": {"type": "string", "enum": ["completed", "incomplete"]},
        "answer": TEXT, "reason": TEXT, "evidence_ids": IDS}),
}


def validate(name, value, store, evidence_resolver=None):
    if name == 'submit_verification':
        return validate_verification(value, store, evidence_resolver)
    schema = SUBMISSIONS[name]["input_schema"]
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        raise ValueError("Required fields: " + ", ".join(schema["required"]))
    for key, spec in schema["properties"].items():
        item = value[key]
        if spec["type"] == "string" and not isinstance(item, str):
            raise ValueError(f"{key} must be text")
        if spec["type"] == "array":
            expected = int if key == "evidence_ids" else str
            if not isinstance(item, list) or any(type(x) is not expected for x in item):
                raise ValueError(f"Invalid {key}")
        if "enum" in spec and item not in spec["enum"]:
            raise ValueError(f"Invalid {key}")
    for event_id in value.get("evidence_ids", []):
        store.read(event_id, limit=1)
    if name in {'submit_learning', 'submit_skill'}:
        if evidence_resolver is None:
            raise ValueError('Learning evidence resolver is unavailable')
        for ref in value['evidence_refs']:
            evidence_resolver(ref, store)
    if name == 'submit_skill':
        if value['status'] == 'candidate' and not all((value['title'].strip(), value['when'].strip(),
                value['steps'], value['boundaries'], value['evidence_refs'])):
            raise ValueError('Candidate needs title, applicability, steps, boundaries and evidence')
        if not value['reason'].strip():
            raise ValueError('Explain candidate justification or abstention')
    if name == 'submit_learning':
        if value['status'] == 'completed' and (not value['answer'].strip() or not value['evidence_refs']):
            raise ValueError('Completed learning needs a diagnosis and qualified evidence_refs')
        if value['status'] == 'incomplete' and not value['reason'].strip():
            raise ValueError('Incomplete learning needs a reason')
    if name == "submit_plan" and not value["plan"].strip():
        raise ValueError("Plan must not be empty")
    if name == "submit_progress":
        if value["status"] == "complete" and value["pending"]:
            raise ValueError("Complete progress cannot contain pending work")
        if value["completed"] and not value["evidence_ids"]:
            raise ValueError("Completed operations need evidence IDs")
    if name == "submit_review" and value["verdict"] == "PASS":
        if not value["checked"] or not value["evidence_ids"] or value["issues"] or value["unchecked"]:
            raise ValueError("PASS needs evidenced checks and no unresolved issues/unchecked items")
    if name == "submit_final":
        if value["status"] == "completed" and (not value["answer"].strip() or not value["evidence_ids"]):
            raise ValueError("Completed answer needs text and evidence IDs")
        if value["status"] == "incomplete" and not value["reason"].strip():
            raise ValueError("Incomplete submission needs a reason")


def validate_verification(value, store, resolver):
    if not isinstance(value, dict) or set(value) != {'status', 'candidates', 'reason'}:
        raise ValueError('Required fields: status, candidates, reason; no task answer')
    if value['status'] not in ('completed', 'incomplete') or not isinstance(value['reason'], str) or not value['reason'].strip():
        raise ValueError('Audit needs status and nonempty reason')
    candidates = value['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError('Submit 1 to 3 candidate audits')
    if resolver is None:
        raise ValueError('Verification evidence resolver unavailable')
    # Normalize a copy: raw model payloads/context remain an exact audit trail.
    value = deepcopy(value)
    candidates = value['candidates']
    with (store.root / 'events.jsonl').open() as stream:
        events = {row['id']: row for row in map(json.loads, stream)}
    test_index = {}
    for event_id, event in events.items():
        if (event.get('kind') != 'tool_result' or event.get('call', {}).get('name') != 'hypothesis_test'
                or event.get('dispatch_error')):
            continue
        try:
            result = json.loads(event.get('output', ''))
        except (ValueError, TypeError):
            continue
        if isinstance(result, dict) and isinstance(result.get('test_id'), str):
            test_index.setdefault(result['test_id'], []).append(event_id)
    for item in candidates:
        if not isinstance(item, dict) or set(item) != set(VERIFICATION_ITEM['required']):
            raise ValueError('Each candidate needs: ' + ', '.join(VERIFICATION_ITEM['required']))
        for key, spec in VERIFICATION_ITEM['properties'].items():
            field = item[key]
            if spec['type'] == 'string' and (not isinstance(field, str) or not field.strip()):
                raise ValueError(f'{key} must be nonempty text')
            if spec['type'] == 'array' and (not isinstance(field, list) or any(not isinstance(x, str) or not x.strip() for x in field)):
                raise ValueError(f'{key} must be a list of nonempty strings')
            if 'enum' in spec and field not in spec['enum']:
                raise ValueError(f'Invalid {key}')
        if not item['source_refs']:
            raise ValueError('Each candidate needs original source evidence')
        for ref in item['source_refs']:
            if not ref.startswith('source-'):
                raise ValueError('source_refs must reference original source events')
            resolver(ref, store)
        normalized_refs = []
        for ref in item['test_refs']:
            if re.fullmatch(r'h[1-9][0-9]*:t[1-9][0-9]*', ref):
                matches = test_index.get(ref, [])
                if len(matches) != 1:
                    raise ValueError(f'Test ID {ref} has {len(matches)} matches in this run; use a unique existing local test or learning:event reference')
                ref = f'learning:{matches[0]}'
            if not ref.startswith('learning:'):
                raise ValueError('test_refs accepts local test IDs (h1:t1) or learning:event references')
            resolver(ref, store)
            event_id = int(ref.split(':', 1)[1])
            event = events.get(event_id, {})
            if event.get('kind') != 'tool_result' or event.get('call', {}).get('name') != 'hypothesis_test':
                raise ValueError('test_refs must identify hypothesis_test tool results, not proposals or prose')
            normalized_refs.append(ref)
        item['test_refs'] = normalized_refs
        if item['verdict'] in ('supported', 'refuted') and not item['test_refs']:
            raise ValueError('Supported/refuted candidates need actual test references')
    return value
