"""One advisory skill after diagnosis, then one fresh paired source-task replay."""
import json
from magents.learning import LearningTools


EXTRACTION_PROMPT = """You are now extracting one reusable remote-sensing tool-use skill
from your preceding diagnosis and its evidence. The diagnosis is fallible. Preserve
the existing context and distinguish established observations from speculative repairs.
Read trace/tool source or run small checks if needed; no mandatory hypothesis protocol.
Explain applicability, concrete actions and boundaries. Match actual tool interfaces
and statistical semantics; do not turn a workaround into an unsupported limitation.
No answer letters, task-specific numbers, dates, local paths or worked solutions in
title/when/steps/boundaries. Use submit_skill to return candidate or abstain. Preserve
uncertainty explicitly. No publication and no task replay by the model. The host will
evaluate any candidate with fresh paired solvers, not certify it automatically.
"""


def extract_and_replay(team, *, task_id, runner_script, profile,
                       replay_max_requests=40, official_only=True):
    store = team.store
    if not store.state.get('learning_submission'):
        raise ValueError('Skill extraction requires an actual diagnosis submission')
    # Eight additional requests, not a restarted diagnosis budget.
    team.config = {**team.config, 'max_requests': store.state['requests'] + 8}
    team.prompts['learner'] = EXTRACTION_PROMPT
    store.state.update(status='extracting_skill', pipeline_stage='extract_skill')
    store.event('skill_extraction_start', max_additional_requests=8, prompt=EXTRACTION_PROMPT)
    store.save()
    team.ask('learner', 'Continue from the diagnosis above. Extract at most one concise '
             'advisory skill, or abstain if no reusable finding is sufficiently supported. '
             'Submit using submit_skill within 8 requests; do not solve the original task.',
             submission='submit_skill', request_limit=8, submission_only_at_end=True)
    candidate = store.state['skill_submission']
    (store.root / 'skill-candidate.json').write_text(json.dumps(candidate, ensure_ascii=False, indent=2))
    if candidate['status'] == 'abstain':
        store.state.update(status='skill_abstained', pipeline_stage='done')
        store.save()
        return
    # Only reusable content goes to solvers, not diagnosis/evidence/host scoring.
    skill = json.dumps({key: candidate[key] for key in ('title', 'when', 'steps', 'boundaries')}, ensure_ascii=False)
    tools = team.tools
    replay = LearningTools(tools.base, tools.sources, store.root, task_id, runner_script,
                           replay_profile=profile, replay_max_requests=replay_max_requests,
                           replay_official_only=official_only)
    store.state.update(status='replaying_skill', pipeline_stage='paired_replay')
    store.event('paired_replay_start', profile=profile, max_requests_each=replay_max_requests,
                official_only=official_only, note='Fresh workspaces; no retrieval; one pair; no automatic promotion')
    store.save()
    result = json.loads(replay.execute('replay_task', {'skill': skill}))
    store.state.update(status='skill_replay_finished_pending_audit', pipeline_stage='done',
                       skill_replay=result)
    store.save()
