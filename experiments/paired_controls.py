"""Paired permission or skill-retrieval controls; no retries or solver changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/srv/user/yangxiao/linzw/Agent/datasets/Earth-Bench')


def job_command(job, root, inputs):
    command = [sys.executable, '-u', str(ROOT / 'experiments/earthbench_q1.py'),
               '--task-id', str(job['task']), '--output', str(root / job['directory'])]
    if job['condition'] != 'with-skill':
        command.append('--no-skill-retrieval')
    if job['condition'] != 'auxiliary':
        command.append('--official-only')
    if job['task'] == 9:
        command.extend(['--data-dir', str(inputs)])
    return command


def summarize(root):
    manifest = json.loads((root / 'batch.json').read_text())
    rows = []
    for job in manifest['jobs']:
        directory = root / job['directory']
        state = json.loads((directory / 'state.json').read_text()) if (directory / 'state.json').exists() else {}
        scores, counts = [], {}
        if (directory / 'events.jsonl').exists():
            for line in (directory / 'events.jsonl').read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A live append may not yet be complete.
                if event['kind'] in ('host_score', 'host_score_correction'):
                    scores.append(event)
                if event['kind'] == 'tool_result':
                    name = event['call']['name']
                    counts[name] = counts.get(name, 0) + 1
        rows.append({**job, 'status': state.get('status', 'queued'),
                     'requests': state.get('requests', 0), 'answer': state.get('answer', ''),
                     'correct': scores[-1]['correct'] if scores else None,
                     'billing': state.get('billing'), 'tool_calls': counts})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--tasks', type=int, nargs='+', default=list(range(4, 11)))
    parser.add_argument('--comparison', choices=['permissions', 'skill-retrieval'], default='permissions')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.status:
        print(json.dumps(summarize(root), ensure_ascii=False, indent=2))
        return
    if not 1 <= args.workers <= 14:
        parser.error('workers must be between 1 and 14')
    if any(task < 1 for task in args.tasks) or len(set(args.tasks)) != len(args.tasks):
        parser.error('tasks must be unique positive IDs')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs/question9'
    excluded = []
    if 9 in args.tasks:
        inputs.mkdir(parents=True)
        for path in sorted((DATA / 'question9').iterdir()):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in {'.tif', '.tiff'}:
                shutil.copy2(path, inputs / path.name)
            else:
                excluded.append(path.name)
    conditions = ('auxiliary', 'official-only') if args.comparison == 'permissions' else ('with-skill', 'without-skill')
    jobs = [{'task': task, 'condition': condition, 'directory': f'q{task}-{condition}'}
            for task in args.tasks for condition in conditions]
    (root / 'batch.json').write_text(json.dumps({
        'created': datetime.now(timezone.utc).isoformat(), 'workers': args.workers,
        'jobs': jobs, 'q9_excluded': excluded, 'repeats': 1, 'comparison': args.comparison,
        'config_snapshot': json.loads((ROOT / 'configs/default.json').read_text()),
        'library_snapshot': [json.loads(p.read_text()) for p in sorted((ROOT / 'library/skills').glob('*.json'))] if args.comparison == 'skill-retrieval' else [],
        'note': 'Independent workspaces; unchanged model/default budgets; no automatic retries. '
                'Official-only is a permission-and-instruction treatment, not guaranteed semantic compliance.'}, indent=2))

    def run(job):
        command = job_command(job, root, inputs)
        print('START ' + job['directory'], flush=True)
        with (root / (job['directory'] + '.log')).open('w') as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        print(f"END {job['directory']} exit={result.returncode}", flush=True)
        return {'job': job['directory'], 'exit_code': result.returncode}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            outcome = future.result()
            with (root / 'completed.jsonl').open('a') as log:
                log.write(json.dumps(outcome) + '\n')
            (root / 'summary.json').write_text(json.dumps(summarize(root), ensure_ascii=False, indent=2))
    print('FINISHED ' + str(root), flush=True)


if __name__ == '__main__':
    main()
