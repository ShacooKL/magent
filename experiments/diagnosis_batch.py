"""Fixed diagnosis comparison; isolated runs, no retries and no audit-note input."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def make_jobs(manifest):
    return [dict(case_id=case['id'], task_id=case['task_id'], source_run=case['source_run'],
                 mode=mode, repetition=repeat + 1,
                 directory=f"{case['id']}-{mode}-r{repeat + 1}")
            for repeat in range(manifest['repetitions_per_case_mode'])
            for case in manifest['cases']
            for mode in (manifest['modes'] if repeat % 2 == 0 else list(reversed(manifest['modes'])))]


def command_for(job, root, manifest):
    return [sys.executable, '-u', str(ROOT / 'experiments/learn_q10.py'),
            '--task-id', str(job['task_id']), '--source-run', str(ROOT / job['source_run']),
            '--diagnosis-only', '--simple-diagnosis', job['mode'],
            '--max-requests', str(manifest['max_requests_per_run']),
            '--profile', manifest['profile'], '--output', str(root / job['directory'])]


def summarize(root):
    batch = json.loads((root / 'batch.json').read_text())
    finished = {}
    if (root / 'completed.jsonl').exists():
        for line in (root / 'completed.jsonl').read_text().splitlines():
            row = json.loads(line)
            finished[row['job']] = row
    rows = []
    for job in batch['jobs']:
        path = root / job['directory'] / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        rows.append({**job, 'status': state.get('status', 'queued'),
                     'phase': state.get('diagnosis_phase'), 'requests': state.get('requests', 0),
                     'billing': state.get('billing'), 'error': state.get('error'),
                     'process': finished.get(job['directory']),
                     'submission_status': (state.get('learning_submission') or {}).get('status')})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT / 'experiments/diagnosis-small-v1/cases.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.status:
        print(json.dumps(summarize(root), ensure_ascii=False, indent=2))
        return
    if not 1 <= args.workers <= 4:
        parser.error('workers must be 1 to 4')
    manifest = json.loads(args.manifest.read_text())
    jobs = make_jobs(manifest)
    if len(jobs) != manifest['total_planned_runs']:
        parser.error('Manifest total does not match jobs')
    for job in jobs:
        source = ROOT / job['source_run']
        assert str(json.loads((source / 'run.json').read_text())['task_id']) == str(job['task_id'])
        assert (source / 'events.jsonl').is_file() and (source / 'workspace').is_dir()
    if args.dry_run:
        print(json.dumps([command_for(job, root, manifest) for job in jobs], indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    (root / 'batch.json').write_text(json.dumps(dict(created=datetime.now(timezone.utc).isoformat(),
        manifest=manifest, jobs=jobs, workers=args.workers,
        note='No retries; no audit notes in model inputs. Repeats are independent API runs, not paired seeds.'), indent=2))

    def run(job):
        print('START ' + job['directory'], flush=True)
        with (root / (job['directory'] + '.log')).open('w') as log:
            result = subprocess.run(command_for(job, root, manifest), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return dict(job=job['directory'], exit_code=result.returncode, finished=datetime.now(timezone.utc).isoformat())

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            outcome = future.result()
            with (root / 'completed.jsonl').open('a') as log:
                log.write(json.dumps(outcome) + '\n')
            temporary = root / 'summary.tmp'
            temporary.write_text(json.dumps(summarize(root), ensure_ascii=False, indent=2))
            temporary.replace(root / 'summary.json')
            print('END ' + json.dumps(outcome), flush=True)
    print('FINISHED ' + str(root), flush=True)


if __name__ == '__main__':
    main()
