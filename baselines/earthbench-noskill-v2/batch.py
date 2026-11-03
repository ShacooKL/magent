"""One attempt per task, isolated worker processes, shared Qwen cost ledger."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'vendor')]
from magents.adapters import api_settings
from rsagent.billing.qwen import budget_for
from rsagent.evolution.request_ledger import RequestLedger, RequestBudgetExceeded

ALLOWED = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}


def inventory(data_root):
    jobs = []
    for i in range(1, 249):
        p = data_root / f'question{i}'
        if not (ROOT / 'earthbench/tasks' / f'{i}.json').is_file():
            raise ValueError(f'Missing public task {i}')
        files = sorted(x for x in p.rglob('*') if x.is_file() or x.is_symlink()) if p.is_dir() else []
        images = [x for x in files if not x.is_symlink() and x.suffix.lower() in ALLOWED]
        omitted = [str(x.relative_to(p)) for x in files if x not in images]
        jobs.append({'task_id': i, 'source': str(p), 'images': len(images),
                     'excluded_files': omitted, 'status': 'ready' if images else 'missing_image_data'})
    return jobs


def report(root):
    batch = json.loads((root / 'batch.json').read_text())
    done = {}
    if (root / 'completed.jsonl').exists():
        done = {r['task_id']: r for r in (json.loads(s) for s in (root / 'completed.jsonl').read_text().splitlines())}
    rows = []
    for j in batch['jobs']:
        task = root / 'runs' / f"q{j['task_id']:03}"
        s = json.loads((task / 'state.json').read_text()) if (task / 'state.json').exists() else {}
        score = None
        if (task / 'events.jsonl').exists():
            for line in (task / 'events.jsonl').read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # live trailing record may still be being written
                if event['kind'] == 'host_score':
                    score = event
        rows.append({'task_id': j['task_id'], 'status': s.get('status', done.get(j['task_id'], {}).get('status', 'queued')),
                     'requests': s.get('requests', 0), 'correct': bool(score and score['correct']),
                     'scored': score is not None, 'error': s.get('error'), 'process': done.get(j['task_id'])})
    ledger = RequestLedger.summarize(json.loads((root / 'ledger/ledger.json').read_text()))
    correct = sum(r['correct'] for r in rows)
    finished = len(done)
    return {'total': 248, 'finished': finished, 'running': sum(bool(r['requests']) and not r['process'] for r in rows),
            'correct': correct, 'accuracy_all_248': correct / 248,
            'accuracy_finished': correct / finished if finished else None,
            'workflow_completed': sum(r['status'] == 'completed' for r in rows),
            'budget': ledger.get('budget'), 'api_requests': ledger['requests_total'] if 'requests_total' in ledger else 248 * 40 - ledger['remaining_requests'],
            'api_failures': ledger['api_failures'], 'rows': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/srv/user/yangxiao/linzw/Agent/datasets/Earth-Bench'))
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--budget-cny', type=str, default='100')
    p.add_argument('--status', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    root = a.output.resolve()
    if a.status:
        r = report(root)
        print(json.dumps({k:v for k,v in r.items() if k != 'rows'}, ensure_ascii=False, indent=2))
        return
    if not 1 <= a.workers <= 16:
        p.error('workers must be 1..16')
    jobs = inventory(a.data_root.resolve())
    if a.dry_run:
        print(json.dumps({'tasks':len(jobs),'ready':sum(j['status']=='ready' for j in jobs),
                          'filtered': [j for j in jobs if j['excluded_files']], 'workers':a.workers}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    (root / 'batch.json').write_text(json.dumps({'version':'earthbench-noskill-v2','created':datetime.now(timezone.utc).isoformat(),
        'pid':os.getpid(), 'workers':a.workers,'budget_cny':a.budget_cny,'jobs':jobs,
        'note':'One attempt per task; no automatic retries/resume. All 248 retained in denominator. No visual model input.'}, indent=2))
    api = api_settings(ROOT / 'configs/api.json', None)
    ledger = RequestLedger(root / 'ledger', max_requests=248*40, max_api_failures=50,
                           budget=budget_for(api, a.budget_cny))
    (root / 'logs').mkdir()
    (root / 'runs').mkdir()

    def run(j):
        i = j['task_id']
        if j['status'] != 'ready':
            return {'task_id':i,'status':j['status']}
        try:
            ledger.assert_can_start()
        except RequestBudgetExceeded as e:
            return {'task_id':i,'status':'not_started_budget','error':str(e)}
        data = Path(j['source'])
        if j['excluded_files']:
            staged = root / 'inputs' / f'question{i}'
            for source in data.rglob('*'):
                if source.is_file() and not source.is_symlink() and source.suffix.lower() in ALLOWED:
                    dest = staged / source.relative_to(data)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    os.link(source, dest)
            data = staged
        command = [sys.executable, '-u', str(ROOT / 'run.py'), '--task-id', str(i),
                   '--data-dir', str(data), '--output', str(root / 'runs' / f'q{i:03}'), '--ledger-root', str(root / 'ledger')]
        print(f'START Q{i}', flush=True)
        with (root / 'logs' / f'q{i:03}.log').open('w') as log:
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, 'PYTHONNOUSERSITE':'1', 'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}).returncode
        return {'task_id':i,'status':'exited','exit_code':code,'finished':datetime.now(timezone.utc).isoformat()}

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run,j):j['task_id'] for j in jobs}
        for f in as_completed(futures):
            try:
                outcome = f.result()
            except Exception as exc:
                outcome = {'task_id':futures[f],'status':'launcher_error','error':f'{type(exc).__name__}: {exc}'}
            with (root / 'completed.jsonl').open('a') as log:
                log.write(json.dumps(outcome)+'\n')
            r = report(root)
            tmp = root / 'progress.tmp'
            tmp.write_text(json.dumps(r, indent=2))
            tmp.replace(root / 'progress.json')
            print('END '+json.dumps(outcome), flush=True)
    print('FINISHED',flush=True)


if __name__ == '__main__':
    main()
