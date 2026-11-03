"""Local API/tool-protocol smoke tests, independent of scored task runs."""
import json
from pathlib import Path
import sys
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'baselines/earthbench-noskill-v2-qwen9b'
sys.path[:0] = [str(BUNDLE/'src'),str(BUNDLE/'vendor')]
from magents.adapters import api_settings, client_for


def main():
    api_path = BUNDLE/'configs/api.json'
    api = api_settings(api_path,None)
    with build_opener(ProxyHandler({})).open(api['base_url']+'/models',timeout=10) as handle:
        models = json.load(handle)
    assert any(x['id']==api['model'] for x in models['data'])
    rows=[json.loads(x) for x in (ROOT/'var/runs/baseline-v2-image-preflight-01/events.jsonl').read_text().splitlines()]
    catalog=next(x['definitions'] for x in rows if x['kind']=='tool_catalog')
    client=client_for(api_path,api)
    root=ROOT/'var/runs/local-qwen-protocol-probe-01'
    root.mkdir(parents=True,exist_ok=False)
    results=[]
    for label,definitions in [('single_tool',[t for t in catalog if t['name']=='tool_inspect']),('full_catalog',catalog)]:
        response=client.create_message('Follow the requested diagnostic action.',
            [{'role':'user','content':'Call tool_inspect with name calculate_mean_lst_by_ndvi. Do not calculate anything.'}],
            definitions,max_tokens=512)
        calls=[b for b in response['content'] if b['type']=='tool_use']
        ok=len(calls)==1 and calls[0]['name']=='tool_inspect' and calls[0]['input']=={'name':'calculate_mean_lst_by_ndvi'}
        results.append({'case':label,'passed':ok,'response':response})
        (root/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
        print(json.dumps({'case':label,'passed':ok,'usage':response.get('usage')}),flush=True)
    if not all(r['passed'] for r in results):
        raise SystemExit('Protocol check failed; do not start batch')


if __name__=='__main__':
    main()
