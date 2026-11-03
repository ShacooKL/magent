"""Start a loopback-only, offline, text-only Qwen3.5-9B vLLM service."""
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    gpu, port = '4', 8714
    python = '/srv/user/yangxiao/anaconda3/envs/qwen35_vllm/bin/python'
    model = '/srv/user/yangxiao/linzw/models/Qwen'
    root = ROOT / 'var/services/qwen35-9b-02'
    # Fail before creating a service if the selected GPU/port is already occupied.
    used = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if int(used.strip()) > 1024:
        raise RuntimeError('Selected GPU is no longer idle; no process was stopped')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    root.mkdir(parents=True, exist_ok=False)
    args = [python, '-m', 'vllm.entrypoints.openai.api_server', '--model', model,
            '--served-model-name', 'Qwen3.5-9B-local', '--host', '127.0.0.1', '--port', str(port),
            '--tensor-parallel-size', '1', '--max-model-len', '262144', '--max-num-seqs', '16',
            '--gpu-memory-utilization', '0.85', '--language-model-only',
            '--reasoning-parser', 'qwen3', '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
            '--enable-prefix-caching', '--default-chat-template-kwargs', '{"enable_thinking":false}']
    overrides = {'CUDA_VISIBLE_DEVICES':gpu, 'PATH':str(Path(python).parent)+':'+os.environ.get('PATH','/usr/bin:/bin'), 'PYTHONNOUSERSITE':'1', 'HF_HUB_OFFLINE':'1',
                 'TRANSFORMERS_OFFLINE':'1', 'VLLM_CACHE_ROOT':str(root/'vllm-cache'),
                 'TORCHINDUCTOR_CACHE_DIR':str(root/'torch-cache'), 'TRITON_CACHE_DIR':str(root/'triton-cache')}
    with (root/'server.log').open('w') as log:
        proc = subprocess.Popen(args, env={**os.environ, **overrides}, cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (root/'service.json').write_text(json.dumps({'pid':proc.pid,'command':args,'environment':overrides,
         'base_url':f'http://127.0.0.1:{port}/v1','model':'Qwen3.5-9B-local'}, indent=2))
    print(json.dumps({'pid':proc.pid,'log':str(root/'server.log')}))


if __name__ == '__main__':
    main()
