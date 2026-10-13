"""Download pinned, public upstream artifacts; never publish weights or binaries."""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import requests
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ('LiquidAI/LFM2-350M-GGUF', ['LFM2-350M-Q4_K_M.gguf', 'LFM2-350M-Q8_0.gguf', 'LFM2-350M-F16.gguf']),
    ('LiquidAI/LFM2-1.2B-GGUF', ['LFM2-1.2B-Q4_K_M.gguf']),
    ('ggml-org/SmolVLM2-500M-Video-Instruct-GGUF', ['SmolVLM2-500M-Video-Instruct-Q8_0.gguf', 'mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf']),
]

def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Independent byte ranges allow restart and protect against truncated transfers.
    head = requests.get(url, headers={'Range':'bytes=0-0'}, stream=True, timeout=60)
    if head.status_code == 206:
        size = int(head.headers['Content-Range'].split('/')[-1])
        head.close()
        parts = path.with_suffix(path.suffix + '.parts')
        parts.mkdir(exist_ok=True)
        chunk = 8 * 1024 * 1024
        def get_part(start):
            end = min(start + chunk, size) - 1
            part = parts / str(start)
            if part.exists() and part.stat().st_size == end-start+1:
                return part
            for attempt in range(4):
                try:
                    r = requests.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=120)
                    r.raise_for_status()
                    assert r.status_code == 206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'), r.headers
                    assert len(r.content) == end-start+1
                    part.write_bytes(r.content)
                    return part
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        with cf.ThreadPoolExecutor(max_workers=6) as pool:
            ordered = list(pool.map(get_part, range(0, size, chunk)))
        with path.with_suffix(path.suffix+'.partial').open('wb') as out:
            for part in ordered:
                out.write(part.read_bytes())
        path.with_suffix(path.suffix+'.partial').replace(path)
    else:
        head.raise_for_status()
        with path.with_suffix(path.suffix+'.partial').open('wb') as out:
            for b in head.iter_content(1024*1024):
                out.write(b)
        head.close()
        path.with_suffix(path.suffix+'.partial').replace(path)
    print('downloaded', path.name, path.stat().st_size, flush=True)

def main():
    manifest = {'runtime_tag':'b11577', 'models':[]}
    jobs=[]
    for repo, files in SOURCES:
        info=requests.get('https://huggingface.co/api/models/'+repo, timeout=60).json()
        for name in files:
            url=f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{name}'
            path=ROOT/'models'/name
            jobs.append((url,path))
            manifest['models'].append({'repo':repo,'revision':info['sha'],'file':name,'url':url})
    for name in ['llama-b11577-bin-win-cuda-12.4-x64.zip','cudart-llama-bin-win-cuda-12.4-x64.zip']:
        jobs.append((f'https://github.com/ggml-org/llama.cpp/releases/download/b11577/{name}',ROOT/'runtime'/name))
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda job:download(*job),jobs))
    for item in manifest['models']:
        p=ROOT/'models'/item['file']
        item['bytes']=p.stat().st_size
        item['sha256']=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    runtime=[]
    for p in (ROOT/'runtime').glob('*.zip'):
        runtime.append({'file':p.name,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
        with zipfile.ZipFile(p) as z:
            z.extractall(ROOT/'runtime'/'cuda')
    manifest['runtime_artifacts']=runtime
    (ROOT/'data'/'artifacts.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('preparation complete',flush=True)

if __name__ == '__main__':
    main()
