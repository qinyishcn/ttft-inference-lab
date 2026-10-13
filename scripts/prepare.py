"""Download pinned, public upstream artifacts; never publish weights or binaries."""
import concurrent.futures as cf
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import requests
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ('LiquidAI/LFM2-350M-GGUF', '8fdc9d526b7ed346b19257551b05816c7912ecc2', ['LFM2-350M-Q4_K_M.gguf', 'LFM2-350M-Q8_0.gguf', 'LFM2-350M-F16.gguf']),
    ('LiquidAI/LFM2-1.2B-GGUF', '5399e76c648f4eb8c053feb1ab747277dea5bf8b', ['LFM2-1.2B-Q4_K_M.gguf']),
    ('ggml-org/SmolVLM2-500M-Video-Instruct-GGUF', 'ccd7aae53bcb1997355c2f094959e72b3642ce17', ['SmolVLM2-500M-Video-Instruct-Q8_0.gguf', 'mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf']),
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
        old_parts = path.with_suffix(path.suffix + '.parts')
        parts = path.with_suffix(path.suffix + '.1m.parts')
        parts.mkdir(exist_ok=True)
        chunk = 1024 * 1024
        if old_parts.exists():
            for old in old_parts.iterdir():
                raw=old.read_bytes()
                for k in range(0,len(raw),chunk):
                    (parts/str(int(old.name)+k)).write_bytes(raw[k:k+chunk])
        def get_part(start):
            end = min(start + chunk, size) - 1
            part = parts / str(start)
            if part.exists() and part.stat().st_size == end-start+1:
                return part
            for attempt in range(4):
                try:
                    r = requests.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=(20,40))
                    r.raise_for_status()
                    assert r.status_code == 206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/'), r.headers
                    assert len(r.content) == end-start+1
                    part.write_bytes(r.content)
                    return part
                except Exception as exc:
                    print('retry',path.name,start,type(exc).__name__,str(exc)[:100],flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        with cf.ThreadPoolExecutor(max_workers=32) as pool:
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
    spec=importlib.util.find_spec('torch')
    torchlib=Path(spec.submodule_search_locations[0])/'lib' if spec else None
    dlls=['cublas64_12.dll','cublasLt64_12.dll','cudart64_12.dll']
    use_torch=bool(torchlib and all((torchlib/n).exists() for n in dlls) and '+cu124' in importlib.metadata.version('torch'))
    archives=['llama-b11577-bin-win-cuda-12.4-x64.zip']
    if not use_torch:
        archives.append('cudart-llama-bin-win-cuda-12.4-x64.zip')
    for name in archives:
        jobs.append((f'https://github.com/ggml-org/llama.cpp/releases/download/b11577/{name}',ROOT/'runtime'/name))
    for repo, revision, files in SOURCES:
        response=requests.get('https://huggingface.co/api/models/'+repo+'/revision/'+revision+'?blobs=true', timeout=60)
        response.raise_for_status()
        info=response.json()
        assert info['sha']==revision
        for name in files:
            url=f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{name}'
            path=ROOT/'models'/name
            jobs.append((url,path))
            meta=next(x for x in info['siblings'] if x['rfilename']==name)
            manifest['models'].append({'repo':repo,'revision':info['sha'],'file':name,'url':url,'upstream_sha256':meta['lfs']['sha256']})
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda job:download(*job),jobs))
    for item in manifest['models']:
        p=ROOT/'models'/item['file']
        item['bytes']=p.stat().st_size
        item['sha256']=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        assert item['sha256']==item['upstream_sha256'],item['file']
    runtime=[]
    for name in archives:
        p=ROOT/'runtime'/name
        runtime.append({'file':p.name,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()})
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            z.extractall(ROOT/'runtime'/'cuda')
    manifest['runtime_artifacts']=runtime
    if use_torch:
        dependencies=[]
        for name in dlls:
            source=torchlib/name
            destination=ROOT/'runtime'/'cuda'/name
            shutil.copy2(source,destination)
            dependencies.append({'file':name,'bytes':source.stat().st_size,'sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest()})
        manifest['cuda_dependencies']={'source':'PyTorch wheel','torch_version':importlib.metadata.version('torch'),'install_index':'https://download.pytorch.org/whl/cu124','files':dependencies}
    (ROOT/'data'/'artifacts.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    dataset=ROOT/'data'/'squad-dev-v1.1.json'
    if not dataset.exists():
        r=requests.get('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',timeout=60)
        r.raise_for_status()
        dataset.write_bytes(r.content)
    print('preparation complete',flush=True)

if __name__ == '__main__':
    main()
