"""Pin small VLM GGUF sources and download verified weights outside Git."""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import requests
import time
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[
 ('qwen25-vl3b','Qwen2.5-VL-3B-Instruct','ggml-org/Qwen2.5-VL-3B-Instruct-GGUF',['Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf','mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf']),
 ('qwen2-vl2b','Qwen2-VL-2B-Instruct','ggml-org/Qwen2-VL-2B-Instruct-GGUF',['Qwen2-VL-2B-Instruct-Q4_K_M.gguf','mmproj-Qwen2-VL-2B-Instruct-f16.gguf']),
 ('minicpm-v4','MiniCPM-V-4','openbmb/MiniCPM-V-4-gguf',None),
]
def get(url,**kwargs):
    for attempt in range(4):
        try:
            with requests.Session() as session:
                session.trust_env=False
                r=session.get(url,timeout=(15,45),**kwargs)
                r.raise_for_status()
                return r
        except requests.RequestException:
            if attempt==3:
                raise
            time.sleep(1+attempt)
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
def download(item):
    path=ROOT/'models'/item['file']
    if path.exists() and digest(path)==item['upstream_sha256']:
        return
    parts=path.with_suffix('.gguf.vlm.parts')
    parts.mkdir(exist_ok=True)
    chunk=4*1024*1024
    size=item['bytes']
    def part(start):
        end=min(start+chunk,size)-1
        dest=parts/str(start)
        if dest.exists() and dest.stat().st_size==end-start+1:
            return dest
        r=get(item.get('download_url',item['url']),headers={'Range':f'bytes={start}-{end}'})
        assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'),r.headers
        assert len(r.content)==end-start+1
        dest.write_bytes(r.content)
        return dest
    print('Downloading',item['file'],round(size/1e9,2),'GB',flush=True)
    with cf.ThreadPoolExecutor(max_workers=16) as pool:
        futures={pool.submit(part,start):start for start in range(0,size,chunk)}
        last=time.monotonic()
        for completed in cf.as_completed(futures):
            completed.result()
            if time.monotonic()-last>25:
                print(item['file'],len(list(parts.iterdir())),'/',len(futures),'parts',flush=True)
                last=time.monotonic()
        paths=[parts/str(start) for start in range(0,size,chunk)]
    temporary=path.with_suffix('.gguf.partial')
    with temporary.open('wb') as f:
        for partpath in paths:
            f.write(partpath.read_bytes())
    assert digest(temporary)==item['upstream_sha256'],item['file']
    temporary.replace(path)
    print('Verified',item['file'],flush=True)

def main():
    destination=ROOT/'data'/'vlm-extension-artifacts.json'
    if destination.exists():
        manifest=json.loads(destination.read_text())
        for entry in manifest['sources']:
            repo=entry['repo'].replace('openbmb/','OpenBMB/')
            for item in entry['files']:
                item['download_url']=f'https://modelscope.cn/models/{repo}/resolve/master/{item["file"]}'
                item['mirror_policy']='Mirror mutable URL; accept only exact pinned Hugging Face SHA256 and size.'
        def job(item):
            download(item)
            item['sha256']=digest(ROOT/'models'/item['file'])
        jobs=[item for entry in sorted(manifest['sources'],key=lambda x:x['key']!='minicpm-v4') for item in entry['files']]
        with cf.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(job,item) for item in jobs]
            for future in cf.as_completed(futures):
                future.result()
                destination.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        return
    manifest={'naming_note':'Qwen2.5-VL has a 3B model. The 2B comparator is Qwen2-VL-2B, a different generation, not Qwen2.5-VL-2B.','sources':[]}
    for key,label,repo,names in SOURCES:
        info=get('https://huggingface.co/api/models/'+repo+'?blobs=true').json()
        siblings=info['siblings']
        if names is None:
            names=[s['rfilename'] for s in siblings if 'q4_k_m' in s['rfilename'].lower()]
            projectors=[s['rfilename'] for s in siblings if 'mmproj' in s['rfilename'].lower() and 'f16' in s['rfilename'].lower()]
            assert len(names)==1 and len(projectors)==1,(names,projectors)
            names+=projectors
        items=[]
        for name in names:
            meta=next(s for s in siblings if s['rfilename']==name)
            items.append({'file':name,'bytes':meta['lfs']['size'],'upstream_sha256':meta['lfs']['sha256'],
                          'url':f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{name}'})
        entry={'key':key,'model':label,'repo':repo,'revision':info['sha'],'files':items}
        manifest['sources'].append(entry)
        print(key,info['sha'],[(x['file'],x['bytes']) for x in items],flush=True)
    destination.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for entry in manifest['sources']:
        for item in entry['files']:
            download(item)
            item['sha256']=digest(ROOT/'models'/item['file'])
        destination.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':
    main()
