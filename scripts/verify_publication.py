"""Verify public/default-branch access and anonymous remote bytes after a push."""
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import requests

ROOT=Path(__file__).resolve().parents[1]
REPO='qinyishcn/ttft-inference-lab'
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
for path in tracked:
    assert not re.search(r'\.(gguf|dll|zip)$|^runtime/|^models/|^results/private/|squad-dev-v1.1.json|server-upstream.md',path),path
    if path.endswith('.png'):
        continue
    body=(ROOT/path).read_text(encoding='utf-8-sig',errors='replace')
    assert not re.search(r'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-proj-[A-Za-z0-9_-]{25,}',body),'Credential-shaped content in '+path
    assert not re.search(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]',body),'Absolute user path in '+path

def anonymous_get(url):
    session=requests.Session()
    session.trust_env=False  # Disable netrc/ambient account authentication and proxies.
    response=session.get(url,timeout=(15,45))
    assert 'Authorization' not in response.request.headers
    response.raise_for_status()
    return response

metadata=anonymous_get('https://api.github.com/repos/'+REPO).json()
assert metadata['private'] is False and metadata['default_branch']=='main'
remote=anonymous_get('https://api.github.com/repos/'+REPO+'/commits/main').json()['sha']
assert remote==head,(remote,head)
paths=['README.md','docs/results-zh.md','data/artifacts.json','data/environment.json',
       'results/requests.jsonl','results/compute_only/summary.csv','results/compute_only/validation.json',
       'results/compute_only/figures/minicpm_native.png','results/minicpm_native/requests.jsonl',
       'results/history/provenance.json','data/minicpm-artifacts.json']
def verify(path):
    expected=subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
    response=anonymous_get(f'https://raw.githubusercontent.com/{REPO}/{head}/{path}')
    assert response.content==expected,'Remote byte mismatch: '+path
    return {'path':path,'bytes':len(expected),'sha256':hashlib.sha256(expected).hexdigest(),'http_status':response.status_code}
with cf.ThreadPoolExecutor(max_workers=4) as pool:
    files=list(pool.map(verify,paths))
# Separately verify the human-facing default-branch URLs, not only commit URLs.
for path in ['README.md','docs/results-zh.md']:
    response=anonymous_get(f'https://raw.githubusercontent.com/{REPO}/main/{path}?verification={head}')
    assert response.content==subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)
result={'verified_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'repository':metadata['html_url'],
        'private':False,'default_branch':'main','verified_content_commit':head,
        'unauthenticated_http':True,'remote_main_matches_local':True,'tracked_files':len(tracked),
        'excluded_artifact_and_credential_scan_passed':True,'files':files,
        'note':'A subsequent verification-record-only commit preserves these verified content bytes.'}
(ROOT/'results'/'publication-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
