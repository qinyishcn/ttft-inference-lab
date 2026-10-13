"""Paired native VLM comparisons following the MiniCPM-o computation study."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from benchmark import ROOT, SEED

p=argparse.ArgumentParser()
p.add_argument('--assets',type=Path,required=True)
p.add_argument('--pilot',action='store_true')
p.add_argument('--models',nargs='+',choices=['qwen25-vl3b','qwen2-vl2b','minicpm-v4'])
a=p.parse_args()
manifest=json.loads((ROOT/'data'/'vlm-extension-artifacts.json').read_text())
cases=[]
for source in manifest['sources']:
    if a.models and source['key'] not in a.models:
        continue
    for kind,frames,dataset,style,resolution,strict in [
        ('single',1,'streamingbench','standard',336,False),
        ('four',4,'streamingbench','standard',336,False),
        ('color',1,'synthetic','compact',224,True)]:
        cases.append((source,kind,frames,dataset,style,resolution,strict))
for block in range(1 if a.pilot else 2):
    order=cases.copy()
    random.Random(SEED+block).shuffle(order)
    for source,kind,frames,dataset,style,resolution,strict in order:
        tag=source['key']+'-'+kind
        raw=ROOT/'results'/'vlm_extension'/'requests.jsonl'
        existing=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines()] if raw.exists() else []
        completed=[r for r in existing if r['config']['quantization']==tag and r['phase']==('pilot' if a.pilot else 'measured') and r['block']==block]
        if completed:
            assert len(completed)==(8 if a.pilot else 24),'Partial block requires audit'
            continue
        model=next(x['file'] for x in source['files'] if not x['file'].startswith('mmproj'))
        projector=next(x['file'] for x in source['files'] if x['file'].startswith('mmproj'))
        cmd=[sys.executable,str(ROOT/'scripts'/'minicpm_native.py'),'--assets',str(a.assets),
             '--model',str(ROOT/'models'/model),'--projector',str(ROOT/'models'/projector),
             '--model-label',source['model'],'--output-group','vlm_extension','--tag',tag,
             '--n','8' if a.pilot else '24','--block',str(block),'--resolutions',str(resolution),
             '--batch','2048','--ubatch','512','--threads','4','--frames',str(frames),
             '--prompt-style',style,'--dataset',dataset]
        if strict:
            cmd.append('--strict-prompt')
        if a.pilot:
            cmd.append('--pilot')
        print('BLOCK',block,tag,flush=True)
        subprocess.run(cmd,check=True)
        if not a.pilot:
            subprocess.run([sys.executable,str(ROOT/'scripts'/'report_vlm_extension.py')],check=True)
