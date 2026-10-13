"""Two randomized, paired blocks; each condition gets 48 fresh full-model calls."""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from benchmark import ROOT, SEED

p=argparse.ArgumentParser()
p.add_argument('--assets',type=Path,required=True)
a=p.parse_args()
cases=[
    ('native-f16-448','f16',448,'standard',4,1,'streamingbench'),
    ('native-f16-336','f16',336,'standard',4,1,'streamingbench'),
    ('native-f16-compact','f16',336,'compact',4,1,'streamingbench'),
    ('native-q8-compact','q8',336,'compact',4,1,'streamingbench'),
    ('native-q4-compact','q4',336,'compact',4,1,'streamingbench'),
    ('native-q4-t8','q4',336,'compact',8,1,'streamingbench'),
    ('native-q8-color','q8',224,'compact',4,1,'synthetic'),
    ('native-q8-video4','q8',336,'compact',4,4,'streamingbench'),
    ('native-q8-color-strict','q8',224,'compact',4,1,'synthetic'),
]
for block in range(2):
    order=cases.copy()
    random.Random(SEED+block).shuffle(order)
    for tag,precision,resolution,style,threads,frames,dataset in order:
        raw=ROOT/'results'/'minicpm_native'/'requests.jsonl'
        existing=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines()] if raw.exists() else []
        completed=[r for r in existing if r['config']['quantization']==tag and r['phase']=='measured' and r['block']==block]
        if completed:
            assert len(completed)==24,'Partial block must be audited rather than silently repeated'
            continue
        projector=a.assets/'models'/'mmproj-model-f16.gguf' if precision=='f16' else ROOT/'models'/f'minicpm-o26-mmproj-{dict(q8="q8_0",q4="q4_k_m")[precision]}.gguf'
        cmd=[sys.executable,str(ROOT/'scripts'/'minicpm_native.py'),'--assets',str(a.assets),
             '--model',str(a.assets/'models'/'Model-7.6B-Q4_K_M.gguf'),'--projector',str(projector),
             '--tag',tag,'--n','24','--block',str(block),'--resolutions',str(resolution),
             '--batch','2048','--ubatch','512','--threads',str(threads),'--frames',str(frames),
             '--prompt-style',style,'--dataset',dataset]
        if tag.endswith('-strict'):
            cmd.append('--strict-prompt')
        print('BLOCK',block,tag,flush=True)
        subprocess.run(cmd,check=True)
