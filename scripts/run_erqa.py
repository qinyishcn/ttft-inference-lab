"""Embodied reasoning perception QA; fresh complete VLM calls, not robot actions."""
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
a=p.parse_args()
cases=[
 ('erqa-o26-q8-224','MiniCPM-o-2.6',a.assets/'models'/'Model-7.6B-Q4_K_M.gguf',ROOT/'models'/'minicpm-o26-mmproj-q8_0.gguf',224),
 ('erqa-v4-f16-224','MiniCPM-V-4',ROOT/'models'/'ggml-model-Q4_K_M.gguf',ROOT/'models'/'mmproj-model-f16.gguf',224),
 ('erqa-v4-f16-336','MiniCPM-V-4',ROOT/'models'/'ggml-model-Q4_K_M.gguf',ROOT/'models'/'mmproj-model-f16.gguf',336),
 ('erqa-qwen25-f16-336','Qwen2.5-VL-3B-Instruct',ROOT/'models'/'Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf',ROOT/'models'/'mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf',336),
]
for block in range(1 if a.pilot else 2):
    order=cases.copy()
    random.Random(SEED+900+block).shuffle(order)
    for tag,label,model,projector,res in order:
        raw=ROOT/'results'/'embodied_erqa'/'requests.jsonl'
        rows=[json.loads(x) for x in raw.read_text(encoding='utf-8').splitlines()] if raw.exists() else []
        completed=[r for r in rows if r['config']['quantization']==tag and r['phase']==('pilot' if a.pilot else 'measured') and r['block']==block]
        if completed:
            assert len(completed)==(8 if a.pilot else 24),'Partial block requires audit'
            continue
        cmd=[sys.executable,str(ROOT/'scripts'/'minicpm_native.py'),'--assets',str(a.assets),
             '--model',str(model),'--projector',str(projector),'--model-label',label,'--output-group','embodied_erqa',
             '--tag',tag,'--n','8' if a.pilot else '24','--block',str(block),'--resolutions',str(res),
             '--batch','2048','--ubatch','512','--threads','4','--frames','1','--prompt-style','compact',
             '--dataset','erqa','--strict-prompt']
        if a.pilot:
            cmd.append('--pilot')
        print('BLOCK',block,tag,flush=True)
        subprocess.run(cmd,check=True)
        if not a.pilot:
            subprocess.run([sys.executable,str(ROOT/'scripts'/'report_erqa.py')],check=True)
