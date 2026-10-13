"""Functional backend/API check; never included in steady-state measurements."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import requests
from benchmark import ROOT,MODELS,stream_request,payload_for
from inputs import squad_samples

p=argparse.ArgumentParser()
p.add_argument('--server',type=Path)
p.add_argument('--model',choices=list(MODELS),default='lfm350-q4')
p.add_argument('--vision',action='store_true')
a=p.parse_args()
binary=a.server or next((ROOT/'runtime'/'cuda').rglob('llama-server.exe'))
folder=ROOT/'results'/'private'
folder.mkdir(exist_ok=True)
cmd=[str(binary),'-m',str(ROOT/'models'/MODELS[a.model]),'-ngl','99','-c','32768','-np','4','-t','4','-tb','4','-b','2048','-ub','512','-fa','on','--cache-ram','0','--no-cache-idle-slots','--host','127.0.0.1','--port','8097','--no-webui','--perf']
if a.model.startswith('smol'):
    cmd+=['--mmproj',str(ROOT/'models'/'mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf')]
with (folder/'smoke.log').open('w') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError((folder/'smoke.log').read_text(errors='replace')[-4000:])
            try:
                if requests.get('http://127.0.0.1:8097/health',timeout=1).status_code==200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.5)
        configs=[{'task':'text','length':64,'dataset':'squad'}]*2
        if a.vision:
            assert a.model.startswith('smol')
            configs+=[{'task':'image','resolution':s} for s in [224,448,896]]+[{'task':'video','resolution':224,'frames':8}]
        for cfg in configs:
            payload,meta=payload_for(cfg,squad_samples()[0],0,requests.Session())
            row=stream_request(payload)
            assert row['error'] is None,row
            assert row['usage']['completion_tokens']==32,row
            assert row['timings']['cache_n']==0,row
            print(json.dumps({'smoke_only':True,'configuration':cfg,'ttft_ms':row['ttft_ms'],'usage':row['usage'],'timings':row['timings'],'first_text_timings':row['first_text_timings']}),flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=20)
