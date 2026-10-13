"""Local streaming TTFT experiments with fixed denominators and incremental evidence."""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
import requests
from inputs import ROOT,SEED,messages,squad_samples

URL='http://127.0.0.1:8097'
MODELS={
 'lfm350-q4':'LFM2-350M-Q4_K_M.gguf',
 'lfm350-q8':'LFM2-350M-Q8_0.gguf',
 'lfm350-f16':'LFM2-350M-F16.gguf',
 'lfm1200-q4':'LFM2-1.2B-Q4_K_M.gguf',
 'smol500-q8':'SmolVLM2-500M-Video-Instruct-Q8_0.gguf',
}
MODEL_META={
 'lfm350-q4':{'family':'LFM2','nominal_parameters':354483968,'weight_quantization':'Q4_K_M'},
 'lfm350-q8':{'family':'LFM2','nominal_parameters':354483968,'weight_quantization':'Q8_0'},
 'lfm350-f16':{'family':'LFM2','nominal_parameters':354483968,'weight_quantization':'F16'},
 'lfm1200-q4':{'family':'LFM2','nominal_parameters':1170340608,'weight_quantization':'Q4_K_M'},
 'smol500-q8':{'family':'SmolVLM2','nominal_parameters':500000000,'weight_quantization':'Q8_0'},
}

def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def stream_request(payload, barrier=None):
    """Consume SSE without requests' default 512-byte latency buffering."""
    session=requests.Session()
    session.trust_env=False
    if barrier:
        barrier.wait(timeout=60)
    start=time.perf_counter_ns()
    result={'utc':stamp(),'ttft_ms':None,'first_delta_ms':None,'header_ms':None,'error':None,'text':''}
    latest={}
    first_timings=None
    try:
        with session.post(URL+'/v1/chat/completions',json=payload,stream=True,timeout=(10,180)) as r:
            result['header_ms']=(time.perf_counter_ns()-start)/1e6
            r.raise_for_status()
            for line in r.iter_lines(chunk_size=1):
                if not line.startswith(b'data: '):
                    continue
                raw=line[6:]
                if raw==b'[DONE]':
                    break
                item=json.loads(raw)
                elapsed=(time.perf_counter_ns()-start)/1e6
                if item.get('error'):
                    raise RuntimeError(str(item['error']))
                latest.update({k:v for k,v in item.items() if k in ['usage','timings']})
                if item.get('prompt_progress'):
                    result.setdefault('first_progress_ms',elapsed)
                for choice in item.get('choices',[]):
                    delta=choice.get('delta',{})
                    text=delta.get('content') or ''
                    reasoning=delta.get('reasoning_content') or ''
                    if (text or reasoning) and result['first_delta_ms'] is None:
                        result['first_delta_ms']=elapsed
                    if text and result['ttft_ms'] is None:
                        result['ttft_ms']=elapsed
                        first_timings=item.get('timings')
                    result['text']+=text
        if result['ttft_ms'] is None:
            result['error']='no_nonempty_text'
    except Exception as e:
        result['error']=f'{type(e).__name__}: {e}'
    result['total_ms']=(time.perf_counter_ns()-start)/1e6
    result['usage']=latest.get('usage',{})
    result['timings']=latest.get('timings',{})
    result['first_text_timings']=first_timings
    result['text']=result['text'][:300]
    session.close()
    return result

class Server:
    def __init__(self):
        self.proc=None
        self.key=None
        self.log=None
    def ensure(self,model,slots=4):
        key=(model,slots)
        if self.key==key and self.proc and self.proc.poll() is None:
            return
        self.stop()
        candidates=list((ROOT/'runtime'/'cuda').rglob('llama-server.exe'))
        assert len(candidates)==1,candidates
        cmd=[str(candidates[0]),'-m',str(ROOT/'models'/MODELS[model]),
            '--host','127.0.0.1','--port','8097','--device','CUDA0','-ngl','99','-c',str(8192*slots),
            '-np',str(slots),'-t','4','-tb','4','-b','2048','-ub','512','-fa','on',
            '--cache-ram','0','--no-cache-idle-slots','--cache-reuse','0',
            '--seed',str(SEED),'--no-webui','--log-timestamps','--perf','-lv','4']
        if model.startswith('smol'):
            cmd+=['--mmproj',str(ROOT/'models'/'mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf')]
        logpath=ROOT/'results'/'private'/f'server-{model}-slots{slots}.log'
        logpath.parent.mkdir(exist_ok=True)
        self.log=logpath.open('ab')
        load_start=time.perf_counter()
        self.proc=subprocess.Popen(cmd,stdout=self.log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(240):
            if self.proc.poll() is not None:
                raise RuntimeError(logpath.read_text(errors='replace')[-5000:])
            try:
                if requests.get(URL+'/health',timeout=1).status_code==200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError('server load timeout')
        self.key=key
        evidence=[line.replace(str(ROOT),'.') for line in logpath.read_text(errors='replace').splitlines() if any(x in line.lower() for x in ['offload','cuda','n_ctx','model size','model params'])]
        with (ROOT/'results'/'startup.jsonl').open('a',encoding='utf-8') as out:
            out.write(json.dumps({'utc':stamp(),'model':model,'slots':slots,'load_to_ready_s':time.perf_counter()-load_start,'backend_evidence':evidence[-60:],'command':[str(Path(x).relative_to(ROOT)) if str(x).startswith(str(ROOT)) else x for x in cmd]})+'\n')
    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.log:
            self.log.close()
        self.proc=None
        self.key=None

class Telemetry:
    def __init__(self,stage,cfg):
        self.stop_event=threading.Event()
        self.stage=stage
        self.cfg=cfg
        self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        fields='timestamp,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu'
        with (ROOT/'results'/'telemetry.jsonl').open('a',encoding='utf-8') as out:
            while not self.stop_event.is_set():
                try:
                    r=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    out.write(json.dumps({'utc':stamp(),'stage':self.stage,'config':self.cfg,'fields':fields.split(','),'values':r.stdout.strip().split(', '),'error':r.stderr.strip()})+'\n')
                    out.flush()
                except Exception as e:
                    out.write(json.dumps({'utc':stamp(),'stage':self.stage,'config':self.cfg,'error':str(e)})+'\n')
                self.stop_event.wait(.5)
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop_event.set()
        self.thread.join(timeout=7)

class Competitor:
    def __init__(self,duty,tag):
        self.duty=duty
        self.tag=tag
        self.proc=None
    def __enter__(self):
        if self.duty is None:
            return self
        folder=ROOT/'results'/'private'
        self.ready=folder/(self.tag+'.ready')
        self.stop=folder/(self.tag+'.stop')
        self.ready.unlink(missing_ok=True)
        self.stop.unlink(missing_ok=True)
        self.log=(folder/(self.tag+'.competitor.log')).open('w')
        self.proc=subprocess.Popen([sys.executable,str(ROOT/'scripts'/'gpu_competitor.py'),'--duty',str(self.duty),'--ready',str(self.ready),'--stop',str(self.stop)],stdout=self.log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(120):
            if self.ready.exists():
                time.sleep(1)
                return self
            if self.proc.poll() is not None:
                raise RuntimeError('GPU competitor failed: '+str(self.tag))
            time.sleep(.25)
        raise TimeoutError('GPU competitor startup timeout')
    def __exit__(self,*args):
        if self.proc:
            self.stop.touch()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            self.log.close()
            lines=(ROOT/'results'/'private'/(self.tag+'.competitor.log')).read_text(errors='replace').splitlines()
            with (ROOT/'results'/'competitors.jsonl').open('a') as out:
                out.write(json.dumps({'tag':self.tag,'ready':json.loads(self.ready.read_text()) if self.ready.exists() else None,'output':lines[-1:]})+'\n')

def payload_for(cfg,sample,index,session):
    msg,meta=messages(sample,index,cfg,session)
    payload={'messages':msg,'stream':True,'stream_options':{'include_usage':True},'temperature':0,'seed':SEED,
        'max_tokens':32,'ignore_eos':True,'cache_prompt':False,'timings_per_token':True,'return_progress':True}
    return payload,meta

def measure_config(server,stage,cfg,samples,n=48,block=0):
    name=cfg['name']
    server.ensure(cfg['model'],cfg.get('slots',4))
    users=cfg.get('users',1)
    session=requests.Session()
    session.trust_env=False
    # Preprocessing / media construction / JSON creation happens outside measured interval.
    offset=64 if stage=='pilot' else block*n
    inputs=[payload_for(cfg,samples[offset+i],offset+i,session) for i in range(n)]
    warm=[payload_for(cfg,samples[112+i],112+i,session) for i in range(8)]
    records=[]
    with Competitor(cfg.get('duty'),f'{stage}-{name}-{block}'), Telemetry(stage,name):
        for i,(p,meta) in enumerate(warm):
            r=stream_request(p)
            records.append({'stage':stage,'config':cfg,'phase':'warmup','block':block,'sample_index':112+i,'input':meta,**r})
        order=list(range(n))
        random.Random(SEED+block).shuffle(order)
        with cf.ThreadPoolExecutor(max_workers=users) as pool:
            for start in range(0,n,users):
                ids=order[start:start+users]
                barrier=threading.Barrier(len(ids))
                futures=[pool.submit(stream_request,inputs[i][0],barrier) for i in ids]
                for i,future in zip(ids,futures):
                    records.append({'stage':stage,'config':cfg,'phase':'measured','block':block,'wave_id':f'{block}-{start//users}','sample_index':offset+i,'input':inputs[i][1],**future.result()})
    with (ROOT/'results'/'requests.jsonl').open('a',encoding='utf-8') as out:
        for r in records:
            out.write(json.dumps(r,ensure_ascii=False)+'\n')
    measured=[r['ttft_ms'] for r in records if r['phase']=='measured' and r['error'] is None]
    print(stage,name,'valid',len(measured),'/',n,'mean_ms',round(sum(measured)/len(measured),2) if measured else None,flush=True)
    session.close()
    return records

def cfg(name,model,**kwargs):
    return {'name':name,'model':model,**MODEL_META[model],'task':'text','length':64,'dataset':'squad','users':1,'slots':4,**kwargs}

def publish(stage,enabled):
    subprocess.run([sys.executable,str(ROOT/'scripts'/'report.py')],check=True,cwd=ROOT)
    if enabled:
        # Explicit allowlist excludes model weights, binaries, private logs and third-party full datasets.
        subprocess.run(['git','add','README.md','.gitignore','requirements.txt','scripts','docs/methodology.md','data/artifacts.json','data/squad-selected.json','data/environment.json','results/requests.jsonl','results/startup.jsonl','results/telemetry.jsonl','results/summary.csv','results/summary.json','results/paired-effects.csv','results/figures','results/competitors.jsonl'],check=True,cwd=ROOT)
        subprocess.run(['git','commit','-m',f'Publish {stage} TTFT measurements and reproducible report'],check=True,cwd=ROOT)
        subprocess.run(['git','push','origin','main'],check=True,cwd=ROOT)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--publish',action='store_true')
    p.add_argument('--stage',default='all',choices=['all','pilot','models','quant_long','inputs','inputs_exact','compute','load','vision','staircase'])
    a=p.parse_args()
    samples=squad_samples()
    (ROOT/'results'/'private').mkdir(exist_ok=True)
    (ROOT/'results'/'competitors.jsonl').touch(exist_ok=True)
    server=Server()
    def run_stage(stage,configs,n=48):
        for block in range(1 if stage=='pilot' else 2):
            random.Random(SEED+block).shuffle(configs)
            for c in configs:
                path=ROOT/'results'/'requests.jsonl'
                existing=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
                done=[r for r in existing if r['stage']==stage and r['config']['name']==c['name'] and r['phase']=='measured' and r['block']==block]
                if len(done)==(n if stage=='pilot' else n//2):
                    print('resume: already measured',stage,c['name'],block,flush=True)
                    continue
                assert not done,'Partial block requires inspection before rerun'
                measure_config(server,stage,c,samples,n if stage=='pilot' else n//2,block)
        publish(stage,a.publish)
    try:
        if a.stage in ['all','pilot']:
            run_stage('pilot',[cfg(m,m) for m in MODELS],8)
        rows=[json.loads(x) for x in (ROOT/'results'/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        pilot={m:[r['ttft_ms'] for r in rows if r['stage']=='pilot' and r['phase']=='measured' and r['config']['model']==m and not r['error']] for m in MODELS}
        means={m:sum(v)/len(v) for m,v in pilot.items() if v}
        best=min(means,key=means.get)
        print('PILOT selected',best,means,flush=True)
        if a.stage in ['all','models']:
            run_stage('models',[cfg(m,m) for m in MODELS])
        rows=[json.loads(x) for x in (ROOT/'results'/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        confirmed={m:[r['ttft_ms'] for r in rows if r['stage']=='models' and r['phase']=='measured' and r['config']['model']==m and not r['error']] for m in MODELS}
        confirmed_means={m:sum(v)/len(v) for m,v in confirmed.items() if len(v)>=48}
        if confirmed_means:
            best=min(confirmed_means,key=confirmed_means.get)
        base=lambda name,**kw:cfg(name,best,**kw)
        print('Final model comparison baseline',best,confirmed_means,flush=True)
        staircase=[base('step-0-minimum')]
        # Keep precision constant when changing model size; precision gets its own step.
        if best=='smol500-q8':
            staircase.append(cfg('step-1-model-lfm350-q8','lfm350-q8'))
        if best!='lfm350-q4':
            staircase.append(cfg(f'step-{len(staircase)}-precision-q4','lfm350-q4'))
        staircase.extend([
            cfg(f'step-{len(staircase)}-context512','lfm350-q4',length=512),
            cfg(f'step-{len(staircase)+1}-context2048','lfm350-q4',length=2048),
            cfg(f'step-{len(staircase)+2}-larger','lfm1200-q4',length=2048),
            cfg(f'step-{len(staircase)+3}-contention','lfm1200-q4',length=2048,duty=.5),
            cfg(f'step-{len(staircase)+4}-users4','lfm1200-q4',length=2048,duty=.5,users=4),
            cfg(f'step-{len(staircase)+5}-users8','lfm1200-q4',length=2048,duty=.5,users=8),
        ])
        stages={
            'quant_long':[cfg(m,m,length=2048) for m in ['lfm350-q4','lfm350-q8','lfm350-f16']],
            'inputs':[base(f'squad-{n}',length=n) for n in [64,256,1024,2048]]+[base(f'synthetic-{n}',length=n,dataset='synthetic') for n in [64,1024]],
            'inputs_exact':[base(f'squad-total-{n}',total_tokens=n,length=None) for n in [128,512,1024,2048]]+[base(f'synthetic-total-{n}',total_tokens=n,length=None,dataset='synthetic') for n in [128,1024]],
            'compute':[base('no-worker',length=512)]+[base(f'gemm-{d}',length=512,duty=d) for d in [0,.25,.5,.75]],
            'load':[base(f'users-{u}',length=512,users=u) for u in [1,2,4,8]]+[base('users-4-slots-1',length=512,users=4,slots=1)],
            'vision':[cfg('text','smol500-q8')]+[cfg(f'image-{s}','smol500-q8',task='image',resolution=s) for s in [224,448,896]]+[cfg(f'video-{f}','smol500-q8',task='video',resolution=224,frames=f) for f in [2,4,8]],
            'staircase':staircase,
        }
        for stage,configs in stages.items():
            if a.stage in ['all',stage]:
                run_stage(stage,configs)
    finally:
        server.stop()

if __name__=='__main__':
    main()
