"""Unsplit MiniCPM vision+language CUDA inference; no feature/prefix cache reuse."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import random
import re
import subprocess
import time
import requests
from PIL import Image,ImageDraw
from benchmark import ROOT,SEED,stream_request,Telemetry

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--projector',type=Path,required=True)
    p.add_argument('--tag',default='q4-k-m')
    p.add_argument('--n',type=int,default=8)
    p.add_argument('--resolutions',type=int,nargs='+',default=[448,336,224])
    p.add_argument('--frames',type=int,default=1)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--ubatch',type=int,default=128)
    p.add_argument('--max-tokens',type=int,default=4)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--prompt-style',choices=['standard','compact'],default='standard')
    p.add_argument('--dataset',choices=['streamingbench','synthetic','erqa'],default='streamingbench')
    p.add_argument('--block',type=int,default=0)
    p.add_argument('--strict-prompt',action='store_true')
    p.add_argument('--model-label',default='MiniCPM-o-2.6')
    p.add_argument('--output-group',default='minicpm_native')
    a=p.parse_args()
    output=ROOT/'results'/a.output_group
    output.mkdir(exist_ok=True)
    private=ROOT/'results'/'private'
    binary=next((ROOT/'runtime'/'cuda').rglob('llama-server.exe'))
    cmd=[str(binary),'-m',str(a.model),'--mmproj',str(a.projector),'--device','CUDA0','--mmproj-device','CUDA0',
         '-ngl','99','-c','4096','-np','1','-b',str(a.batch),'-ub',str(a.ubatch),
         '-t',str(a.threads),'-tb',str(a.threads),'-fa','on','--fit','off',
         '--cache-ram','0','--no-cache-idle-slots','--host','127.0.0.1','--port','8097','--no-webui','--perf','-lv','4']
    logpath=private/f'minicpm-{a.tag}.log'
    if a.dataset=='erqa':
        source=ROOT/'data'/'erqa-selected.json'
        selected=json.loads(source.read_text(encoding='utf-8'))
        data={offset+i:r for phase,offset in [('measured',0),('pilot',64),('warmup',112)] for i,r in enumerate(selected[phase])}
    else:
        source=a.assets/'results'/'q4_k_m_four_native'/'predictions.jsonl'
        data=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
        random.Random(SEED).shuffle(data)
    ids=list(range(64,64+a.n)) if a.pilot else list(range(a.block*a.n,(a.block+1)*a.n))
    ids=list(range(112,120))+ids
    log=logpath.open('w')
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        for _ in range(240):
            if proc.poll() is not None:
                raise RuntimeError(logpath.read_text(errors='replace')[-3500:])
            try:
                if requests.get('http://127.0.0.1:8097/health',timeout=1).status_code==200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.5)
        else:
            raise TimeoutError('MiniCPM server startup')
        evidence=[x for x in logpath.read_text(errors='replace').splitlines()
                  if any(k in x.lower() for k in ['offloaded','cuda0','clip using','model size'])]
        counts=[re.search(r'offloaded (\d+)/(\d+) layers',x) for x in evidence]
        assert any(m and m[1]==m[2] for m in counts),evidence
        (output/f'{a.tag}-backend.json').write_text(json.dumps({'model_file':a.model.name,'projector_file':a.projector.name,
            'runtime':'b11577 CUDA12.4','slots':1,'context':4096,'batch':a.batch,'ubatch':a.ubatch,
            'threads':a.threads,'gpu_evidence':evidence,'source_predictions_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
        for resolution in a.resolutions:
            config={'name':f'{a.tag}-{resolution}-f{a.frames}-b{a.batch}-ub{a.ubatch}-{a.prompt_style}-{a.dataset}','model':a.model_label,
                    'quantization':a.tag,'resolution_area_budget':resolution,'frames':a.frames,'batch':a.batch,
                    'ubatch':a.ubatch,'max_tokens':a.max_tokens,'prefix_cache':False,'unsplit':True,'prompt_style':a.prompt_style,
                    'threads':a.threads,'projector_file':a.projector.name,'dataset':a.dataset}
            config['strict_prompt']=a.strict_prompt
            prepared=[]
            for i in ids:
                row=data[i]
                folder=ROOT/'data'/'erqa-cache' if a.dataset=='erqa' else a.assets/'data'/'native_frames'/row['question_id'].replace(' ','_')
                content=[]
                meta=[]
                for frame in (row['images'] if a.dataset=='erqa' else row['frames'][-a.frames:]):
                    if a.dataset=='synthetic':
                        rng=random.Random(SEED+i)
                        colors=['red','blue','green','yellow']
                        label=rng.randrange(4)
                        im=Image.new('RGB',(224,224),'white')
                        draw=ImageDraw.Draw(im)
                        x,y=rng.randrange(20,80),rng.randrange(20,80)
                        draw.rectangle((x,y,x+110,y+110),fill=colors[label])
                    else:
                        with Image.open(folder/frame['file']) as source_image:
                            im=source_image.convert('RGB')
                    if im:
                        scale=min(1,resolution/(im.width*im.height)**.5)
                        width=max(28,int(im.width*scale)//28*28)
                        height=max(28,int(im.height*scale)//28*28)
                        im=im.resize((width,height),Image.Resampling.LANCZOS)
                        b=io.BytesIO()
                        im.save(b,format='JPEG',quality=95)
                    raw=b.getvalue()
                    meta.append({'width':width,'height':height,'sha256':hashlib.sha256(raw).hexdigest()})
                    content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(raw).decode()}})
                question=row['question'] if a.dataset=='erqa' else 'You are an advanced video question-answering AI assistant. Analyze the provided frames and answer the multiple-choice question. Respond with only the letter (A, B, C, or D).\n\nQuestion: '+row['question']+'\n\nOptions:\n'+'\n'.join(row['options'])+'\n\nThe best option is:'
                if a.prompt_style=='compact' and a.dataset!='erqa':
                    question=row['question']+'\n'+'\n'.join(row['options'])+'\nAnswer with one letter:'
                if a.dataset=='synthetic':
                    question='Color? A red B blue C green D yellow. Letter:'
                    row={**row,'question_id':f'synthetic-color-{i}','answer':'ABCD'[label]}
                if a.strict_prompt:
                    question=question+' Reply with exactly one letter, no other text.'
                content.append({'type':'text','text':question})
                payload={'messages':[{'role':'user','content':content}],'temperature':0,'seed':SEED,
                    'stream':True,'stream_options':{'include_usage':True},'max_tokens':a.max_tokens,
                    'cache_prompt':False,'timings_per_token':True}
                prepared.append((i,row,payload,meta))
            vals=[]
            with Telemetry(a.output_group,config['name']):
                for pos,(i,row,payload,meta) in enumerate(prepared):
                    r=stream_request(payload)
                    t=r.get('first_text_timings') or r.get('timings',{})
                    compute=t.get('prompt_ms')
                    final=r.get('timings',{})
                    total=final.get('prompt_ms',0)+final.get('predicted_ms',0)
                    record={'config':config,'phase':'warmup' if pos<8 else ('pilot' if a.pilot else 'measured'),'block':a.block,
                        'sample_index':i,'question_id':row['question_id'],'images':meta,'compute_first_token_ms':compute,
                        'compute_full_answer_ms':total,'expected_answer':row['answer'],'prompt_text':payload['messages'][0]['content'][-1]['text'],
                        'question_type':row.get('question_type'),'seed':SEED,**r}
                    with (output/'requests.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if pos>=8 and not r['error']:
                        vals.append(compute)
            print(config['name'],'valid',len(vals),'/',a.n,'compute TTFT mean_ms',sum(vals)/len(vals) if vals else None,flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()

if __name__=='__main__':
    main()
