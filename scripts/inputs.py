"""Paired, deterministic inputs. No accuracy benchmark is implied."""
import base64
import hashlib
import io
import json
import random
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
SEED=20261013

def squad_samples():
    path=ROOT/'data'/'squad-dev-v1.1.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                rows.append({'id':qa['id'],'question':qa['question'],'context':paragraph['context'],'answers':qa['answers']})
    random.Random(SEED).shuffle(rows)
    chosen=rows[:128]
    (ROOT/'data'/'squad-selected.json').write_text(json.dumps({'source':'SQuAD 1.1 dev','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'seed':SEED,'samples':chosen},indent=2),encoding='utf-8')
    return chosen

def scene(index,frame=0,size=224):
    rng=random.Random(SEED+index)
    im=Image.new('RGB',(896,896),(235,235,235))
    d=ImageDraw.Draw(im)
    count=2+index%5
    colors=['red','blue','green','orange','purple','black']
    for k in range(count):
        x=(rng.randrange(60,700)+frame*35)%730
        y=rng.randrange(60,700)
        if k%2:
            d.ellipse((x,y,x+100,y+100),fill=colors[k%6])
        else:
            d.rectangle((x,y,x+100,y+100),fill=colors[k%6])
    im=im.resize((size,size),Image.Resampling.LANCZOS)
    b=io.BytesIO()
    im.save(b,format='PNG')
    raw=b.getvalue()
    return {'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}

def text_content(sample,index,length,dataset,session):
    question=sample['question'] if dataset=='squad' else f'What is {index+13} plus {index+7}?'
    context=sample['context'] if dataset=='squad' else 'The quick brown fox walks beside a calm river. '
    # Crop or repeat context by TOKEN IDs, not characters. Question remains intact.
    tokens=session.post('http://127.0.0.1:8097/tokenize',json={'content':context,'add_special':False},timeout=30).json()['tokens']
    tokens=(tokens*(length//len(tokens)+1))[:length]
    cropped=session.post('http://127.0.0.1:8097/detokenize',json={'tokens':tokens},timeout=30).json()['content']
    return f'Context:\n{cropped}\nQuestion: {question}\nAnswer briefly:', {'context_tokens_requested':length,'dataset':dataset,'source_id':sample['id'] if dataset=='squad' else f'synthetic-{index}'}

def messages(sample,index,cfg,session):
    if cfg.get('task','text')=='text':
        text,meta=text_content(sample,index,cfg.get('length',64),cfg.get('dataset','squad'),session)
        return [{'role':'user','content':text}],meta
    frames=cfg.get('frames',1)
    size=cfg.get('resolution',224)
    content=[]
    hashes=[]
    for f in range(frames):
        if frames>1:
            content.append({'type':'text','text':f'Frame {f} at {f*.5:.1f} seconds:'})
        image=scene(index,f,size)
        hashes.append({k:v for k,v in image.items() if k!='base64'})
        content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+image['base64']}})
    text='What shapes and colors are visible? Answer briefly.' if frames==1 else 'Describe the direction the shapes move across these ordered video frames. Answer briefly.'
    content.append({'type':'text','text':text})
    return [{'role':'user','content':content}],{'dataset':'synthetic-shapes-v1','source_id':f'scene-{index}','resolution':size,'frames':frames,'images':hashes,'video_sampling_fps':2 if frames>1 else None}
