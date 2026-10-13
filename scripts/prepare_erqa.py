"""Pin ERQA, verify the Git blob, and parse TF Examples without TensorFlow."""
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import concurrent.futures as cf
from collections import Counter
from PIL import Image
from prepare_vlm_extension import get
ROOT=Path(__file__).resolve().parents[1]
REV='5bba488252c5b345a43335bfb9239b6bf12bd91c'
URL=f'https://raw.githubusercontent.com/embodiedreasoning/ERQA/{REV}/data/erqa.tfrecord'

def varint(raw,pos):
    value=shift=0
    while True:
        b=raw[pos];pos+=1
        value|=(b&127)<<shift
        if b<128:return value,pos
        shift+=7
        assert shift<=70
def fields(raw):
    pos=0
    while pos<len(raw):
        tag,pos=varint(raw,pos)
        wire=tag&7
        if wire==0:value,pos=varint(raw,pos)
        elif wire==2:
            n,pos=varint(raw,pos);value=raw[pos:pos+n];pos+=n
            assert len(value)==n
        elif wire in [1,5]:
            n=8 if wire==1 else 4;value=raw[pos:pos+n];pos+=n
        else:raise ValueError('Unexpected protobuf wire type')
        yield tag>>3,wire,value
def example(raw):
    features=next(value for number,wire,value in fields(raw) if number==1)
    result={}
    for number,wire,entry in fields(features):
        assert number==1
        pair={n:v for n,w,v in fields(entry)}
        key=pair[1].decode()
        f=list(fields(pair[2]))
        assert len(f)==1
        kind,_,value=f[0]
        if kind==1:
            values=[v for n,w,v in fields(value) if n==1]
        elif kind==3:
            values=[]
            for n,w,v in fields(value):
                if w==0:values.append(v)
                elif w==2:
                    pos=0
                    while pos<len(v):
                        item,pos=varint(v,pos);values.append(item)
        else:raise ValueError('Unexpected ERQA feature type')
        result[key]=values
    return result
def main():
    cache=ROOT/'data'/'erqa-cache'
    cache.mkdir(exist_ok=True)
    path=cache/'erqa.tfrecord'
    tree=get(f'https://api.github.com/repos/embodiedreasoning/ERQA/git/trees/{REV}?recursive=1').json()
    entry=next(x for x in tree['tree'] if x['path']=='data/erqa.tfrecord')
    if not path.exists() or path.stat().st_size!=entry['size']:
        parts=cache/'parts'
        parts.mkdir(exist_ok=True)
        chunk=1024*1024
        def part(start):
            end=min(start+chunk,entry['size'])-1
            dest=parts/str(start)
            if dest.exists() and dest.stat().st_size==end-start+1:
                return dest
            response=get(URL,headers={'Range':f'bytes={start}-{end}'})
            assert response.status_code==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{entry["size"]}'
            assert len(response.content)==end-start+1
            dest.write_bytes(response.content)
            return dest
        with cf.ThreadPoolExecutor(32) as pool:
            files=list(pool.map(part,range(0,entry['size'],chunk)))
        with path.open('wb') as f:
            for source in files:
                f.write(source.read_bytes())
    raw=path.read_bytes()
    assert len(raw)==entry['size']
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    assert blob==entry['sha'],'ERQA source Git blob mismatch'
    rows=[];pos=0
    while pos<len(raw):
        size=struct.unpack_from('<Q',raw,pos)[0]
        data=raw[pos+12:pos+12+size]
        assert len(data)==size
        pos+=16+size
        features=example(data)
        index=len(rows)
        images=[]
        for j,b in enumerate(features['image/encoded']):
            dest=cache/f'{index:03d}-{j}.image'
            dest.write_bytes(b)
            with Image.open(io.BytesIO(b)) as im:
                images.append({'file':dest.name,'sha256':hashlib.sha256(b).hexdigest(),'size':list(im.size)})
        rows.append({'source_index':index,'question_id':f'erqa-{index:03d}',
            'question':features['question'][0].decode(),'answer':features['answer'][0].decode(),
            'question_type':[x.decode() for x in features.get('question_type',[])],
            'visual_indices':features.get('visual_indices',[]),'images':images})
    assert pos==len(raw) and len(rows)==400
    (cache/'all.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    # Single-image subset supports an unambiguous full-image latency/accuracy control.
    eligible=[r for r in rows if len(r['images'])==1]
    random.Random(20261013).shuffle(eligible)
    assert len(eligible)>=64,'Need separate 48 final + 8 warm + 8 pilot questions'
    selected={'measured':eligible[:48],'warmup':eligible[48:56],'pilot':eligible[56:64]}
    (ROOT/'data'/'erqa-selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'repository':'https://github.com/embodiedreasoning/ERQA','revision':REV,'url':URL,
        'license':'CC-BY-4.0','attribution':'Embodied Reasoning QA (ERQA), embodiedreasoning / Google DeepMind',
        'bytes':len(raw),'git_blob_sha1':blob,'sha256':hashlib.sha256(raw).hexdigest(),'records':len(rows),
        'eligible_single_image':len(eligible),'selection_seed':20261013,'selection':'shuffle single-image records; first48 final, next8 warm, next8 pilot; no selection by difficulty, correctness or latency',
        'all_question_types':dict(Counter(t for r in rows for t in r['question_type'])),
        'measured_question_types':dict(Counter(t for r in selected['measured'] for t in r['question_type']))}
    (ROOT/'data'/'erqa-provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
    print('Examples:',[(r['question_id'],r['question'][:200],r['visual_indices']) for r in selected['measured'][:3]])
if __name__=='__main__':
    main()
