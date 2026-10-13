"""Final fixed-denominator audit across original and follow-up computation studies."""
from collections import Counter
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
paths=['results/requests.jsonl','results/minicpm_native/requests.jsonl','results/vlm_extension/requests.jsonl','results/embodied_erqa/requests.jsonl']
counts=Counter();sources=[];errors=0;cache_max=0
for name in paths:
    path=ROOT/name
    raw=path.read_bytes()
    rows=[json.loads(x) for x in raw.decode('utf-8').splitlines()]
    for r in rows:
        phase='pilot' if r.get('stage')=='pilot' and r['phase']=='measured' else r['phase']
        counts[phase]+=1
        errors+=bool(r.get('error'))
        assert r['timings']['prompt_ms']>0
        cache_max=max(cache_max,r['timings']['cache_n'])
    sources.append({'path':name,'requests':len(rows),'sha256':hashlib.sha256(raw).hexdigest()})
validation_files=['results/validation.json','results/compute_only/validation.json','results/vlm_extension/validation.json','results/embodied_erqa/validation.json']
assert all(json.loads((ROOT/p).read_text())['passed'] for p in validation_files)
assert counts=={'measured':3264,'warmup':1312,'pilot':224},counts
assert errors==0 and cache_max==0
groups=sum(json.loads((ROOT/p).read_text())['formal_configurations'] for p in validation_files[1:])
assert groups==68
output={'passed':True,'formal_configurations':groups,'formal_requests':counts['measured'],
        'phase_counts':dict(counts),'all_current_requests':sum(counts.values()),'failed_requests':errors,
        'max_cached_tokens':cache_max,'per_condition_formal_n':48,'sources':sources,
        'validation_files':validation_files,'historical_records_excluded_from_current_denominator':{'native_minicpm':250,'cached_llava_answer':23760,'grace_codec':'20 repeats per operation, aggregate only'},
        'metric':'runtime model first-token computation; not HTTP or robot action-loop latency'}
(ROOT/'results'/'acceptance-all.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
