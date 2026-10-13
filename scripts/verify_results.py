"""Acceptance audit of sample denominators, timing boundaries and actual cache state."""
from collections import Counter,defaultdict
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
rows=[json.loads(x) for x in (ROOT/'results'/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
groups=defaultdict(list)
issues=[]
for r in rows:
    if r['phase']=='measured':
        groups[(r['stage'],r['config']['name'])].append(r)
    if r['error']:
        issues.append(f'Failed request: {r["stage"]}/{r["config"]["name"]}/{r["sample_index"]}: {r["error"]}')
        continue
    if not (r['header_ms']<=r['ttft_ms']<=r['total_ms']):
        issues.append('Invalid timing ordering: '+str((r['stage'],r['config']['name'],r['sample_index'])))
    if r['timings'].get('cache_n')!=0:
        issues.append('Nonzero or missing actual cache_n: '+str((r['stage'],r['config']['name'],r['sample_index'])))
    if r['usage'].get('completion_tokens')!=32:
        issues.append('Output budget mismatch: '+str((r['stage'],r['config']['name'],r['sample_index'])))
for (stage,name),rs in groups.items():
    expected=8 if stage=='pilot' else 48
    if len(rs)!=expected or len(set(r['sample_index'] for r in rs))!=expected:
        issues.append(f'Denominator mismatch {stage}/{name}: {len(rs)} records')
    if stage!='pilot' and Counter(r['block'] for r in rs)!={0:24,1:24}:
        issues.append(f'Block mismatch {stage}/{name}')
    if any(r['sample_index']>=48 for r in rs) and stage!='pilot':
        issues.append(f'Final sample boundary mismatch {stage}/{name}')
    if stage=='pilot' and any(not 64<=r['sample_index']<72 for r in rs):
        issues.append(f'Pilot boundary mismatch {name}')
stages=Counter(stage for stage,name in groups)
for stage,n in {'pilot':5,'models':5,'inputs':6,'compute':5,'load':5,'vision':7}.items():
    if stages[stage]!=n:
        issues.append(f'Missing stage configurations {stage}: {stages[stage]}/{n}')
if stages['staircase']<7:
    issues.append('Incomplete staircase')
startups=[json.loads(x) for x in (ROOT/'results'/'startup.jsonl').read_text().splitlines()]
if not startups or any('--device' not in s['command'] or 'CUDA0' not in s['command'] for s in startups):
    issues.append('Explicit CUDA device evidence missing')
data={'passed':not issues,'issues':issues,'per_stage_configurations':dict(stages),'measured_requests':sum(len(rs) for rs in groups.values()),'all_requests_including_warmup':len(rows),'failed_requests':sum(bool(r['error']) for r in rows),'max_cached_tokens':max((r['timings'].get('cache_n',-1) for r in rows),default=-1)}
(ROOT/'results'/'validation.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(data,indent=2))
raise SystemExit(0 if data['passed'] else 1)
