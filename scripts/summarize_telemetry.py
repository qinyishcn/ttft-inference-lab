"""Configuration-window GPU telemetry summaries; includes warm-up and measurement."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
groups=defaultdict(list)
for line in (ROOT/'results'/'telemetry.jsonl').read_text().splitlines():
    row=json.loads(line)
    groups[(row['stage'],row['config'])].append(row)
out=[]
for (stage,name),rs in groups.items():
    item={'stage':stage,'configuration':name,'samples':len(rs)}
    values=defaultdict(list)
    for r in rs:
        for k,v in zip(r.get('fields',[]),r.get('values',[])):
            if k=='timestamp':
                continue
            try:
                values[k].append(float(v))
            except (ValueError,TypeError):
                pass
    for k in ['utilization.gpu','utilization.memory','memory.used','power.draw','clocks.sm','clocks.mem','temperature.gpu']:
        for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:
            item[k+'_'+stat]=float(fn(values[k])) if values[k] else None
    out.append(item)
(ROOT/'results'/'telemetry-summary.json').write_text(json.dumps(out,indent=2))
with (ROOT/'results'/'telemetry-summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print('telemetry windows summarized',len(out))
