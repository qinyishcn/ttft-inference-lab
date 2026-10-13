"""Regenerate public statistics and figures from per-request measurements."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'

def ci(values,groups=None):
    rng=np.random.default_rng(20261013)
    if groups:
        buckets=defaultdict(list)
        for v,g in zip(values,groups):
            buckets[g].append(v)
        bs=list(buckets.values())
        means=[np.mean([v for k in rng.integers(0,len(bs),len(bs)) for v in bs[k]]) for _ in range(2000)]
    else:
        v=np.asarray(values)
        means=np.mean(rng.choice(v,(2000,len(v)),replace=True),axis=1)
    return [float(x) for x in np.percentile(means,[2.5,97.5])]

def main():
    rows=[json.loads(x) for x in (R/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    grouped=defaultdict(list)
    for r in rows:
        if r['phase']=='measured':
            grouped[(r['stage'],r['config']['name'])].append(r)
    summaries=[]
    for (stage,name),rs in grouped.items():
        valid=[r for r in rs if r['error'] is None and r['ttft_ms'] is not None]
        vals=[r['ttft_ms'] for r in valid]
        cfg=rs[0]['config']
        tokens=[r['usage'].get('prompt_tokens',r['timings'].get('prompt_n')) for r in valid]
        tokens=[x for x in tokens if x is not None]
        cached=[r['timings'].get('cache_n',r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)) for r in valid]
        prompt_times=[r['timings']['prompt_ms'] for r in valid if 'prompt_ms' in r['timings']]
        residual=[r['ttft_ms']-r['timings']['prompt_ms'] for r in valid if 'prompt_ms' in r['timings']]
        summary={'stage':stage,'configuration':name,'model':cfg['model'],'attempted':len(rs),'valid':len(valid),'failed':len(rs)-len(valid),
            'distinct_samples':len(set(r['sample_index'] for r in rs)),'mean_ms':float(np.mean(vals)) if vals else None,
            'median_ms':float(np.median(vals)) if vals else None,'p95_ms':float(np.percentile(vals,95)) if vals else None,
            'std_ms':float(np.std(vals,ddof=1)) if len(vals)>1 else None,'ci95_low_ms':None,'ci95_high_ms':None,
            'mean_input_tokens':float(np.mean(tokens)) if tokens else None,'max_cached_tokens':max(cached) if cached else None,
            'mean_server_prompt_ms':float(np.mean(prompt_times)) if prompt_times else None,'mean_ttft_minus_prompt_ms':float(np.mean(residual)) if residual else None,
            'ci_method':'wave bootstrap' if cfg.get('users',1)>1 else 'request bootstrap'}
        if vals:
            summary['ci95_low_ms'],summary['ci95_high_ms']=ci(vals,[r['wave_id'] for r in valid] if cfg.get('users',1)>1 else None)
        summaries.append(summary)
    (R/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    with (R/'summary.csv').open('w',newline='',encoding='utf-8') as out:
        w=csv.DictWriter(out,fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)
    bases={'models':'lfm350-q4','quant_long':'lfm350-q4','inputs':'squad-64','inputs_exact':'squad-total-128','compute':'gemm-0','load':'users-1','vision':'text','staircase':'step-0-minimum'}
    effects=[]
    for (stage,name),rs in grouped.items():
        if stage not in bases or name==bases[stage]:
            continue
        reference=bases[stage]
        if stage=='inputs_exact' and name.startswith('synthetic-total-'):
            reference=name.replace('synthetic','squad',1)
        br=grouped.get((stage,reference),[])
        byid={r['sample_index']:r for r in br if r['error'] is None}
        deltas=[r['ttft_ms']-byid[r['sample_index']]['ttft_ms'] for r in rs if r['error'] is None and r['sample_index'] in byid]
        if deltas:
            lo,hi=ci(deltas)
            effects.append({'stage':stage,'configuration':name,'reference':reference,'paired_n':len(deltas),'mean_delta_ms':float(np.mean(deltas)),'ci95_low_ms':lo,'ci95_high_ms':hi})
    with (R/'paired-effects.csv').open('w',newline='',encoding='utf-8') as out:
        w=csv.DictWriter(out,fieldnames=['stage','configuration','reference','paired_n','mean_delta_ms','ci95_low_ms','ci95_high_ms'])
        w.writeheader()
        w.writerows(effects)
    figdir=R/'figures'
    figdir.mkdir(exist_ok=True)
    stages=['pilot','models','quant_long','inputs','inputs_exact','compute','load','vision','staircase']
    content=['\n<!-- RESULTS:START -->','## Measured results','',
        'All times below are **milliseconds**, warm requests only. Pilot rows have 8 requests and are excluded from final conclusions. Final configurations target 48 paired requests in two randomized blocks. CI: seeded bootstrap; concurrent conditions resample whole waves. See [methodology](docs/methodology.md), [raw requests](results/requests.jsonl), [summary CSV](results/summary.csv), and [paired differences](results/paired-effects.csv).','']
    for stage in stages:
        ss=[s for s in summaries if s['stage']==stage]
        if not ss:
            continue
        levels={
            'models':['lfm350-q4','lfm350-q8','lfm350-f16','lfm1200-q4','smol500-q8'],
            'quant_long':['lfm350-q4','lfm350-q8','lfm350-f16'],
            'inputs':['squad-64','squad-256','squad-1024','squad-2048','synthetic-64','synthetic-1024'],
            'inputs_exact':['squad-total-128','squad-total-512','squad-total-1024','squad-total-2048','synthetic-total-128','synthetic-total-1024'],
            'compute':['no-worker','gemm-0','gemm-0.25','gemm-0.5','gemm-0.75'],
            'load':['users-1','users-2','users-4','users-8','users-4-slots-1'],
            'vision':['text','image-224','image-448','image-896','video-2','video-4','video-8'],
        }
        if stage=='staircase':
            ss.sort(key=lambda s:s['configuration'])
        elif stage in levels:
            ss.sort(key=lambda s:levels[stage].index(s['configuration']))
        content+=['### '+stage,'','| Configuration | Valid / attempted | Mean | 95% CI | Median | p95 | Input tokens (mean) |','|---|---:|---:|---:|---:|---:|---:|']
        if stage=='inputs':
            content[-3:-3]=['Original context-budget experiment: token-array repeat + text round-trip can merge tokens (especially the synthetic repeated sentence). Names are requested budgets, **not matched final token counts**. The actual counts below are retained. Use `inputs_exact` for strictly matched total input tokens.','']
        for s in ss:
            fmt=lambda v:f'{v:.2f}' if v is not None else 'NA'
            content.append(f'| {s["configuration"]} | {s["valid"]} / {s["attempted"]} | {fmt(s["mean_ms"])} | {fmt(s["ci95_low_ms"])}–{fmt(s["ci95_high_ms"])} | {fmt(s["median_ms"])} | {fmt(s["p95_ms"])} | {fmt(s["mean_input_tokens"])} |')
        good=[s for s in ss if s['mean_ms'] is not None]
        if good:
            fig,ax=plt.subplots(figsize=(max(8,len(good)*1.4),4.6))
            yy=np.array([s['mean_ms'] for s in good])
            err=np.array([[max(0,s['mean_ms']-s['ci95_low_ms']) for s in good],[max(0,s['ci95_high_ms']-s['mean_ms']) for s in good]])
            ax.bar(range(len(good)),yy,yerr=err,capsize=4,color='#347cb4')
            ax.set_xticks(range(len(good)),[s['configuration'] for s in good],rotation=25,ha='right')
            ax.set_ylabel('TTFT (ms), mean and bootstrap 95% CI')
            ax.set_title(stage+' | RTX 3070, llama.cpp CUDA')
            ax.grid(axis='y',alpha=.2)
            ax.set_axisbelow(True)
            fig.tight_layout()
            fig.savefig(figdir/(stage+'.png'),dpi=150)
            plt.close(fig)
        content+=['',f'![{stage} TTFT](results/figures/{stage}.png)','']
    final=[s for s in summaries if s['stage']=='models' and s['valid']>=40]
    if final:
        winner=min(final,key=lambda s:s['mean_ms'])
        pilot=[s for s in summaries if s['stage']=='pilot' and s['valid']]
        selection=min(pilot,key=lambda s:s['mean_ms']) if pilot else winner
        content+=['## Observations and limits','',f'Pilot selected **{selection["configuration"]}**. Independent final model comparison minimum: **{winner["configuration"]}**, {winner["mean_ms"]:.2f} ms. Subsequent factor tests use this final comparison minimum as their text baseline. The separate staircase remeasures it. This is the observed minimum among tested candidates, not a global optimum.','']
    stairs=sorted([s for s in summaries if s['stage']=='staircase'],key=lambda s:s['configuration'])
    if len(stairs)>1:
        reversals=[f'{a["configuration"]} → {b["configuration"]}' for a,b in zip(stairs,stairs[1:]) if a['mean_ms'] is not None and b['mean_ms'] is not None and b['mean_ms']<a['mean_ms']]
        content+=['Staircase mean reversals: '+('; '.join(reversals) if reversals else 'none observed in completed steps')+'. Retained without filtering.','']
    content+=['Input resolution and visual tokens may saturate at the model processor limit. Quantization may increase prefill overhead. GPU GEMM duty is not a hardware-equivalence claim. Concurrent results include server queueing and batching. Visual inputs are controlled synthetic scenes, not a real-world vision benchmark. No quality, FLOP-count, production-throughput or cross-GPU conclusion is established.','', '<!-- RESULTS:END -->','']
    readme=ROOT/'README.md'
    text=readme.read_text(encoding='utf-8')
    if '<!-- RESULTS:START -->' in text:
        text=text.split('<!-- RESULTS:START -->')[0].rstrip()+'\n'
    readme.write_text(text+'\n'.join(content),encoding='utf-8')
    print('report regenerated:',len(summaries),'configurations',flush=True)

if __name__=='__main__':
    main()

