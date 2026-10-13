"""Incremental report for fully native small-VLM follow-up measurements."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_compute_report import summarize, save_csv

ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'results'/'vlm_extension'
KINDS=['single','four','color']
KEYS=['qwen25-vl3b','qwen2-vl2b','minicpm-v4']
def main():
    raw=[json.loads(x) for x in (O/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    groups=defaultdict(list)
    for r in raw:
        if r['phase']=='measured':
            groups[r['config']['name']].append(r)
    if not groups:
        return
    summaries=[summarize('vlm_extension',name,rs) for name,rs in groups.items()]
    (O/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    save_csv(O/'summary.csv',summaries)
    issues=[]
    completed=all(len(rs)==48 for rs in groups.values()) and len(groups)==9
    for name,rs in groups.items():
        if len(rs) not in [24,48] or len({r['sample_index'] for r in rs})!=len(rs):
            issues.append('Denominator mismatch '+name)
        if any(r.get('error') or r['timings'].get('cache_n')!=0 for r in rs):
            issues.append('Failure or cache reuse '+name)
        if any(r['compute_first_token_ms']!=r['timings']['prompt_ms'] for r in rs):
            issues.append('First-token timer mismatch '+name)
        if completed and Counter(r['block'] for r in rs)!={0:24,1:24}:
            issues.append('Block mismatch '+name)
    if any(r.get('error') or r['timings'].get('cache_n')!=0 for r in raw):
        issues.append('Warmup/pilot error/cache')
    for path in O.glob('*-backend.json'):
        evidence='\n'.join(json.loads(path.read_text())['gpu_evidence'])
        if not re.search(r'offloaded (\d+)/\1 layers',evidence) or 'CLIP using CUDA0 backend' not in evidence:
            issues.append('GPU offload missing '+path.name)
    # Equal payloads across models, despite each model having its own tokenizer/template.
    bykind=defaultdict(list)
    for name,rs in groups.items():
        kind=rs[0]['config']['quantization'].rsplit('-',1)[1]
        bykind[kind].append(rs)
    for kind,sets in bykind.items():
        reference={r['sample_index']:r for r in sets[0]}
        for rs in sets[1:]:
            for r in rs:
                ref=reference.get(r['sample_index'])
                if ref and (r['images']!=ref['images'] or r['prompt_text']!=ref['prompt_text']):
                    issues.append('Payload parity failed '+kind)
    validation={'passed':completed and not issues,'status':'complete' if completed else 'in_progress',
                'issues':issues,'formal_configurations':len(groups),'formal_requests':sum(map(len,groups.values())),
                'phase_counts':dict(Counter(r['phase'] for r in raw)),'payloads_paired_across_models':not issues,
                'metric':'server prompt_ms; native fresh vision plus language first-token computation','max_cached_tokens':max(r['timings']['cache_n'] for r in raw)}
    (O/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    assert not issues,issues
    manifest=json.loads((ROOT/'data'/'vlm-extension-artifacts.json').read_text())
    lookup={s['configuration']:s for s in summaries}
    def table():
        lines=['| 模型 | 输入 | 有效/尝试 | 首token计算均值 ms | 95% CI ms | p95 ms | 生成结束计算均值 ms | 严格单字母正确/总数 |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
        for key in KEYS:
            label=next(s['model'] for s in manifest['sources'] if s['key']==key)
            for kind in KINDS:
                name=next((n for n,rs in groups.items() if rs[0]['config']['quantization']==key+'-'+kind),None)
                if name:
                    s=lookup[name]
                    lines.append(f'| {label} | {dict(single="真实单帧336",four="真实四帧336",color="颜色单帧224严格指令")[kind]} | {s["valid"]}/{s["attempted"]} | **{s["mean_ms"]:.2f}** | {s["ci95_low_ms"]:.2f}–{s["ci95_high_ms"]:.2f} | {s["p95_ms"]:.2f} | {s["mean_full_answer_compute_ms"]:.2f} | {s["answer_correct"]}/{s["answer_total"]} |')
        return '\n'.join(lines)
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(KEYS))
    for index,kind in enumerate(KINDS):
        vals=[]
        for key in KEYS:
            s=next((s for s in summaries if groups[s['configuration']][0]['config']['quantization']==key+'-'+kind),None)
            vals.append(s['mean_ms'] if s else np.nan)
        ax.bar(x+(index-1)*.25,vals,width=.24,label=kind)
    ax.axhline(140,color='#bd3d3d',ls='--',label='140 ms target')
    ax.set_xticks(x,KEYS)
    ax.set_ylabel('Server computation TTFT (ms)')
    ax.set_title('Native fresh vision + language | RTX 3070 | Q4 language / F16 vision')
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(O/'comparison.png',dpi=150)
    plt.close(fig)
    note='''语言权重均为Q4_K_M，视觉组件均为F16；每个请求重新编码图片并执行语言模型，无特征或前缀缓存。单槽4096context、batch2048/ubatch512、4线程、温度0、固定种子20261013、最多4输出token、EOS正常生效，与本次MiniCPM运行时和计时方法一致。9档按两块随机顺序测量，每块24个配对正式样本+8个独立预热；正式/预热/预试验ID分离。色彩任务与MiniCPM严格指令档使用相同生成规则、JPEG和提示词。实际输入token因模型模板、tokenizer和视觉处理器而不同，跨模型差异不能仅归因参数量。

面积预算336表示像素面积不超过336²并保留宽高比、对齐28，不是固定336×336；颜色任务224×224。真实题取自相同48个StreamingBench问题，四帧含时序上下文；不是把视觉和语言拆开计时。严格正确率要求单字母A–D（允许句点），解释/截断算格式失败；“生成结束计算”只到EOS或4token上限，截断时不能称完整答案。CSV同时保留开头选项正确数和达到输出上限数。48题准确率不代表完整模型基准排名。

命名核实：实际测试 [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)、[Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)、[MiniCPM-V-4（4B）](https://huggingface.co/openbmb/MiniCPM-V-4)。2B属于Qwen2-VL，未虚构Qwen2.5-VL-2B；两个Qwen的GGUF来自ggml-org，MiniCPM-V的GGUF来自原作者OpenBMB。此处MiniCPM-V-4是为8GB显存选择的具体版本，不代表所有MiniCPM-V。
'''
    detail='# Qwen / MiniCPM-V 完整模型计算TTFT补测\n\n日期：2026-10-13（上海）。状态：'+validation['status']+f'；正式{validation["formal_requests"]}条。'+('每档48条，共9档432条。' if completed else '进行中，表中显示当前实际样本数，未完成档不作正式结论。')+'\n\n'+table()+'\n\n![模型对照](../results/vlm_extension/comparison.png)\n\n'+note+'\n[逐请求](../results/vlm_extension/requests.jsonl)、[CSV](../results/vlm_extension/summary.csv)、[验收](../results/vlm_extension/validation.json)、[固定revision与SHA256](../data/vlm-extension-artifacts.json)。\n\n复现：`python scripts/prepare_vlm_extension.py` 下载并校验权重；`python scripts/run_vlm_extension.py --assets <本地原生素材目录>` 测量；`python scripts/report_vlm_extension.py` 重建本页。\n'
    (ROOT/'docs'/'vlm-extension-zh.md').write_text(detail,encoding='utf-8')
    for path,isroot in [(ROOT/'README.md',True),(ROOT/'docs'/'results-zh.md',False)]:
        text=path.read_text(encoding='utf-8').split('<!-- VLM-EXTENSION:START -->')[0].rstrip()
        link='docs/vlm-extension-zh.md' if isroot else 'vlm-extension-zh.md'
        addition=f'\n\n<!-- VLM-EXTENSION:START -->\n## Qwen 与 MiniCPM-V 补测\n\n[完整补测报告]({link})；状态 {validation["status"]}，{validation["formal_requests"]} 条正式测量。'+('连同原实验共64配置、3072条正式测量。' if completed else '')+'\n\n'+table()+'\n\n2B实际模型为Qwen2-VL-2B，3B为Qwen2.5-VL-3B。语言Q4_K_M+视觉F16，每次完整执行。真实任务和颜色任务分列；严格格式失败、输出截断均保留。\n<!-- VLM-EXTENSION:END -->\n'
        path.write_text(text+addition,encoding='utf-8')
    if (ROOT/'results'/'embodied_erqa'/'requests.jsonl').exists():
        from report_erqa import main as erqa_report
        erqa_report()
    print(json.dumps(validation,indent=2))
if __name__=='__main__':
    main()
