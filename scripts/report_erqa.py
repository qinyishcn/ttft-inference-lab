"""Report latency, correctness and the 140 ms target for fixed embodied QA cases."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_compute_report import summarize, save_csv
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'results'/'embodied_erqa'
ORDER=['erqa-o26-q8-224','erqa-v4-f16-224','erqa-v4-f16-336','erqa-qwen25-f16-336']
def main():
    rows=[json.loads(x) for x in (O/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    groups=defaultdict(list)
    for r in rows:
        if r['phase']=='measured':
            groups[r['config']['quantization']].append(r)
    if not groups:
        return
    summaries=[];issues=[]
    for tag in ORDER:
        rs=groups.get(tag)
        if not rs:continue
        s=summarize('embodied_erqa',tag,rs)
        s['under_140ms']=sum(r['compute_first_token_ms']<=140 for r in rs)
        s['correct_and_under_140ms']=sum(r['compute_first_token_ms']<=140 and bool(re.fullmatch(r'\s*'+r['expected_answer']+r'[.\s]*',r['text'])) for r in rs)
        s['resolution_area_budget']=rs[0]['config']['resolution_area_budget']
        s['model']=rs[0]['config']['model']
        s['mean_target_met']=s['mean_ms']<=140
        summaries.append(s)
        if len(rs) not in [24,48] or len({r['question_id'] for r in rs})!=len(rs):issues.append('Denominator '+tag)
        if any(r['error'] or r['timings'].get('cache_n')!=0 or r['compute_first_token_ms']!=r['timings']['prompt_ms'] for r in rs):issues.append('Timing/cache/failure '+tag)
    completed=len(groups)==4 and all(len(rs)==48 for rs in groups.values())
    for path in O.glob('*-backend.json'):
        evidence='\n'.join(json.loads(path.read_text())['gpu_evidence'])
        if not re.search(r'offloaded (\d+)/\1 layers',evidence) or 'CLIP using CUDA0 backend' not in evidence:issues.append('CUDA offload '+path.name)
    selected=json.loads((ROOT/'data'/'erqa-selected.json').read_text(encoding='utf-8'))
    expected_ids={r['question_id'] for r in selected['measured']}
    for tag,rs in groups.items():
        if not {r['question_id'] for r in rs}<=expected_ids:issues.append('Source sample '+tag)
        if completed and {r['question_id'] for r in rs}!=expected_ids:issues.append('Missing source sample '+tag)
        if completed and Counter(r['block'] for r in rs)!={0:24,1:24}:issues.append('Block mismatch '+tag)
    phase_ids=[{r['question_id'] for r in selected[phase]} for phase in ['measured','warmup','pilot']]
    if any(phase_ids[i]&phase_ids[j] for i in range(3) for j in range(i+1,3)):issues.append('Selection overlap')
    if any(r.get('error') or r['timings'].get('cache_n')!=0 for r in rows):issues.append('Warmup/pilot error/cache')
    source_by_id={r['question_id']:r for r in selected['measured']}
    by_resolution=defaultdict(dict)
    for tag,rs in groups.items():
        for r in rs:
            source=source_by_id[r['question_id']]
            if r['prompt_text']!=source['question']+' Reply with exactly one letter, no other text.' or r['expected_answer']!=source['answer']:
                issues.append('Source/prompt mismatch '+tag)
            key=(r['config']['resolution_area_budget'],r['sample_index'])
            if key in by_resolution and by_resolution[key]['images']!=r['images']:
                issues.append('Image parity mismatch '+tag)
            by_resolution[key]=r
    validation={'passed':completed and not issues,'status':'complete' if completed else 'in_progress','issues':issues,
        'formal_configurations':len(groups),'formal_requests':sum(map(len,groups.values())),
        'phase_counts':dict(Counter(r['phase'] for r in rows)),
        'target':'140 ms model first sampled token; latency and correctness both retained',
        'scope':'Offline embodied reasoning QA, not closed-loop robot action generation or success',
        'max_cached_tokens':max(r['timings']['cache_n'] for r in rows)}
    (O/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    assert not issues,issues
    (O/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    save_csv(O/'summary.csv',summaries)
    table=['| 模型 / 视觉精度 | 面积预算 | 有效/尝试 | 首token均值 ms | 95% CI ms | p95 ms | 严格正确 | ≤140ms | 正确且≤140ms |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        label=s['model']+(' / 混合Q8' if s['configuration']=='erqa-o26-q8-224' else ' / F16')
        table.append(f'| {label} | {s["resolution_area_budget"]} | {s["valid"]}/{s["attempted"]} | **{s["mean_ms"]:.2f}** | {s["ci95_low_ms"]:.2f}–{s["ci95_high_ms"]:.2f} | {s["p95_ms"]:.2f} | {s["answer_correct"]}/{s["answer_total"]} | {s["under_140ms"]}/{s["attempted"]} | {s["correct_and_under_140ms"]}/{s["attempted"]} |')
    table='\n'.join(table)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(range(len(summaries)),[s['mean_ms'] for s in summaries],color='#347cb4',
           yerr=[[max(0,s['mean_ms']-s['ci95_low_ms']) for s in summaries],[max(0,s['ci95_high_ms']-s['mean_ms']) for s in summaries]],capsize=4)
    ax.axhline(140,color='#bd3d3d',ls='--',label='140 ms target')
    ax.set_xticks(range(len(summaries)),[s['configuration'] for s in summaries],rotation=20,ha='right')
    ax.set_ylabel('Model computation TTFT (ms), mean and 95% CI')
    ax.set_title('ERQA embodied reasoning | 48 fixed single-image questions per final condition')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(O/'comparison.png',dpi=150);plt.close(fig)
    provenance=json.loads((ROOT/'data'/'erqa-provenance.json').read_text())
    doc='# ERQA 具身推理任务：完整模型首 token 计算时延\n\n日期：2026-10-13（上海）。状态：'+validation['status']+f'，正式{validation["formal_requests"]}条。\n\n'+table+'\n\n![ERQA计算时延](../results/embodied_erqa/comparison.png)\n\n'
    doc+='''这次使用真实公开的 [ERQA具身推理基准](https://github.com/embodiedreasoning/ERQA)，测试空间/物理关系、操作可行性和轨迹等场景理解。它与此前单色矩形任务分别统计，**不能用颜色问答140.25ms代替具身任务结果**。ERQA为离线感知/推理选择题，不执行机器人控制，也不证明动作策略、任务成功率或闭环实时性。

项目原有StreamingBench子集含18道Action Perception和2道Spatial Understanding，仅20道相关题且非专门机器人基准；因此新增ERQA固定48题以满足数十条样本要求。原始400题中只选单图片目，完整保留每题全部问题/选项，不按长度、难度、正确率或时延筛题：固定种子20261013打乱合格题，前48正式、后8预热、再8预试验。不同模型/分辨率使用同48题；每档两个24题块，配置顺序按块重新随机，各块另做8题独立预热。'''
    primary_types=Counter(r['question_type'][0] if r['question_type'] else 'Unknown' for r in selected['measured'])
    doc+=f'合格单图题数为{provenance["eligible_single_image"]}；正式题主要类型分布为 `{json.dumps(dict(primary_types),ensure_ascii=False)}`。原多标签分布另存清单，各标签计数不应相加为题数。\n\n'
    doc+='''模型运行时b11577 CUDA12.4、RTX3070 8GB；语言均Q4_K_M；MiniCPM-o视觉为前述混合Q8组件，MiniCPM-V-4与Qwen2.5-VL-3B视觉为F16。因此跨MiniCPM-o与其他模型不只改变参数量/架构，还改变视觉精度，不能作单因素量化结论。MiniCPM-V-4的224/336比较仅改变客户端输入面积预算。每次完整执行视觉编码及语言prefill/首次采样，未拆分部署或缓存视觉特征。单槽4096、batch2048/ubatch512、4线程、温度0、种子固定、前缀缓存关闭并核验cache_n=0、最多4输出token，正常EOS。

主计时为内部 `prompt_ms`，排除权重加载、HTTP、客户端图片准备及进入槽位之前的排队；是运行时模型执行墙钟，不是纯CUDA内核之和。保留宽高比、面积上限224²/336²、尺寸向下对齐28、JPEG95；问题及全部选项不截短，只加 `Reply with exactly one letter, no other text.`。不同分辨率可能改变图中小字/箭头的可读性，须同时看正确率。处理器缩放可能使两档视觉token相同，实际token在CSV保留。

主表分母含全部正式尝试。严格正确要求唯一A–D字母（允许句点），格式错误/截断计错。≤140ms列统计全部题的延迟达标，最后一列同时要求答对；均值接近140ms不等于p95或每题均满足140ms。4token上限下的生成结束耗时如被截断，不称完整答案。CI为固定种子2000次bootstrap，小样本准确率不能外推完整ERQA排名。

模型区别：本报告中的MiniCPM-o2.6和MiniCPM-V-4生成文本；官方 [MiniCPM-RobotManip/RobotTrack](https://github.com/OpenBMB/MiniCPM-Robot) 是另一个面向机器人操作/跟踪的系列。RobotManip输出动作块，不能把其动作步耗时当文本TTFT；本次未声称运行RobotManip、RobotTrack或机器人仿真。官方不同硬件指标也不作为本机实测。

来源与复现：[原ERQA固定commit](https://github.com/embodiedreasoning/ERQA/tree/5bba488252c5b345a43335bfb9239b6bf12bd91c)，CC-BY-4.0，归属Embodied Reasoning QA / embodiedreasoning / Google DeepMind。[源Git blob及SHA256](../data/erqa-provenance.json)、[固定样本与图片哈希](../data/erqa-selected.json)、[逐请求](../results/embodied_erqa/requests.jsonl)、[汇总CSV](../results/embodied_erqa/summary.csv)、[验收](../results/embodied_erqa/validation.json)。未上传完整TFRecord、图片缓存或私人素材；从固定源下载后按Git blob核验。`python scripts/prepare_erqa.py`准备，`python scripts/run_erqa.py --assets <原MiniCPM素材目录>`测量，`python scripts/report_erqa.py`重建报告。
'''
    (ROOT/'docs'/'embodied-erqa-zh.md').write_text(doc,encoding='utf-8')
    for path,root in [(ROOT/'README.md',True),(ROOT/'docs'/'results-zh.md',False)]:
        text=path.read_text(encoding='utf-8').split('<!-- ERQA:START -->')[0].rstrip()
        link='docs/embodied-erqa-zh.md' if root else 'embodied-erqa-zh.md'
        text+=f'\n\n<!-- ERQA:START -->\n## ERQA 具身推理补测\n\n[完整报告]({link})；状态{validation["status"]}，{validation["formal_requests"]}条正式样本。'+('连同此前实测共68配置、3264条正式请求。' if completed else '')+'\n\n'+table+'\n\n这是具身场景理解问答，未执行机器人动作。使用固定真实ERQA题目，完整视觉+语言计算，无特征/前缀复用；低分辨率的速度须结合正确率判断。\n<!-- ERQA:END -->\n'
        if completed:
            chosen=next(s for s in summaries if s['configuration']=='erqa-v4-f16-336')
            baseline=next(s for s in summaries if s['configuration']=='erqa-o26-q8-224')
            text=re.sub(r'\n<!-- CURRENT:START -->.*?<!-- CURRENT:END -->\n','\n',text,flags=re.S)
            first_line,rest=text.split('\n',1)
            overview=f'\n<!-- CURRENT:START -->\n**最新完成**：累计68个配置、3264条正式测量（每配置48条），已补Qwen与MiniCPM-V及真实ERQA具身推理。ERQA单帧MiniCPM-V-4：**{chosen["mean_ms"]:.2f}ms**，{chosen["answer_correct"]}/48答对，{chosen["under_140ms"]}/48不超过140ms；MiniCPM-o2.6：{baseline["mean_ms"]:.2f}ms，{baseline["answer_correct"]}/48答对。**时延达标不等于具身能力可用；未测试机器人动作控制。** [具身报告]({link})。\n<!-- CURRENT:END -->\n'
            text=first_line+'\n'+overview+rest
        path.write_text(text,encoding='utf-8')
    print(json.dumps(validation,indent=2))
if __name__=='__main__':
    main()
