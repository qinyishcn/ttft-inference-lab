"""Rebuild computation-only TTFT tables, paired effects, figures and Chinese report."""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report import ci

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
O=R/'compute_only'
def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
def metric(r):
    return r['timings']['prompt_ms']
def summarize(stage,name,rs):
    valid=[r for r in rs if not r.get('error') and r['timings'].get('prompt_ms') is not None]
    values=[metric(r) for r in valid]
    wave=rs[0]['config'].get('users',1)>1
    lo,hi=ci(values,[r['wave_id'] for r in valid] if wave else None)
    expected=[r for r in valid if 'expected_answer' in r]
    correct=sum(bool(re.fullmatch(r'\s*([A-D])[.\s]*',r['text'])) and r['text'].strip().rstrip('.')==r['expected_answer'] for r in expected)
    leading_correct=sum(bool(re.match(r'^\s*'+re.escape(r['expected_answer'])+r'(?:\b|[.\s])',r['text'])) for r in expected)
    return {'stage':stage,'configuration':name,'attempted':len(rs),'valid':len(valid),'failed':len(rs)-len(valid),
            'distinct_samples':len({r['sample_index'] for r in rs}),
            'mean_ms':float(np.mean(values)),'median_ms':float(np.median(values)),
            'p95_ms':float(np.percentile(values,95)),'std_ms':float(np.std(values,ddof=1)),
            'ci95_low_ms':lo,'ci95_high_ms':hi,
            'mean_input_tokens':float(np.mean([r['timings']['prompt_n'] for r in valid])),
            'max_cached_tokens':max(r['timings']['cache_n'] for r in valid),
            'mean_full_answer_compute_ms':float(np.mean([metric(r)+r['timings']['predicted_ms'] for r in valid])),
            'mean_generated_tokens':float(np.mean([r['timings']['predicted_n'] for r in valid])),
            'answer_correct':correct if expected else None,'answer_total':len(expected) or None,
            'leading_choice_correct':leading_correct if expected else None,
            'output_budget_reached':sum(r['timings']['predicted_n']>=r['config'].get('max_tokens',32) for r in valid),
            'ci_method':'whole-wave bootstrap' if wave else 'request bootstrap'}

LEVELS={
 'models':['lfm350-q4','lfm350-q8','lfm350-f16','lfm1200-q4','smol500-q8'],
 'quant_long':['lfm350-q4','lfm350-q8','lfm350-f16'],
 'inputs':['squad-64','squad-256','squad-1024','squad-2048','synthetic-64','synthetic-1024'],
 'inputs_exact':['squad-total-128','squad-total-512','squad-total-1024','squad-total-2048','synthetic-total-128','synthetic-total-1024'],
 'compute':['no-worker','gemm-0','gemm-0.25','gemm-0.5','gemm-0.75'],
 'load':['users-1','users-2','users-4','users-8','users-4-slots-1'],
 'vision':['text','image-224','image-448','image-896','video-2','video-4','video-8'],
}
MINI_TAGS=['native-f16-448','native-f16-336','native-f16-compact','native-q8-compact','native-q4-compact','native-q4-t8','native-q8-color','native-q8-video4','native-q8-color-strict']
MINI_LABELS=['F16视觉 / 448面积预算 / 标准词 / 1帧','F16视觉 / 336 / 标准词 / 1帧',
             'F16视觉 / 336 / 短词 / 1帧','混合Q8视觉 / 336 / 短词 / 1帧',
             '混合Q4视觉 / 336 / 短词 / 1帧','混合Q4视觉 / 336 / 短词 / 8线程',
             '混合Q8视觉 / 224 / 颜色问答 / 1帧','混合Q8视觉 / 336 / 短词 / 4帧','混合Q8视觉 / 224 / 颜色问答 / 严格指令']

def main():
    O.mkdir(exist_ok=True)
    (O/'figures').mkdir(exist_ok=True)
    original=read(R/'requests.jsonl')
    mini=read(R/'minicpm_native'/'requests.jsonl')
    groups=defaultdict(list)
    for r in original:
        if r['phase']=='measured' and r['stage']!='pilot':
            groups[r['stage'],r['config']['name']].append(r)
    for r in mini:
        if r['phase']=='measured':
            groups['minicpm_native',r['config']['name']].append(r)
    summaries=[summarize(stage,name,rs) for (stage,name),rs in groups.items()]
    lookup={(s['stage'],s['configuration']):s for s in summaries}
    (O/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    save_csv(O/'summary.csv',summaries)
    def table(stage,names=None,labels=None,full=False):
        names=names or LEVELS[stage]
        result=['| 配置 | 输入 token 均值 | 首 token 计算均值 ms | 95% CI ms | p95 ms | 有效/尝试 |'+(' 生成结束计算均值 ms | 严格单字母正确/总数 |' if full else ''),
                '|---|---:|---:|---:|---:|---:|'+('---:|---:|' if full else '')]
        for index,name in enumerate(names):
            s=lookup[stage,name]
            label=labels[index] if labels else name
            text=f'| {label} | {s["mean_input_tokens"]:.2f} | **{s["mean_ms"]:.2f}** | {s["ci95_low_ms"]:.2f}–{s["ci95_high_ms"]:.2f} | {s["p95_ms"]:.2f} | {s["valid"]}/{s["attempted"]} |'
            if full:
                text+=f' {s["mean_full_answer_compute_ms"]:.2f} | {s["answer_correct"]}/{s["answer_total"]} |'
            result.append(text)
        return '\n'.join(result)
    def plot(stage,names):
        ss=[lookup[stage,n] for n in names]
        labels=names if stage!='minicpm_native' else MINI_TAGS
        fig,ax=plt.subplots(figsize=(max(8,len(ss)*1.25),4.8))
        vals=[s['mean_ms'] for s in ss]
        err=[[max(0,s['mean_ms']-s['ci95_low_ms']) for s in ss],[max(0,s['ci95_high_ms']-s['mean_ms']) for s in ss]]
        ax.bar(range(len(ss)),vals,yerr=err,capsize=4,color='#347cb4')
        ax.set_xticks(range(len(ss)),labels,rotation=28,ha='right')
        ax.set_ylabel('Server computation TTFT (ms); mean and 95% CI')
        ax.set_title(stage+' | RTX 3070 | no prefix reuse')
        if stage=='minicpm_native':
            ax.axhline(140,color='#bd3d3d',linestyle='--',label='140 ms target')
            ax.legend()
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(O/'figures'/f'{stage}.png',dpi=150)
        plt.close(fig)
    for stage,names in LEVELS.items():
        plot(stage,names)
    stairs=sorted(n for stage,n in groups if stage=='staircase')
    plot('staircase',stairs)
    mnames=[next(n for stage,n in groups if stage=='minicpm_native' and groups[stage,n][0]['config']['quantization']==tag) for tag in MINI_TAGS]
    plot('minicpm_native',mnames)
    # Paired mean differences, with each source question appearing exactly once per condition.
    effects=[]
    def paired(stage,name,reference):
        ref={r['sample_index']:r for r in groups[stage,reference]}
        deltas=[metric(r)-metric(ref[r['sample_index']]) for r in groups[stage,name]]
        lo,hi=ci(deltas)
        row={'stage':stage,'configuration':name,'reference':reference,'paired_n':len(deltas),
             'mean_delta_ms':float(np.mean(deltas)),'ci95_low_ms':lo,'ci95_high_ms':hi}
        effects.append(row)
        return row
    bases={'models':'smol500-q8','quant_long':'lfm350-q4','inputs_exact':'squad-total-128','compute':'gemm-0','load':'users-1','vision':'text'}
    for stage,base in bases.items():
        for name in LEVELS[stage]:
            if name!=base:
                reference=name.replace('synthetic','squad') if stage=='inputs_exact' and name.startswith('synthetic') else base
                paired(stage,name,reference)
    for a,b in zip(stairs,stairs[1:]):
        paired('staircase',b,a)
    for target in [3,4]:
        paired('minicpm_native',mnames[target],mnames[2])
    save_csv(O/'paired-effects.csv',effects)
    # Report validation never silently drops failed observations or retries them.
    issues=[]
    for (stage,name),rs in groups.items():
        if len(rs)!=48 or len({r['sample_index'] for r in rs})!=48 or Counter(r['block'] for r in rs)!={0:24,1:24}:
            issues.append('Denominator/block mismatch: '+stage+'/'+name)
        if any(r.get('error') or r['timings'].get('cache_n')!=0 or metric(r)<=0 for r in rs):
            issues.append('Failed/cache/timing record: '+stage+'/'+name)
        if stage=='minicpm_native' and any(abs(r['compute_first_token_ms']-metric(r))>1e-6 for r in rs):
            issues.append('First-token versus final prompt timer mismatch: '+name)
    for path in (R/'minicpm_native').glob('native-*-backend.json'):
        evidence='\n'.join(json.loads(path.read_text())['gpu_evidence'])
        if not re.search(r'offloaded (\d+)/\1 layers',evidence) or 'CLIP using CUDA0 backend' not in evidence:
            issues.append('GPU execution evidence missing: '+path.name)
    validation={'passed':not issues,'issues':issues,'formal_configurations':len(groups),'formal_requests':sum(map(len,groups.values())),
                'minicpm_formal_requests':sum(r['phase']=='measured' for r in mini),'minicpm_pilot_requests':sum(r['phase']=='pilot' for r in mini),
                'minicpm_warmup_requests':sum(r['phase']=='warmup' for r in mini),'metric':'server timings.prompt_ms; first sampled token inclusive',
                'first_token_and_final_prompt_timers_match':not any('mismatch' in x for x in issues),'max_cached_tokens':0}
    (O/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    assert not issues,issues
    h=read(R/'history'/'minicpm-native.jsonl')
    hist=[]
    for mode in ['single','four']:
        rs=[r for r in h if r['condition']==mode]
        values=[r['compute_ttft_ms'] for r in rs]
        hist.append({'condition':mode,'n':len(rs),'mean_ms':float(np.mean(values)),
                     'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),
                     'mean_answer_ms':float(np.mean([r['compute_answer_ms'] for r in rs])),
                     'correct':sum(r['correct'] for r in rs),'first_observation_ms':values[0],
                     'drop_first_sensitivity_mean_ms':float(np.mean(values[1:]))})
    save_csv(R/'history'/'minicpm-summary.csv',hist)
    ll=read(R/'history'/'llava-cached-answer.jsonl')
    ls=[]
    for name in sorted({r['condition'] for r in ll}):
        rs=[r for r in ll if r['condition']==name]
        ls.append({'condition':name,'n':len(rs),'answered':sum(r['status']=='answered' for r in rs),
                   'mean_cached_answer_ms':float(np.mean([r['cached_answer_ms'] for r in rs])),
                   'p95_cached_answer_ms':float(np.percentile([r['cached_answer_ms'] for r in rs],95)),
                   'eligible_full_model_ttft':False})
    save_csv(R/'history'/'llava-cached-summary.csv',ls)
    grace=json.loads((R/'history'/'grace-codec.json').read_text())
    def section(title,body,stage=None):
        return '\n## '+title+'\n\n'+body+'\n'+(f'\n![{stage}](../results/compute_only/figures/{stage}.png)\n' if stage else '')
    best=min((s for s in summaries if s['stage']=='models'),key=lambda s:s['mean_ms'])
    first,last=lookup['staircase',stairs[0]],lookup['staircase',stairs[-1]]
    doc=f'''# 大模型推理首 token 纯计算时延实测报告

日期：2026-10-13（上海时间）。本版主表统一采用模型运行时内部计算计时，移除通信与 Testbed 分段相加口径。新增 **9 个完整 MiniCPM-o 2.6 配置 × 48 条 = 432 条**正式样本；原小模型实验 **46 × 48 = 2208 条**。合计 **55 个配置、2640 条正式请求**，另保留预热与预试验。历史结果单列，不冒充本次受控实验。[验收记录](../results/compute_only/validation.json)。

MiniCPM 的 140 ms 目标指 **首 token 计算 TTFT**。完整视觉+语言模型的一帧合成颜色问答在明确单字母指令下为 **{lookup['minicpm_native',mnames[8]]['mean_ms']:.2f} ms**，严格答对 **{lookup['minicpm_native',mnames[8]]['answer_correct']}/48**；更短但指令服从较差的原提示词为 {lookup['minicpm_native',mnames[6]]['mean_ms']:.2f} ms，不能只按速度推荐。真实 StreamingBench 单帧、短提示词、混合 Q8 视觉组件约 **{lookup['minicpm_native',mnames[3]]['mean_ms']:.2f} ms**。两者输入、任务不同，不能互换。所有 MiniCPM 请求都重新执行视觉编码与语言模型，没有拆为两段部署或复用视觉特征。

## 计时边界和复现条件

主指标为 llama.cpp 的 `timings.prompt_ms`：从服务端槽位开始处理输入，到首次采样 token 完成。当前 b11577 源码确认，视觉编码和语言 prefill 均在该计时区间；第一个 token 从最终 prefill logits 采样，不另加一次完整 decode。它是运行时执行的墙钟时延，包含执行期间的 CPU/GPU 调度、同步及模型内部处理，**不是纯 CUDA 内核用时之和**。不含 HTTP 往返、槽位之前的排队、权重加载、客户端 JPEG 准备及视频解码/采帧；部分输入解析也在计时外。并发时仍可能包含批处理和其他执行等待，不能视为每请求独占 GPU 的 FLOP 耗时。

完整答案计算辅助指标为 `prompt_ms + predicted_ms`；用户目标只依据首 token。原客户端 HTTP 计时及原始数据保留在[旧口径报告](request-latency-archive-zh.md)和原始 JSONL，不混入本版主表。源码依据：[槽位处理与首 token](https://github.com/ggml-org/llama.cpp/blob/b11577/tools/server/server-context.cpp)、[计时状态](https://github.com/ggml-org/llama.cpp/blob/b11577/tools/server/server-common.h)。历史 b11516 同类原生 `prompt_ms` 单列，运行时/后端不同，不能把新旧差异全部归因量化。

RTX 3070 8 GB，i5-13600K，约 32 GiB RAM，Windows 11；本次 b11577 CUDA 12.4。显卡既有功率上限 154 W，本实验未修改功率或频率。新 MiniCPM 所有语言层和视觉编码器实际使用 CUDA，单槽 4096 context，batch 2048、ubatch 512，4 CPU 线程（8线程档单列）；语言权重统一原作者 Q4_K_M。每配置两块各 24 条，原8档的块内配置顺序重新随机；严格颜色指令档为发现输出截断后另做两块补测，未随机夹入原8档。每块每配置另做 8 条不重叠预热。预试验 ID 64–71、正式 0–47、预热 112–119，温度 0、种子 20261013、`cache_prompt=false`，实测 `cache_n=0`。新 MiniCPM 允许 EOS、最多 4 token；原小模型固定生成 32 token。故生成结束耗时不能跨输出预算直接比较。

均值 CI 为固定种子 2000 次 bootstrap；并发主表按完整请求波次重采样。少数块/少数波次不足以估计长期系统漂移，p95 也仅是样本分位数。所有正式尝试均保留，未按时延筛样。量化的质量影响用同一 48 题配对小样本检查，不能当完整精度评测。
'''
    doc+=section('MiniCPM-o 2.6：完整模型和 140 ms 目标',table('minicpm_native',mnames,MINI_LABELS,True)+'''

前 6 档及第8档四帧使用同一组 StreamingBench 问题，单帧使用问题时刻最后一帧，四帧使用对应四帧。分辨率列为 **面积上限的平方根**，不是强制正方形；保持宽高比并向下对齐 28，例如宽屏图像的 336 档通常为 448×252。模型处理器的缩放/切片导致视觉 token 数可能不随面积连续减少，实际总输入 token 如表所示。

颜色任务为确定性生成的 224×224 单色矩形场景，位置、颜色变化，原提示词 `Color? A red B blue C green D yellow. Letter:`；严格档再加 `Reply with exactly one letter, no other text.`。它验证完整视觉路径的低时延，不能代表视频理解质量或复杂真实任务达到 140 ms。真实问答短词档仅删去冗余系统说明，保留问题和全部选项。

正确率列采用严格A–D单字母解析（允许句点），额外解释或未完成文本计错。因此该列同时反映答题正确与格式服从，不能把紧凑提示词的格式失败全解释为视觉能力下降。原颜色短词有26/48条只生成 `The color of the`，4 token达到上限；这些TTFT仍有效，但未完成答案。短词真实问答也常输出字母加解释。本表“生成结束计算”是到EOS或4 token上限的耗时，**输出截断时不是完整语义答案耗时**。严格颜色补测48条全部输出单字母并EOS结束。汇总CSV另存开头选项答对数和达到输出预算数，不隐藏格式失败。

低精度方案从原 F16 视觉组件量化得到混合 Q8_0 或 Q4_K_M，语言模型仍为 Q4_K_M。分别有 27/455、163/455 张量发生量化 fallback；实际文件平均约 11.44、10.15 BPW，**不是整模型均匀 8 bit / 4 bit**。没有把 Q4 语言权重再次量化成 Q2 并宣称等同原始 Q2。量化组件缩小显存占用，但本次真实短词档的时延收益较小，短提示词与输入切片变化贡献更大。[权重与派生哈希](../data/minicpm-artifacts.json)、[逐请求数据](../results/minicpm_native/requests.jsonl)、[GPU 后端证据](../results/minicpm_native/native-q8-compact-backend.json)。
''','minicpm_native')
    doc+=section('模型参数量与量化',f'短文本模型比较观察最低为 **{best["configuration"]}，{best["mean_ms"]:.2f} ms**，作为本轮文本因素与阶梯的基线选择。它是已测候选的观察最低，非全局最优。\n\n'+table('models')+'\n\nLFM2 350M 与 1.2B 为同系列；跨 SmolVLM2 比较同时改变架构、tokenizer、模板，不能只归因参数量。新增 MiniCPM-o 2.6 是用户指定的额外较大模型，超过初始“优先 7B 以下”范围；不把它混入同参数量比较。LFM2 为自定义开放权重许可证，SmolVLM2 为 Apache-2.0。','models')
    doc+=section('固定长文本的精度对照',table('quant_long')+'\n\n同一 LFM2-350M、同一批输入，2048 请求上下文预算。本次没有证据表明更低位宽必然降低 prefill / TTFT；请结合配对差值 CI。','quant_long')
    doc+=section('文本长度和数据集',table('inputs_exact')+'\n\n这些档位为包括模板的**实际总输入 token 数**，逐请求核验精确匹配。等长 SQuAD 与合成文本对照关注内容差异；原 SQuAD 被裁剪/重复，不作为标准准确率评测。原 inputs 阶段 token ID 转文本再编码导致部分空格合并，名义预算不等长；旧记录及实际 token 数保留在汇总中，严格结论使用 inputs_exact。','inputs_exact')
    doc+=section('单卡可用算力代理',table('compute')+'\n\n固定 SmolVLM2、SQuAD 和单用户。独立 CUDA FP16 2048² GEMM 进程以 40 ms 周期控制活动占空比，保留空闲驻留和无进程对照。实测活动比例约 0 / 25.5 / 50.1 / 74.8%。这只模拟资源竞争，不能称为 GPU 使用率硬限制、SM 配额或等价另一张显卡；50% 不等于算力减半。','compute')
    doc+=section('多用户负载',table('load')+'\n\n多用户共享一份权重，按同步波次发出请求，主对照固定 4 槽；32 token 全部完成后发下一波。单槽档额外改变了服务容量。主表排除进入槽位之前的排队，故仅观察运行中的计算竞争，不能代表用户体验总延迟，也不能据此断言排队消失。','load')
    doc+=section('文本、图像与多帧视频任务',table('vision')+'\n\n固定 SmolVLM2；48 个确定性形状/运动场景。视频为按顺序输入的 2 fps 抽帧图片序列。分辨率可能受处理器缩放、切片上限约束，不预设视觉 token 线性增加。该组测任务输入开销，不是公开真实视频基准准确率；新增 MiniCPM 另用真实 StreamingBench。','vision')
    stair_labels=['0：SmolVLM2 Q8，短文本单用户','1：仅切换 LFM2-350M Q8','2：仅切换 Q4_K_M','3：上下文预算 64→512','4：上下文预算 512→2048','5：仅切换 LFM2-1.2B Q4','6：增加 50% GEMM 竞争','7：并发用户 1→4','8：并发用户 4→8']
    reversals=[(a,b) for a,b in zip(stairs,stairs[1:]) if lookup['staircase',b]['mean_ms']<lookup['staircase',a]['mean_ms']]
    doc+=section('逐项累加的计算时延阶梯',table('staircase',stairs,stair_labels)+f'\n\n独立重测起点 {first["mean_ms"]:.2f} ms，终点 {last["mean_ms"]:.2f} ms。均值回落次数 **{len(reversals)}**，全部保留；阶梯是实际累计配置路径，不能通过对结果排序来伪造单调因果。配对差值见 [CSV](../results/compute_only/paired-effects.csv)。','staircase')
    doc+='\n## 其他会话：原生 MiniCPM 历史计算记录\n\n源自“测试 MiniCPM-o 2.6 推理准确率”会话对应的公开原生结果；直接读取模型 `timings`，不用请求总耗时，也不相加任何分段网络实验。\n\n| 原生输入 | 样本 | 首 token 计算均值 ms | p95 ms | 完整答案计算均值 ms | 正确/总数 |\n|---|---:|---:|---:|---:|---:|\n'
    for s in hist:
        doc+=f'| {s["condition"]} | {s["n"]} | **{s["mean_ms"]:.2f}** | {s["p95_ms"]:.2f} | {s["mean_answer_ms"]:.2f} | {s["correct"]}/{s["n"]} |\n'
    doc+='\n历史环境是 b11516 / Vulkan、语言 Q4_K_M + 视觉 F16、输入面积≤448²、最多16输出 token、种子20261009。未发现明确独立预热协议，因此保留每组全部125条，并标为历史非严格热态对照；不能以新旧差值证明单一优化收益。首条分别为 '+', '.join(f'{s["first_observation_ms"]:.2f}' for s in hist)+' ms；仅去首条的敏感性均值分别为 '+', '.join(f'{s["drop_first_sensitivity_mean_ms"]:.2f}' for s in hist)+' ms，不替换主统计。[数值原始摘录](../results/history/minicpm-native.jsonl)、[源文件与SHA256](../results/history/provenance.json)。\n'
    doc+='\n## 其他会话：LLaVA 与非 LLM 计算记录\n\nLLaVA-Video-7B-Qwen2 / NF4 双重量化 / BF16计算，WSL RTX3070：已保存 48 条件×495题=23760条。计时为 CUDA 同步包围的**缓存视觉特征及共享前缀之后的语言答案调用**，含问题后缀处理、生成和文本解码；没有内部首 token 时间。下面只记录已有纯本地答案计算，明确不能充当完整视觉+语言 TTFT，亦不与 MiniCPM 拆段相加。\n\n| 历史条件 | 样本 | 缓存后答案计算均值 ms | p95 ms |\n|---|---:|---:|---:|\n'
    for name in ['text_only','native_center_1','native_center_4','uniform_16','uniform_64']:
        s=next((s for s in ls if s['condition']==name),None)
        if s:
            doc+=f'| {name} | {s["n"]} | {s["mean_cached_answer_ms"]:.2f} | {s["p95_cached_answer_ms"]:.2f} |\n'
    doc+='\n其余48条件详见[历史缓存计算汇总](../results/history/llava-cached-summary.csv)。视觉缓存与前缀构造原计时还夹带磁盘读写、哈希/准备操作，因此没有拼成所谓完整模型纯计算时间。Liquid_V1_7B 没有可核验的本地逐题计算时延；它与本报告 LiquidAI LFM2 不是同一模型。DCVC 未发现已完成的对应计算实测，这两项明确留缺。\n\nGRACE 会话另有 CUDA 同步的编解码微基准，属于神经视频编解码计算，**不属于大模型 TTFT**：256×192，batch1，同帧串行，预热5次后20次，包含CPU/GPU拷贝；只在本节附录保留，不与LLM排序。\n\n| 计算操作 | 样本 | 均值 ms | p95 ms |\n|---|---:|---:|---:|\n'
    for k,v in grace['metrics'].items():
        doc+=f'| {k} | 20 | {v["mean"]:.3f} | {v["p95"]:.3f} |\n'
    doc+='\n[历史来源与缺失项](../results/history/provenance.json)保留公开仓库固定 commit、逐文件 SHA256；未上传其他会话的私人转录、视频、权重和服务日志。\n\n## 数据与复现\n\n[计算汇总CSV](../results/compute_only/summary.csv)、[配对效应](../results/compute_only/paired-effects.csv)、[原小模型逐请求](../results/requests.jsonl)、[新MiniCPM逐请求](../results/minicpm_native/requests.jsonl)、[原模型清单](../data/artifacts.json)、[MiniCPM清单](../data/minicpm-artifacts.json)。运行 `python scripts/build_compute_report.py` 从已公开的数值记录重建报告和图；运行 `python scripts/run_minicpm_sweep.py --assets <本地原生素材目录>` 重测新MiniCPM。素材需由公开 StreamingBench 来源及历史原生抽帧流程准备，权重按清单下载，视觉组件按清单命令从F16派生；不分发完整数据集或权重。\n\n单卡桌面、单版本运行时和小样本准确率限制结论外推；参数量不是FLOPs，低精度不保证降低TTFT，处理器切片会改变视觉token，硬件竞争代理不代表另一张GPU。\n'
    (ROOT/'docs'/'results-zh.md').write_text(doc,encoding='utf-8')
    readme='''# TTFT 推理计算时延实验

中文完整报告：**[大模型推理首 token 纯计算时延实测报告](docs/results-zh.md)**。

RTX 3070 8 GB；原小模型46配置×48条，加完整MiniCPM-o 2.6的9配置×48条，共55配置、2640条正式请求。每档另做预热、固定种子、关闭并检查前缀缓存。主指标统一为模型内部 `timings.prompt_ms`，涵盖视觉编码到首次token采样；排除HTTP、权重加载及进入槽位之前的排队。这是模型执行墙钟，非CUDA内核时间总和。

## MiniCPM 完整模型

'''+table('minicpm_native',mnames,MINI_LABELS,True)+'''

语言权重统一原作者Q4_K_M；视觉混合Q8/Q4有张量fallback，非均匀位宽。颜色问答严格指令档接近140 ms目标，真实视频问答单列，不能互换。最短词档出现输出截断，严格格式正确率较低。每次完整执行视觉和语言模型，无特征/前缀复用。MiniCPM-o为额外指定的较大模型，与原350M/500M/1.2B测试范围分别说明。

![MiniCPM TTFT](results/compute_only/figures/minicpm_native.png)

## 控制变量和证据

报告包含模型/量化、严格匹配文本长度与数据集、图像分辨率/帧数、GPU竞争代理、多用户并发以及逐项累加阶梯。保留均值回落和失败/缺失项，不按时延筛样。其他会话的原生MiniCPM历史计算已纳入；LLaVA缓存后答案计算和GRACE编解码仅列在标明口径的附录，不混作完整TTFT。没有Testbed分段相加表。

- [计算汇总CSV](results/compute_only/summary.csv)、[配对差值](results/compute_only/paired-effects.csv)、[验收](results/compute_only/validation.json)
- [新MiniCPM原始逐请求](results/minicpm_native/requests.jsonl)、[原小模型原始逐请求](results/requests.jsonl)
- [历史纯计算摘录与来源](results/history/provenance.json)、[权重及派生量化清单](data/minicpm-artifacts.json)
- [计时方法](docs/methodology.md)、[旧HTTP请求口径归档](docs/request-latency-archive-zh.md)

```powershell
python scripts/build_compute_report.py
python scripts/verify_results.py
python scripts/run_minicpm_sweep.py --assets <local-native-assets-directory>
```

只发布脚本、公开来源文本及数值证据，不发布模型权重、运行时二进制、完整视频或本地私人日志。SQuAD来源见[data说明](data/README.md)；硬件和原模型来源见[environment](data/environment.json)与[artifacts](data/artifacts.json)。原有GPU功率上限154W未修改。多个用户共享一份权重；GEMM活动占空比仅为资源竞争代理，不能等同另一张显卡。
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')
    print(json.dumps(validation,indent=2))
    print('MiniCPM:',[(MINI_TAGS[i],round(lookup['minicpm_native',n]['mean_ms'],3),lookup['minicpm_native',n]['answer_correct']) for i,n in enumerate(mnames)])

if __name__=='__main__':
    main()
