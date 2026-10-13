"""Generate a concise Chinese evidence report from actual measurements."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from report import ci

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'
summary=json.loads((R/'summary.json').read_text())
lookup={(s['stage'],s['configuration']):s for s in summary}
validation=json.loads((R/'validation.json').read_text())
raw=[json.loads(x) for x in (R/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
effects=list(csv.DictReader((R/'paired-effects.csv').open()))
final=[s for s in summary if s['stage']!='pilot']
best=min(final,key=lambda s:s['mean_ms'])

def table(stage,names,labels=None):
    out=['| 配置 | 实际输入 token 均值 | TTFT 均值 ms | 95% CI ms | 中位数 ms | p95 ms | 有效/尝试 |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        s=lookup[(stage,name)]
        label=(labels or {}).get(name,name)
        out.append(f'| {label} | {s["mean_input_tokens"]:.2f} | {s["mean_ms"]:.2f} | {s["ci95_low_ms"]:.2f}–{s["ci95_high_ms"]:.2f} | {s["median_ms"]:.2f} | {s["p95_ms"]:.2f} | {s["valid"]}/{s["attempted"]} |')
    return '\n'.join(out)

names=sorted(s['configuration'] for s in summary if s['stage']=='staircase')
labels={
 'step-0-minimum':'0：SmolVLM2-500M Q8，短文本，单用户',
 'step-1-model-lfm350-q8':'1：仅切换为 LFM2-350M Q8',
 'step-2-precision-q4':'2：仅切换为 Q4_K_M',
 'step-3-context512':'3：上下文预算 64 → 512',
 'step-4-context2048':'4：上下文预算 512 → 2048',
 'step-5-larger':'5：仅切换为 LFM2-1.2B Q4',
 'step-6-contention':'6：增加 50% GEMM 占空比竞争',
 'step-7-users4':'7：单用户 → 4 用户',
 'step-8-users8':'8：4 用户 → 8 用户',
}
byconfig=defaultdict(dict)
for r in raw:
    if r['stage']=='staircase' and r['phase']=='measured' and not r['error']:
        byconfig[r['config']['name']][r['sample_index']]=r
adjacent=[]
for a,b in zip(names,names[1:]):
    ids=sorted(set(byconfig[a])&set(byconfig[b]))
    delta=[byconfig[b][i]['ttft_ms']-byconfig[a][i]['ttft_ms'] for i in ids]
    lo,hi=ci(delta)
    adjacent.append({'from':a,'to':b,'paired_n':len(ids),'mean_delta_ms':sum(delta)/len(delta),
        'ci95_low_ms':lo,'ci95_high_ms':hi,'ci_method':'paired request bootstrap; concurrency correlation not modeled'})
with (R/'staircase-adjacent-effects.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(adjacent[0]))
    w.writeheader()
    w.writerows(adjacent)
start=lookup[('staircase',names[0])]['mean_ms']
end=lookup[('staircase',names[-1])]['mean_ms']
telemetry=json.loads((R/'telemetry-summary.json').read_text())
maxmem=max(s['memory.used_max'] for s in telemetry if s['memory.used_max'] is not None)
text=f"""# RTX 3070 小模型推理 TTFT 实测报告

测量日期：2026-10-13（上海时间）。已完成 **{len(final)} 个正式配置 × 48 条有效样本 = {sum(s['valid'] for s in final)} 条正式测量**；另有 40 条预试验及 776 条预热，总计 **{validation['all_requests_including_warmup']} 次请求**。失败 0，实际缓存复用 token 最大值 0。所有正式配置均覆盖 48 个不同且配对的样本 ID。见 [验收记录](../results/validation.json)。

正式配置中的观察最低均值为 **{best['model']}，{best['mean_ms']:.2f} ms**。累计阶梯独立重测的起点为 {start:.2f} ms，终点为 {end:.2f} ms，整体约 {end/start:.1f} 倍。除累计叠加的大模型、竞争和高并发末端两档外，其他正式配置的均值均低于 600 ms。置信区间重叠时，均值排序不是确定的性能排名。

## 计时和控制

TTFT 为本机 HTTP 请求发出前，到首个非空生成文本 SSE delta 的单调时钟耗时。响应头、角色事件、空事件和心跳不计为首 token。逐字节读取 SSE，避免默认缓冲延迟。首个可见文本可能晚于内部首个生成 token，此处不是纯 GPU 内核时间。

计时包含本机传输、服务端视觉处理、prefill、调度、排队和首文本输出；排除权重加载、预热、客户端数据准备及视频解码/采帧。每档两块各 24 条，配置顺序每块重新随机化，每块每档另做 8 条预热。预试验、预热、正式输入 ID 分离；温度 0、固定种子、32 token 输出预算、禁用前缀复用。所有请求均保留。

服务器 prompt_ms 与客户端 TTFT 分别保留；差值含排队、调度、预处理和传输，**不是单独测得的排队时间**。并发条件均值的 CI 按完整波次重采样；高并发波次数少，CI 是探索性估计。

硬件为 RTX 3070 8 GB、i5-13600K、约 32 GiB 内存，Windows 11，llama.cpp b11577 / CUDA 12.4。显卡原有功率上限 **154 W**，驱动默认值为 220 W，实验未改功率或频率。遥测最高采样显存约 **{maxmem:.0f} MiB**，不保证捕获瞬时峰值。日志确认全部语言模型层及视觉编码器使用 CUDA；多个用户共享一份模型。

## 模型规模与精度

短文本使用同一组 SQuAD 问题和 64 个请求上下文 token 预算。LFM2 350M/1.2B 是同一系列；SmolVLM2 涉及架构、tokenizer、模板和训练差异，跨系列差异不能只归因参数量。

{table('models',['lfm350-q4','lfm350-q8','lfm350-f16','lfm1200-q4','smol500-q8'])}

![模型与精度](../results/figures/models.png)

固定 LFM2-350M、同一组问题及 2048 个请求上下文 token 预算，再单独改变精度：

{table('quant_long',['lfm350-q4','lfm350-q8','lfm350-f16'])}

**本次未观察到量化带来稳定 TTFT 优势**，短、长输入的精度差异均较小。Q4_K_M 是混合张量量化，不是所有权重均为 4 bit。权重文件更小，不保证 warm-server prefill/TTFT 更低。LFM2 使用自定义开放权重许可证，SmolVLM2 使用 Apache-2.0，许可证性质不同。

## 文本长度与数据集

严格对照包含模板、问题和上下文的全部输入 token；服务端实际计数逐请求验证为目标值：

{table('inputs_exact',['squad-total-128','squad-total-512','squad-total-1024','squad-total-2048'])}

![严格输入长度](../results/figures/inputs_exact.png)

相同总 token 数下，真实 SQuAD 问答与合成算术/重复句子对照：

{table('inputs_exact',['squad-total-128','synthetic-total-128','squad-total-1024','synthetic-total-1024'])}

"""
for effect in effects:
    if effect['stage']=='inputs_exact' and effect['configuration'].startswith('synthetic'):
        text+=f'- {effect["configuration"]} 相对 {effect["reference"]}：配对均值差 {float(effect["mean_delta_ms"]):.2f} ms，95% CI [{float(effect["ci95_low_ms"]):.2f}, {float(effect["ci95_high_ms"]):.2f}] ms。\n'
text+="""

两种匹配长度对照的 CI 均跨 0，本次没有稳定的独立数据集内容效应。长度影响更清楚。SQuAD 上下文会裁剪或重复，不能作为原始 SQuAD 准确率结果。

原 inputs 阶段发现：重复 token ID 再转回文本，会在再次编码时合并边界空格。标称 synthetic-1024 的实际总输入均值为 958.94，SQuAD 对应值为 1054.19，不能作为严格等长对照。旧记录及实际计数完整保留；inputs_exact 改为先构造重复文本，再校准完整模板后的总 token 数。补测未覆盖或隐藏原数据。

## 单卡可用算力代理

独立 CUDA FP16 2048×2048 GEMM 进程按 40 ms 周期控制活动占空比，不修改显卡设置。0% 档保留同一个驻留 GPU 的空闲进程，另有完全无竞争进程的对照。模型、SQuAD 输入预算和单用户固定。

"""+table('compute',['no-worker','gemm-0','gemm-0.25','gemm-0.5','gemm-0.75'])+"""

![GPU 竞争](../results/figures/compute.png)

独立校准的活动时间比例约 0%、25.5%、50.1%、74.8%。它不是 SM 配额、利用率上限或可换算的另一张显卡；调度、带宽、功率和频率均可能参与竞争。50% 占空比不等于算力减半。[遥测汇总](../results/telemetry-summary.csv) 覆盖配置窗口的预热及测量。

## 并发用户

同一模型由多个用户共享，使用同步请求波次和固定 32 token 输出预算，主对照固定 4 个服务器槽位。全部请求结束后才发下一波，不是泊松到达或生产流量模拟。

"""+table('load',['users-1','users-2','users-4','users-8','users-4-slots-1'])+"""

![用户并发](../results/figures/load.png)

8 用户超过 4 槽位，包含排队。额外一槽位条件明确改变服务器容量，不混同于固定四槽位的用户数量对照。

## 文本、图像、视频任务

全部使用 SmolVLM2-500M Q8。48 个固定种子的图形场景变化形状、颜色和位置；视频按运动规则以 2 fps 取有序帧并带时间戳输入。

"""+table('vision',['text','image-224','image-448','image-896','video-2','video-4','video-8'])+"""

![视觉任务](../results/figures/vision.png)

224 与 448 均为 153 个总 token，896 为 352，源分辨率与视觉 token 并非线性关系。帧数同时增加视觉 token 和时间戳文本，帧数对照保持同一问题；单图与视频问题语义不同，模态对照包含任务自然变化。

视觉输入是可复现的合成场景，覆盖图像/多帧视频推理路径，**不是真实图像、真实视频或流式视频 benchmark**。视频以有序图像帧提交，非原生压缩视频 API；解码/采帧不在 TTFT 内。不评价答案质量。

## 累计阶梯

这是独立测量，按计划只改变下一项配置。系列切换含 tokenizer 等固有模型差异；精度单列一步，避免与模型规模同时改变。

"""+table('staircase',names,labels)+"""

![累计阶梯](../results/figures/staircase.png)

"""
for delta in adjacent:
    if delta['mean_delta_ms']<0:
        text+=f'保留的回落：{delta["from"]} → {delta["to"]}，均值变化 **{delta["mean_delta_ms"]:.2f} ms**，配对 95% CI [{delta["ci95_low_ms"]:.2f}, {delta["ci95_high_ms"]:.2f}] ms。\n\n'
text+="""
整体上升，但**每一阶并非严格单调**；精度切换的小幅回落及不确定性完整保留。末端约 1.30 s 是长输入、大模型、竞争和并发的组合结果，不能归因单一变量。用单变量对照理解各项作用。相邻差异表对并发相关性未建模，仅作描述；每档并发均值 CI 已按波次重采样。

## 证据与复现

单台 Windows 桌面、单 GPU、单引擎、现有 154 W 上限仍有显示及操作系统噪声，不能推广为显卡性能比例。候选均值选择有选择偏差；预试验最低与正式最低不同，独立阶梯也有波动。这里只有观察最小值，没有全局最优证明。

结果只衡量延迟，不证明质量等价、答案正确、FLOP 数、吞吐或生产表现。SQuAD 选中记录保留来源、答案、Wikipedia 标题链接和数据哈希，保持 CC BY-SA 4.0。权重、二进制、完整第三方数据集、私人日志和凭据未上传。

- [方法](methodology.md)；[验收](../results/validation.json)
- [逐请求数据](../results/requests.jsonl)；[汇总](../results/summary.csv)；[配对差异](../results/paired-effects.csv)；[相邻阶梯差异](../results/staircase-adjacent-effects.csv)
- [模型/运行时哈希](../data/artifacts.json)；[环境](../data/environment.json)；[遥测](../results/telemetry-summary.csv)

复现命令：

    python -m pip install -r requirements.txt
    python scripts/prepare.py
    python scripts/capture_environment.py
    python scripts/calibrate_compute.py
    python scripts/benchmark.py --stage all
    python scripts/verify_results.py
    python scripts/summarize_telemetry.py
    python scripts/write_chinese_report.py

竞争进程需要 CUDA 版 PyTorch，本机为 2.6.0+cu124。完成块会自动跳过；新的独立重复实验请使用新结果目录。维护者可加 --publish 每阶段更新 README，复现者可以只在本地运行。
"""
(ROOT/'docs'/'results-zh.md').write_text(text,encoding='utf-8')
print('Chinese report written; final configurations',len(final))
