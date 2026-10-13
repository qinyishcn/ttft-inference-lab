# TTFT 推理计算时延实验

中文完整报告：**[大模型推理首 token 纯计算时延实测报告](docs/results-zh.md)**。

RTX 3070 8 GB；原小模型46配置×48条，加完整MiniCPM-o 2.6的9配置×48条，共55配置、2640条正式请求。每档另做预热、固定种子、关闭并检查前缀缓存。主指标统一为模型内部 `timings.prompt_ms`，涵盖视觉编码到首次token采样；排除HTTP、权重加载及进入槽位之前的排队。这是模型执行墙钟，非CUDA内核时间总和。

## MiniCPM 完整模型

| 配置 | 输入 token 均值 | 首 token 计算均值 ms | 95% CI ms | p95 ms | 有效/尝试 | 生成结束计算均值 ms | 严格单字母正确/总数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F16视觉 / 448面积预算 / 标准词 / 1帧 | 322.02 | **390.98** | 389.31–392.80 | 397.62 | 48/48 | 405.31 | 39/48 |
| F16视觉 / 336 / 标准词 / 1帧 | 190.02 | **162.07** | 160.63–163.57 | 172.85 | 48/48 | 176.46 | 38/48 |
| F16视觉 / 336 / 短词 / 1帧 | 146.04 | **149.65** | 148.32–150.95 | 157.48 | 48/48 | 184.87 | 14/48 |
| 混合Q8视觉 / 336 / 短词 / 1帧 | 146.04 | **147.84** | 146.51–149.18 | 157.46 | 48/48 | 183.29 | 13/48 |
| 混合Q4视觉 / 336 / 短词 / 1帧 | 146.04 | **146.78** | 145.55–148.08 | 154.80 | 48/48 | 181.53 | 15/48 |
| 混合Q4视觉 / 336 / 短词 / 8线程 | 146.04 | **149.26** | 148.09–150.47 | 157.76 | 48/48 | 184.83 | 15/48 |
| 混合Q8视觉 / 224 / 颜色问答 / 1帧 | 108.00 | **136.92** | 136.51–137.35 | 139.12 | 48/48 | 166.01 | 21/48 |
| 混合Q8视觉 / 336 / 短词 / 4帧 | 344.04 | **482.79** | 481.55–484.09 | 490.63 | 48/48 | 517.26 | 18/48 |
| 混合Q8视觉 / 224 / 颜色问答 / 严格指令 | 118.00 | **140.25** | 139.95–140.56 | 141.78 | 48/48 | 154.60 | 48/48 |

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
