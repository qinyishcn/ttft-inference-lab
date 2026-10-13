# Small-model TTFT inference lab

RTX 3070 (8 GiB), Windows, single-machine controlled experiment. Work in progress; no measured result is claimed until its per-request evidence is committed.

## Experimental contract

- At least **48 measured requests per configuration**, plus separate warm-up. Model loading and cold-start requests are excluded from steady-state TTFT and retained separately.
- Primary metric: monotonic wall-clock elapsed time from immediately before local HTTP request submission to the first **nonempty generated text** SSE delta. Headers, role-only deltas and keepalives do not count. First generated token and first visible text may differ; both are recorded where available.
- Same engine, greedy decoding, fixed seed, disabled prompt/KV prefix reuse, fixed output budget, paired sample IDs, fixed CPU threads and fixed context budget. Preserve failed/missing requests in the denominator.
- Separate one-factor comparisons for model, quantization, input length/dataset, modality/resolution/video frames, GPU contention, and concurrent users. One loaded model serves multiple users; weights need not be replicated.
- Use disjoint pilot data for functional checks, select the fastest **observed eligible configuration** in the 48-request model comparison, and remeasure it in an independent staircase. An observed minimum is not a proof of the global optimum.
- Present measured means, median/p95, sample spread and bootstrap confidence intervals. A staircase is an empirical, cumulative scenario; never discard a nonmonotonic result or conflate sorting with causality.
- GPU competition is a resource-contention proxy, **not another physical GPU or a guaranteed percentage of GPU compute**. Keep existing GPU power/clock settings unchanged. Log telemetry.
- Server prompt-processing timings are distinct from client TTFT. Any remainder includes queueing, preprocessing, scheduling, transport and streaming; it is not a direct measurement of queue time.

## Hardware and candidates

Detected GPU: NVIDIA GeForce RTX 3070, 8192 MiB; NVIDIA driver 610.60. Existing power ceiling: 154 W (driver-reported default 220 W). RAM: approximately 32 GiB. This is an interactive Windows desktop, so residual OS/display contention is a limitation.

Candidates, all below 7B:

| Model | Role | Weights |
|---|---|---|
| LiquidAI LFM2-350M | small text baseline and quantization comparison | Q4_K_M, Q8_0, F16 |
| LiquidAI LFM2-1.2B | larger text model, same family | Q4_K_M |
| HuggingFaceTB SmolVLM2-500M-Video-Instruct | same-model text/image/multi-frame video comparisons | Q8_0 language model + Q8_0 vision projector |

Liquid models are open-weight under the [LFM Open License](https://huggingface.co/LiquidAI/LFM2-350M); they are not described here as OSI-approved open source. SmolVLM2 is [Apache-2.0](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct). Model size alone is not a FLOP count or a quality comparison. No answer-quality conclusion follows from these latency results.

## Data and reproducibility

Text: seeded selection from the public [SQuAD 1.1 development set](https://rajpurkar.github.io/SQuAD-explorer/). Packed/cropped context variants are explicitly labeled and not treated as original SQuAD accuracy evaluation. Synthetic controls isolate token length. Visual controls use reproducible original shape scenes and ordered video frames; they are not a representative real-world vision benchmark. Input manifests and hashes are published; model weights, runtime binaries and full third-party datasets are excluded.

Pinned runtime: [llama.cpp b11577](https://github.com/ggml-org/llama.cpp/releases/tag/b11577), CUDA 12.4 Windows build. See [server API documentation](https://github.com/ggml-org/llama.cpp/blob/b11577/tools/server/README.md). Revisions and SHA-256 hashes will be in `data/artifacts.json` after downloads complete.

When PyTorch 2.6.0+cu124 is already installed, preparation reuses its CUDA 12.4 DLLs and hashes each dependency. Otherwise it downloads the matching llama.cpp CUDA dependency archive. The CUDA GEMM competition experiment requires a CUDA-enabled PyTorch installation (this machine uses 2.6.0+cu124).

```powershell
python -m pip install -r requirements.txt
python scripts/prepare.py
python scripts/capture_environment.py
python scripts/test_measurement.py
python scripts/calibrate_compute.py
python scripts/benchmark.py --stage all --publish
# Or measure one stage: --stage models|inputs|compute|load|vision|staircase
python scripts/report.py
```

## Progress

Hardware, SSE timing and CUDA inference functional checks passed. Model downloads and digest checks are in progress. Measurement stage completion is shown in the generated tables below once available.

This public repository is updated after each completed measurement stage. Raw per-request records remain available alongside summaries, including negative results and limitations.


<!-- RESULTS:START -->
## Measured results

All times below are **milliseconds**, warm requests only. Pilot rows have 8 requests and are excluded from final conclusions. Final configurations target 48 paired requests in two randomized blocks. CI: seeded bootstrap; concurrent conditions resample whole waves. See [methodology](docs/methodology.md), [raw requests](results/requests.jsonl), [summary CSV](results/summary.csv), and [paired differences](results/paired-effects.csv).

### pilot

| Configuration | Valid / attempted | Mean | 95% CI | Median | p95 | Input tokens (mean) |
|---|---:|---:|---:|---:|---:|---:|
| lfm350-f16 | 8 / 8 | 18.97 | 14.75–24.54 | 15.65 | 31.02 | 91.12 |
| lfm350-q8 | 8 / 8 | 33.59 | 28.05–37.66 | 35.65 | 40.53 | 91.12 |
| lfm350-q4 | 8 / 8 | 29.33 | 22.82–35.48 | 31.87 | 40.36 | 91.12 |
| lfm1200-q4 | 8 / 8 | 29.00 | 22.00–35.99 | 28.39 | 40.36 | 91.12 |
| smol500-q8 | 8 / 8 | 30.41 | 24.46–35.13 | 31.44 | 39.26 | 92.50 |

![pilot TTFT](results/figures/pilot.png)

Input resolution and visual tokens may saturate at the model processor limit. Quantization may increase prefill overhead. GPU GEMM duty is not a hardware-equivalence claim. Concurrent results include server queueing and batching. Visual inputs are controlled synthetic scenes, not a real-world vision benchmark. No quality, FLOP-count, production-throughput or cross-GPU conclusion is established.

<!-- RESULTS:END -->
