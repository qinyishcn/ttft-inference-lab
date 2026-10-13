# Small-model TTFT inference lab

RTX 3070 (8 GiB), Windows, single-machine controlled experiment. Work in progress; no measured result is claimed until its per-request evidence is committed.

## Experimental contract

- At least **40 measured requests per configuration**, plus separate warm-up. Model loading and cold-start requests are excluded from steady-state TTFT and retained separately.
- Primary metric: monotonic wall-clock elapsed time from immediately before local HTTP request submission to the first **nonempty generated text** SSE delta. Headers, role-only deltas and keepalives do not count. First generated token and first visible text may differ; both are recorded where available.
- Same engine, greedy decoding, fixed seed, disabled prompt/KV prefix reuse, fixed output budget, paired sample IDs, fixed CPU threads and fixed context budget. Preserve failed/missing requests in the denominator.
- Separate one-factor comparisons for model, quantization, input length/dataset, modality/resolution/video frames, GPU contention, and concurrent users. One loaded model serves multiple users; weights need not be replicated.
- Choose the fastest **observed eligible configuration** using pilot data; confirm it with independent measured requests. An observed minimum is not a proof of the global optimum.
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

```powershell
python -m pip install -r requirements.txt
python scripts/prepare.py
# Measurement and report commands will be committed with the runnable harness.
```

## Progress

1. Hardware inspected; model/runtime downloads started.
2. Measurement harness and pilot pending.
3. Controlled-factor measurements pending.
4. Cumulative staircase and final validation pending.

This public repository is updated after each completed measurement stage. Raw per-request records remain available alongside summaries, including negative results and limitations.
