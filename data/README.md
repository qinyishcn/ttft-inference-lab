# Input and artifact provenance

`artifacts.json` pins model repositories, revisions, files, byte counts and SHA-256 digests; model digests are checked against upstream Hugging Face LFS metadata. Runtime ZIPs are locally hashed and ZIP CRC-checked. Weights and binaries are excluded from this repository.

`squad-selected.json` contains seeded selected records from [SQuAD 1.1 development](https://rajpurkar.github.io/SQuAD-explorer/), created by Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev and Percy Liang. SQuAD is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Its contexts derive from Wikipedia. The included subset retains that license; it is not covered by a software license for our scripts. Cropped/repeated variants are generated at runtime and explicitly labeled; no official accuracy score is reported.

`environment.json` records the measured hardware/software without credentials, account settings, machine name, local paths or network addresses.

Visual scenes are generated entirely by `scripts/inputs.py` with a fixed seed. Pixel hashes, dimensions, sizes and frame order are recorded in each request's input metadata. No image model or externally hosted inference service is used to generate these controls.

The native MiniCPM and small-VLM follow-up questions/frames come from the public [StreamingBench](https://github.com/THUNLP-MT/StreamingBench) subset used by [the earlier native evaluation](https://github.com/qinyishcn/minicpm-o-2_6-streamingbench-eval/tree/7896d6c0540ba411147364a54cf190900ff0e336). Dataset revision, numeric-source hashes and model hashes are recorded in `minicpm-artifacts.json`, `vlm-extension-artifacts.json` and `results/history/provenance.json`. Full videos, original frame caches and weights are excluded. These source data are not relicensed as MIT software.

`erqa-selected.json` contains a fixed48-question single-image subset plus disjoint8-question warmup/pilot subsets of [Embodied Reasoning QA (ERQA)](https://github.com/embodiedreasoning/ERQA/tree/5bba488252c5b345a43335bfb9239b6bf12bd91c), attributed to Embodied Reasoning QA / embodiedreasoning / Google DeepMind under [CC BY4.0](https://creativecommons.org/licenses/by/4.0/). Questions/options/answers retain that license. Only text and image hashes are published; source TFRecord and extracted images remain in the ignored cache. `erqa-provenance.json` retains source Git blob identity, SHA256, deterministic selection rule and multilabel category counts. Resized JPEG inputs are derived at runtime with recorded dimensions and hashes.
