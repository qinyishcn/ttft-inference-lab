# Input and artifact provenance

`artifacts.json` pins model repositories, revisions, files, byte counts and SHA-256 digests; model digests are checked against upstream Hugging Face LFS metadata. Runtime ZIPs are locally hashed and ZIP CRC-checked. Weights and binaries are excluded from this repository.

`squad-selected.json` contains seeded selected records from [SQuAD 1.1 development](https://rajpurkar.github.io/SQuAD-explorer/), created by Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev and Percy Liang. SQuAD is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Its contexts derive from Wikipedia. The included subset retains that license; it is not covered by a software license for our scripts. Cropped/repeated variants are generated at runtime and explicitly labeled; no official accuracy score is reported.

`environment.json` records the measured hardware/software without credentials, account settings, machine name, local paths or network addresses.

Visual scenes are generated entirely by `scripts/inputs.py` with a fixed seed. Pixel hashes, dimensions, sizes and frame order are recorded in each request's input metadata. No image model or externally hosted inference service is used to generate these controls.
