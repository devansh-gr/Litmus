# Litmus

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24557055.svg)](https://doi.org/10.5281/zenodo.24557055)

Code and data for the paper **"Cross-Request Prefix Caching and the Cost of Short Forward Passes in
On-Device LLM Classification"** (Devansh Gaur, University of Waterloo).

- Paper (PDF on Zenodo): https://zenodo.org/records/24557055
- DOI: [10.5281/zenodo.24557055](https://doi.org/10.5281/zenodo.24557055)

Paper sources, figures and the reproduction guide are in [`docs/paper/`](docs/paper/). The version
used for the paper is tag `paper-v23`.

The repository also contains Litmus, the macOS application the classifier was built for (see
[The application](#the-application)).

## Research question

A local LLM used as a closed-set classifier scores each candidate label by its length-normalized
log-likelihood as a continuation of the prompt. In this workload the prompt is about 1.4k tokens and
about 99% of it is identical across requests, and each request scores 2 gate labels and 15 technique
labels of 1 to 7 tokens. The paper asks where latency goes once repeated prefix work is removed, and
how much of the rest can be recovered on Apple silicon.

## Findings

Apple M4 Pro (24 GB), MLX 0.32.0, 4-bit checkpoints, 300 benchmark inputs per model, warm caches.
Full path = gate stage plus technique stage on every input.

| | Llama-3.2-3B | Qwen2.5-14B |
|---|---|---|
| Base, prompt prefilled once per request and shared by all labels (median) | 3.53 s | 15.21 s |
| + static prefix cached across requests | 0.306 s | 1.140 s |
| + all technique labels in one tree-masked pass | 0.170 s | 0.683 s |
| Median paired speedup, base to final | 20.4x | 22.0x |
| Technique stage, cached then packed (median) | 0.260 s to 0.127 s | 0.966 s to 0.512 s |
| Decisions changed, base to final (of 300) | 3 (2 at the gate threshold, 1 exact tie) | 0 |

- **The baseline already reuses the prompt across labels.** The speedups above do not include the
  much larger gain over scoring each label from scratch. That no-reuse baseline was measured on only
  10 inputs (3B, 27.0 s technique stage) and 3 inputs (14B, 122.8 s, from an earlier harness on
  hand-written inputs, see [provenance](docs/paper/REPRODUCE.md#provenance-notes)).
- **After caching, the remaining cost is a series of short passes** (one 19-token suffix pass and
  14 label passes of 1 to 6 tokens). Their cost rises in steps with the number of new tokens rather
  than following the weight-read bound. An isolated 13824 x 5120 4-bit matmul steps at M = 33 and
  M = 65, consistent with the 32-row tile of MLX's quantized matmul kernel. No kernel-dispatch trace
  was taken.
- **Packing the 47 forwarded label tokens into one pass** under a tree-shaped mask (each token sees
  the prompt and earlier tokens of its own label, tokens at the same depth share a RoPE position)
  roughly halves the cached technique stage.
- Accuracy is not the contribution. On the 150 holdout inputs the pipeline reached 86.0% top-1 with
  the 14B and 66.0% with the 3B (exact McNemar p < 1e-6), on 5 of the 14 technique classes plus `none`.

## Established techniques and what is specific here

The paper cites these established techniques. Label scoring by log-likelihood, sharing a prompt's KV cache
across continuations, cross-request prefix caching (Prompt Cache, SGLang, PagedAttention), tree
attention masks for scoring many continuations at once (SpecInfer), and the small-M cost of MLX 4-bit
matmul (ml-explore/mlx issue #4265). What this repository adds is the measurement, namely the per-pass
cost curve after caching on one Apple silicon machine, its kernel-level explanation, an MLX
implementation of the packed label pass with depth-grouped RoPE (`server/bench/packed.py`), and
end-to-end latency and decision agreement on 300 inputs for two model sizes.

## Reproducing

```sh
cd server
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r bench/requirements.txt
.venv/bin/python bench/check_claims.py      # recompute every number in the paper from bench/results/
```

`check_claims.py` needs no model and runs in seconds. It prints PASS or FAIL for each number as
printed in the paper. At `paper-v23` all 71 checks pass. Tags before `paper-v23` had 3
failures, small overstatements in two figure captions that were since corrected (see
[REPRODUCE.md](docs/paper/REPRODUCE.md#checking-the-numbers)).

Rerunning the measurements needs an Apple silicon Mac with at least 24 GB of memory for the 14B.
Commands, run times, output formats and the origin of each result file are in
[`docs/paper/REPRODUCE.md`](docs/paper/REPRODUCE.md). Figures are regenerated with
`docs/paper/make_figs.py`.

## Limitations

- One machine (Apple M4 Pro) and one runtime (MLX). The tile explanation is specific to MLX's
  quantized kernels and was inferred from op-level timing.
- No-reuse baselines cover 10 inputs (3B) and 3 inputs (14B). The 14B point comes from a different
  harness and input set than the 3B point.
- The benchmark covers 5 of the 14 technique classes. Labels come from three public corpora mapped
  onto this taxonomy, and the mapping is approximate (see `server/tests/data/README.md`).
- Caching and packing reorder half-precision sums, so scores are not bit-exact. The largest score
  difference between the base and final paths was 0.031 nats (3B) and 0.11 nats (14B).

## Repository layout

```
docs/paper/            paper source (make_docx.py), figures (make_figs.py), REPRODUCE.md
server/bench/          benchmark scripts, packed.py (tree-masked pass), check_claims.py
server/bench/results/  raw benchmark output used by the paper
server/server.py       prompts, labels and thresholds (imported by the benchmarks), and the app's server
server/tests/data/     benchmark inputs: external_test.jsonl = external_dev.jsonl + external_holdout.jsonl
server/tests/          application test suite and earlier accuracy benchmarks
server/experiments/    earlier experiments behind the application's design
Sources/, scripts/     macOS application
```

## The application

Litmus is a macOS menu-bar tool. Select text, press Cmd-B, and a local model names the persuasion
technique with a confidence. An optional view renders a predicted cortical response with Meta's
TRIBE v2. Everything runs locally. The paper's benchmarks do not need the application or TRIBE v2.

Accuracy figures from application development (different prompt versions, configurations and
harness) are in [`docs/RETROSPECTIVE.md`](docs/RETROSPECTIVE.md) and
[`server/tests/RESULTS.md`](server/tests/RESULTS.md). They are not the paper's numbers.

Setup needs macOS 14+ on Apple silicon with Xcode, Homebrew and `uv`.
```sh
cd server
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e vendor/tribev2 fastapi uvicorn nilearn scipy mlx mlx-lm scikit-learn
.venv/bin/python apply_patches.py        # Apple-silicon and offline-TTS patches to TRIBE v2
.venv/bin/hf auth login                  # TRIBE v2 loads gated meta-llama weights, accept the license first
.venv/bin/python build_baseline.py       # neutral baseline for the cortical view (one-time)
.venv/bin/python server.py               # listens on 127.0.0.1:8765
cd .. && brew install xcodegen && ./scripts/build.sh && open build/Debug/CorticalPersuasionDecoder.app
```
`server/vendor/tribev2` is not tracked and must be cloned from Meta's TRIBE v2 repository. The
launchd agent in `scripts/launchd/` contains absolute paths for the author's machine and must be
edited before use. Architecture and configuration are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Third-party components

This repository does not redistribute model weights. Each component keeps its own license.
- Models are Llama 3.2 (Llama 3.2 Community License) and Qwen2.5-14B-Instruct, used as the 4-bit
  conversions published by `mlx-community` on Hugging Face.
- The runtime is MLX and mlx-lm (ml-explore).
- TRIBE v2 (Meta), used only by the application, is licensed CC BY-NC 4.0.
- Benchmark text in `server/tests/data/external_*.jsonl` is drawn from the e-commerce dark-pattern
  dataset of Yada et al. (2022), the LOGIC logical-fallacy dataset of Jin et al. (2022) and the
  clickbait dataset of Chakraborty et al. (2016), relabeled onto this taxonomy by
  `server/tests/build_external_set.py`. The original licenses of those datasets apply.

## Citation

```bibtex
@misc{gaur2026prefix,
  author    = {Gaur, Devansh},
  title     = {Cross-Request Prefix Caching and the Cost of Short Forward Passes in On-Device LLM Classification},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24557055},
  url       = {https://doi.org/10.5281/zenodo.24557055}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).
