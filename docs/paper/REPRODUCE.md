# Reproducing the paper

Every number in the paper comes from a script in `server/bench/` and a raw result file in
`server/bench/results/`. Nothing below is hand-entered.

Fixed version for the paper: git tag `paper-v15`. Data split: `server/tests/data/external_test.jsonl`
(300) = `external_dev.jsonl` (150, used to tune prompts and thresholds) + `external_holdout.jsonl`
(150, never used for tuning; all accuracy in the paper is holdout-only). Classes: 5 of the 14
techniques plus `none`.

## Environment
- Apple M4 Pro, 24 GB unified memory, macOS 26; machine otherwise idle.
- Python 3.11.15, `pip install mlx==0.32.0 mlx-lm==0.31.3` (plus the server's FastAPI deps).
- Checkpoints (4-bit, group size 64, as published):
  `mlx-community/Llama-3.2-3B-Instruct-4bit`, `mlx-community/Qwen2.5-14B-Instruct-4bit`.
- Both tokenizers give the same label lengths: technique labels 1 to 7 tokens, 62 in total,
  47 forwarded (the last token of each label is never an input); gate labels 1 token each.
- All timings are warm: model loaded and static prefix cached before timing starts. The cache
  build (2.7 s on the 3B, 13.8 s on the 14B) is reported separately and excluded.

## Method code
- `server/bench/packed.py`: the tree-masked label pass.
  - Tokens are ordered by depth: every label's token 0, then token 1, and so on.
  - Mask row q allows the whole cached prompt, plus packed token k only when k belongs to the
    same label and is at a depth no greater than q. Sibling labels never see each other.
  - RoPE: every token at depth j gets position `offset + j`. Each depth group is folded into the
    heads axis so the stock RoPE op applies one offset per group (`_GroupRoPE`).
  - The cache is trimmed back by the packed length afterwards.
  - `control="causal"` and `control="rope"` deliberately break the mask or the positions.
- `server/bench/prefix_cache_bench.py`: naive, deployed (label reuse), static (cross-request
  prefix cache) scorers.

## Experiments
Run from `server/`:

| Paper item | Command | Raw output |
|---|---|---|
| Table I, agreement, accuracy | `.venv/bin/python bench/full_bench.py --model <ckpt> --out <file>` (14B: deployed run in two parts, `--deployed-limit 30`, then `--deployed-start 30 --strategies deployed`; `--naive 3`) | `full_3b_2026-11-14.jsonl`, `full_14b_2026-11-14.jsonl` |
| Analysis (percentiles, flips, Wilson, McNemar) | `.venv/bin/python bench/analyze_full.py <3b file> <14b file>` | stdout |
| Eq. (2) check | `.venv/bin/python bench/prefix_cache_bench.py --model <ckpt>` | `prefix_cache_blocked_2026-11-14.jsonl` |
| Fig. 1(b) pass cost | `.venv/bin/python bench/pass_cost_bench.py --model <ckpt>` | `pass_cost_2026-11-14.jsonl` (medians); spread: `--reps 30` to `pass_cost_2026-11-16.jsonl` (raw reps) |
| Fig. 2 kernel, independent and dependent chain | `.venv/bin/python bench/qmm_bench.py` | `qmm_2026-11-15.json` |
| Packed-token sweep and broken controls | `.venv/bin/python bench/packed_sweep.py --model <ckpt>` | `packed_sweep_2026-11-15.jsonl` |

Benchmark inputs: `server/tests/data/external_test.jsonl` (300 public examples); the 150 that
also appear in `external_holdout.jsonl` played no part in tuning. The sweep uses the first 50 inputs of `external_test.jsonl` (`--inputs 50 --data
tests/data/external_test.jsonl`) and synthetic label sets of K labels x 5 tokens drawn with seed 0
from the label and input vocabulary, giving 8 to 96 packed tokens. Output:
`packed_sweep_50_2026-11-15.jsonl` (an earlier 5-input run is `packed_sweep_2026-11-15.jsonl`).

## Sweep result (`fig_sweep.png`)
Speedup of packed over sequential scoring per input, median (5th to 95th percentile), 50 inputs:

| Packed tokens | 3B | 14B |
|---|---|---|
| 8 | 1.06 (1.04 to 1.10) | 1.00 (0.97 to 1.03) |
| 16 | 1.43 (1.37 to 1.47) | 1.44 (1.39 to 1.45) |
| 32 | 2.80 (2.64 to 2.97) | 2.82 (2.68 to 3.03) |
| 48 | 2.40 (2.26 to 2.55) | 2.27 (2.05 to 2.36) |
| 64 | 3.06 (2.90 to 3.19) | 2.98 (2.79 to 3.12) |
| 96 | 3.46 (3.26 to 3.52) | 2.81 (2.46 to 3.07) |

Max packed vs sequential score difference: 0.027 nats (3B), 0.067 (14B). Broken controls: 0.64 to
12.9 nats. Top label agreed on 597 of 600 comparisons; the 3 disagreements are synthetic random-token
labels, where near-ties are common. On the real label set, packing changed 0 of 300 decisions.

## Decision changes, by cause (300 inputs)
- Deployed vs cached prefix: 3 of 300 on the 3B (two gate probabilities within 0.003 of the
  threshold, one exact technique tie); 0 of 300 on the 14B (largest technique-score difference 0.17 nats). Median paired speedup of packed over deployed on the 14B: 22.0x (5th to 95th percentile 15.4 to 26.0).
- Cached vs packed: 0 of 300 on both models.

## Limits
One machine and one runtime (MLX). The deployed baselines cover all 300 inputs on both models; no-reuse baselines
10 (3B) and 3 (14B). The kernel attribution rests on the op-level timing and is consistent
with the 32-row tile of MLX's quantized matmul; no kernel-dispatch trace was taken.
