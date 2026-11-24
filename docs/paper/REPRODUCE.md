# Reproducing the paper

Every number in the paper comes from a script in `server/bench/` and a raw result file in
`server/bench/results/`. `server/bench/check_claims.py` recomputes each stated number from those
files and prints PASS or FAIL next to it (see [Checking the numbers](#checking-the-numbers)). The
figure script `make_figs.py` holds medians and percentiles copied from the same files as constants,
and `check_claims.py` covers those values too.

The fixed version for the paper is git tag `paper-v23`. The data split is `server/tests/data/external_test.jsonl`
(300) = `external_dev.jsonl` (150, used to tune prompts and thresholds) + `external_holdout.jsonl`
(150, never used for tuning, and all accuracy in the paper is holdout-only). The classes are 5 of
the 14 techniques plus `none`.

## Environment
- Apple M4 Pro, 24 GB unified memory, macOS 26. For timing runs the machine should otherwise be idle.
- Python 3.11.15 and the exact versions in `server/bench/requirements.txt`.
  ```sh
  cd server
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -r bench/requirements.txt
  ```
  The benchmarks import `server.py` for its prompts and labels, so they need `torch`, `fastapi` and
  `pandas` even though they never use them. TRIBE v2 and the macOS app are not needed.
- Checkpoints (4-bit, group size 64, as published on Hugging Face), downloaded on first use, are
  `mlx-community/Llama-3.2-3B-Instruct-4bit`, `mlx-community/Qwen2.5-14B-Instruct-4bit`.
- Both tokenizers give the same label lengths. Technique labels are 1 to 7 tokens, 62 in total,
  47 forwarded (the last token of each label is never an input). Gate labels are 1 token each.
- All timings are warm. The model is loaded and the static prefix cached before timing starts. The
  cache build is reported separately and excluded.

## Method code
- `server/bench/packed.py` is the tree-masked label pass.
  - Tokens are ordered by depth, every label's token 0 first, then token 1, and so on.
  - Mask row q allows the whole cached prompt, plus packed token k only when k belongs to the
    same label and is at a depth no greater than q. Sibling labels never see each other.
  - For RoPE, every token at depth j gets position `offset + j`. Each depth group is folded into the
    heads axis so the stock RoPE op applies one offset per group (`_GroupRoPE`).
  - The cache is trimmed back by the packed length afterwards.
  - `control="causal"` and `control="rope"` deliberately break the mask or the positions.
- `server/bench/prefix_cache_bench.py` holds the naive (no reuse), deployed (label reuse) and static
  (cross-request prefix cache) scorers that `full_bench.py` times.
- `server/server.py` holds the prompts (`SYSTEM`, `GATE_SYSTEM`, `TECHNIQUE_FEWSHOT`), labels (`VECTORS`,
  `GATE_LABELS`), the gate threshold (0.65) and temperatures. The benchmarks read them from there.

## Experiments
Run from `server/`. Results are appended to `--out`, so use a new file name per run.

| Paper item | Command | Raw output |
|---|---|---|
| Table I, decision changes, accuracy, McNemar | `.venv/bin/python bench/full_bench.py --model <ckpt> --out <file>` (for the 14B, the deployed run is in two parts, `--deployed-limit 30`, then `--deployed-start 30 --strategies deployed`, and `--naive 0`) | `full_3b_2026-11-14.jsonl`, `full_14b_2026-11-14.jsonl` |
| Summary of the above (percentiles, flips, Wilson, McNemar) | `.venv/bin/python bench/analyze_full.py <3b file> <14b file>` | stdout |
| Eq. (2), 14B no-reuse baseline | `prefix_cache_bench.py` at commit `70595f6` (see [Provenance](#provenance-notes)) | `prefix_cache_2026-11-14.jsonl` |
| Eq. (2) check, all strategies on 10 sample inputs | `.venv/bin/python bench/prefix_cache_bench.py --model <ckpt> [--stage gate]` | `prefix_cache_blocked_2026-11-14.jsonl` |
| Fig. 1(b) pass cost | `.venv/bin/python bench/pass_cost_bench.py --model <ckpt>` (medians of 9), and for the spread `--reps 30` | `pass_cost_2026-11-14.jsonl`, `pass_cost_2026-11-16.jsonl` (raw reps) |
| Fig. 2 kernel, independent and dependent chain | `.venv/bin/python bench/qmm_bench.py` | `qmm_2026-11-15.json` |
| Packed-token sweep and broken controls | `.venv/bin/python bench/packed_sweep.py --model <ckpt> --inputs 50 --data tests/data/external_test.jsonl` | `packed_sweep_50_2026-11-15.jsonl` (an earlier 5-input run is `packed_sweep_2026-11-15.jsonl`) |
| Figures | `cd ../docs/paper && ../../server/.venv/bin/python make_figs.py` and `make_sweep_fig.py` | `fig_hero.*`, `fig_qmm.*`, `fig_sweep.png` |

The timed work recorded in the two `full_*` files sums to 26 min (3B) and 87 min (14B),
excluding model loads and warm-up. The deployed path dominates. 14B no-reuse took about 2 min per
input in the earlier harness (122.8 s median).

### Output formats
- `full_*.jsonl` has one `{"meta": true, ...}` line per run (model, load time, static prefix length and
  build time per stage, prompt lengths), then one line per (strategy, input) with `strategy`, `i`
  (row in `external_test.jsonl`), `label`, `holdout`, `gate_s`, `tech_s` (seconds) and the raw
  `gate` and `tech` score vectors (length-normalized log-likelihood, nats, in `GATE_LABELS` and
  `VECTORS` order). `naive` lines carry `tech_s` and `tech` only.
- `pass_cost_*.jsonl` has one line per model, and `ms[m]` is the median latency of one forward pass of m
  new tokens over the cached technique prefix, `raw_ms[m]` the individual repetitions (11-16 only).
  The m new tokens are copies of one token, so every m sees the same context.
- `qmm_*.json` gives microseconds per call, `q4_us` and `fp16_us` (independent calls, 20 per sync),
  `q4_chain_us` and `fp16_chain_us` (dependent up and down projections, per matmul).
- `packed_sweep_*.jsonl` has one line per (model, packed tokens) with per-input `seq_s` and `packed_s`,
  the largest score difference to sequential scoring (`max_diff`) and for the two broken controls.
- `prefix_cache_blocked_*.jsonl` has per-input times for each strategy on the 10 inputs of
  `bench/latency_bench.py` `SAMPLES`, plus prompt and static-prefix token counts.

### Randomness
Model scoring is deterministic given the inputs. `packed_sweep.py` draws its synthetic labels with
`random.Random(0)`. `qmm_bench.py` uses unseeded random weights and activations. Timing does not
depend on their values, but the exact tensors are not reproducible.

## Checking the numbers
```sh
cd server && .venv/bin/python bench/check_claims.py
```
It runs in seconds and needs no model (the label-length check reads the cached 3B tokenizer and is
skipped if that is unavailable offline). At `paper-v23` all 71 checks pass. Tags before `paper-v23` had 3
failures, overstatements in two captions that `paper-v23` corrected.
- Fig. 1(b) caption. In the 30-run repeat, the 5th to 95th percentiles lie within 6.5% of the
  median (3B, m = 1 and 2), and one 30-run median (14B, m = 48) differs from the 9-run median by
  5.1%. The caption said 5% and now says 7%.
- Fig. 2 caption. The dependent chain agrees with independent calls within 3.5% from M = 12 up,
  but differs by up to 11.6% at M = 2 to 10 (-11.6% at M = 6). The caption said 7% and now gives
  both ranges.

## Provenance notes
- **14B no-reuse baseline (Fig. 1(a) hollow square, Eq. (2) 15.8x).** It does not come from
  `full_bench.py`. It was measured on 2026-11-13 by an earlier `prefix_cache_bench.py` (commit
  `70595f6`) on the first 3 inputs of `bench/latency_bench.py` `SAMPLES` (hand-written sentences, not
  `external_test.jsonl`), with the no-reuse, deployed and static paths interleaved per input. The
  file keeps only the medians and one same-input pair (`naive_vs_deployed_same_input`, 122.76 s vs
  7.76 s). The per-input no-reuse times were not kept, so n = 3 follows from the script default and
  is not recorded in the file. 15.8x is that single pair. The ratio of the medians is 15.15x.
  The same harness gave a 3B no-reuse median of 43.7 s where `full_bench.py` measured 27.0 s over
  10 benchmark inputs, so the two no-reuse points in Fig. 1(a) come from different harnesses and
  are not directly comparable. To remeasure the 14B in the paper's harness, run
  `.venv/bin/python bench/full_bench.py --model mlx-community/Qwen2.5-14B-Instruct-4bit --strategies "" --naive 10 --out bench/results/naive_14b_<date>.jsonl`
- **14B static prefix build time.** `full_14b_2026-11-14.jsonl` holds three runs (three meta
  lines) with build times of 12.03, 13.75 and 11.54 s. The paper quotes 13.8 s.
- **Static fraction of the prompt (98.7% technique, 98.5% gate).** This is computed on the first
  `SAMPLES` prompt (`prefix_cache_blocked_2026-11-14.jsonl`, 1446 of 1465 and 1290 of 1309 tokens).
  Over the 300 benchmark inputs the median is 98.9% and 98.8%.
- **Predicted stage latencies (239 and 878 ms sequential, 0.116 and 0.442 s packed).** They sum
  measured pass costs from `pass_cost_2026-11-14.jsonl`, the 19-token suffix pass then one pass of
  l - 1 tokens per label of length l > 1 (sequential) or one 48-token pass (packed). m = 5 and
  m = 19 were not measured. t(5) is the mean of t(4) and t(6), and t(19) the mean of t(16) and
  t(24), which lie on the same plateau.
- **Figure whiskers.** In `make_figs.py`, the deployed-row 5th and 95th percentiles in `LAD_P` are
  order statistics (numpy `method="higher"`, 3B 1.746 to 3.05 s, 14B 7.893 to 9.874 s), while the
  other rows match linear interpolation. With linear interpolation the deployed 95th percentiles
  are 2.99 s and 9.83 s. The comment above `LAD` that says "deployed: 30" for the 14B predates the
  completed 300-input deployed run.

## Sweep result (`fig_sweep.png`)
Speedup of packed over sequential scoring per input, median (5th to 95th percentile), 50 inputs.

| Packed tokens | 3B | 14B |
|---|---|---|
| 8 | 1.06 (1.04 to 1.10) | 1.00 (0.97 to 1.03) |
| 16 | 1.43 (1.37 to 1.47) | 1.44 (1.39 to 1.45) |
| 32 | 2.80 (2.64 to 2.97) | 2.82 (2.68 to 3.03) |
| 48 | 2.40 (2.26 to 2.55) | 2.27 (2.05 to 2.36) |
| 64 | 3.06 (2.90 to 3.19) | 2.98 (2.79 to 3.12) |
| 96 | 3.46 (3.26 to 3.52) | 2.81 (2.46 to 3.07) |

The max packed vs sequential score difference was 0.027 nats (3B) and 0.067 (14B). Broken controls gave 0.64 to
12.9 nats. Top label agreed on 597 of 600 comparisons. The 3 disagreements are synthetic random-token
labels, where near-ties are common. On the real label set, packing changed 0 of 300 decisions.

## Decision changes, by cause (300 inputs)
- Deployed vs cached prefix, 3 of 300 on the 3B (inputs 70 and 147, whose gate p(no) moved between
  0.6478 and 0.6514 across the 0.65 threshold, and input 229, an exact tie between `false-urgency`
  and `fomo`). 0 of 300 on the 14B. The largest score difference to the deployed path was 3B 0.031 nats,
  14B 0.17 nats (cached) and 0.11 nats (packed).
- Cached vs packed, 0 of 300 on both models.
- Median paired speedup of packed over deployed, full path, was 3B 20.4x (5th to 95th percentile 14.7
  to 34.7), 14B 22.0x (15.4 to 26.0).

## Limits
One machine and one runtime (MLX). The deployed baselines cover all 300 inputs on both models.
The no-reuse baselines cover 10 inputs (3B) and 3 inputs (14B, from an earlier harness, see above).
Accuracy covers 5 of the 14 techniques. The kernel attribution rests on op-level timing and is
consistent with the 32-row tile of MLX's quantized matmul. No kernel-dispatch trace was taken.
