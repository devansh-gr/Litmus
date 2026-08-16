# Technique fine-tune (LoRA) — experiment record

**Question:** can few-shot (strategy 2) or LoRA (strategy 3) push technique accuracy past
the zero-shot 61%? **Answer (3B): not robustly — near its ceiling on this set. (14B): the LoRA
tied zero-shot / lost to few-shot too, because the training data was starved on most vectors.**

> **2026-08-12 update — data expansion.** The split was rebalanced to fix that starvation: added
> `external_propaganda_train` (592 ex of fear/outrage/tribal/authority/crit-think/hype from the
> disjoint `anismahmahi/10_techniques_train` split — leakage-free vs the propaganda bench) and a
> `PER_CLASS_CAP=90` in `prepare_data.py`. The split is now **train 884 / valid 98** (was 503/55),
> with the six previously-starved vectors at 90 each. Retrain of the 14B on this pending a charge.
> (Counts below are from the OLD 267/29 run; the accuracy table is the OLD result.)

## The 14B round (2026-08-12 → 15) — fine-tune still loses to prompting

The "bigger model" the 3B round pointed at got run. On `external_test` (N=300):

| config | accuracy |
|---|---|
| Qwen2.5-14B zero-shot | 79.3% |
| **Qwen2.5-14B + few-shot** | **82.3%** (best) |
| Qwen2.5-14B LoRA (iter-150, old 503 data) | 78.9% (partial) — ties zero-shot, below few-shot |
| DeepSeek-R1-Distill-14B (reasoning path) | 76.2% on a balanced-42 subset, ~35s/scan → disqualified |

Same story as the 3B: **a LoRA fine-tune does not beat in-context few-shot** at this data scale, so
few-shot stays the shipped high-accuracy config. The one thing not yet tried: retrain on the rebalanced
**884/98** split (above) and bench on `external_propaganda` — forecast a wash on `external_test` (only 6
of 14 classes present) but a real win on the starved vectors the new data feeds. Raw:
`results/bench_14b*.txt`, `results/pred_lora14b_partial.jsonl`, `results/pred_reason_r1_14b.jsonl`.

## Setup
- `prepare_data.py` → leakage-free split: (now) train 884 / valid 98 from self-authored sets +
  external_**dev** + Mathur-train + propaganda-train; external_**holdout** (150) NEVER trained on → the honest test.
- `train.sh` → `mlx_lm.lora`, adapters only (~3.5M params, 8 layers), `--mask-prompt`, peak 5.2GB.
- Overfitting watch: val loss 6.46 → **0.44 (iter 50/100)** → 0.51 (150) → 2.54 (200, overfit).
  Shipped the **iter-100** checkpoint.

## Three-way comparison — all on the untouched holdout (150)
| config | accuracy | macro-F1 | notes |
|---|---|---|---|
| zero-shot (definitions only) | 62.0% [54–69] | 0.41 | baseline |
| **few-shot (shipped default)** | **62.7%** [54.7–70] | 0.34 | best accuracy; fomo/dopamine dip |
| LoRA (opt-in) | 60.7% [52.7–68] | **0.49** | best balance; social-proof collapsed to 0.11 recall |

All three are **statistically tied on accuracy** (CIs overlap heavily). Few-shot is the nominal
accuracy winner (the user's priority) but the edge is ~1 example — noise — and it costs balance.
LoRA is the most balanced but not more accurate: 267 imbalanced examples over-fit some classes
and starved social-proof. **The real accuracy lever is a bigger model (parked, strategy 1).**

## How to run the opt-in LoRA
    python lora/prepare_data.py && lora/train.sh
    CPD_MLX_ADAPTER=lora/adapters CPD_FEWSHOT=0 <start server>
