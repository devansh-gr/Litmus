# Technique fine-tune (LoRA) — experiment record

**Question:** can few-shot (strategy 2) or LoRA (strategy 3) push technique accuracy past
the zero-shot 61%? **Answer: not robustly — the 3B-4bit model is near its ceiling on this set.**

## Setup
- `prepare_data.py` → leakage-free split: train 267 / valid 29 from self-authored sets +
  external_**dev**; external_**holdout** (150) NEVER trained on → the honest test.
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
