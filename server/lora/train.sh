#!/usr/bin/env bash
# LoRA fine-tune the 4-bit Llama-3.2-3B technique classifier on the leakage-free split.
# Trains adapters only (base frozen 4-bit); --mask-prompt = loss on the label completion
# only, matching the log-prob label-scoring inference path. Holdout is never seen here.
#
#   lora/train.sh            # full run
#   lora/train.sh 5          # smoke test (5 iters)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${CPD_MLX_MODEL:-mlx-community/Llama-3.2-3B-Instruct-4bit}"
ITERS="${1:-400}"

.venv/bin/mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data lora/data \
  --adapter-path lora/adapters \
  --fine-tune-type lora \
  --num-layers 8 \
  --batch-size 2 \
  --iters "$ITERS" \
  --learning-rate 1e-4 \
  --steps-per-report 25 \
  --steps-per-eval 50 \
  --save-every 50 \
  --val-batches 10 \
  --mask-prompt \
  --grad-checkpoint
