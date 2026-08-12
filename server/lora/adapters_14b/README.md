# 14B LoRA adapter (Qwen2.5-14B-Instruct-4bit)

Trained 2026-08-12 on the leakage-free `lora/data` split (503/55), 8 layers, batch 2, lr 1e-4,
`--mask-prompt`. Stopped at iter 150 by validation loss (the minimum): 4.045 → 0.400 (50) →
0.109 (100) → **0.077 (150)** → 0.285 (200, overfitting). `adapters.safetensors` = the iter-150
checkpoint. Peak train mem 12.6 GB. Use: `CPD_MLX_MODEL=mlx-community/Qwen2.5-14B-Instruct-4bit
CPD_MLX_ADAPTER=lora/adapters_14b CPD_FEWSHOT=0`. Full log: `../results/lora14b_train.log`.
