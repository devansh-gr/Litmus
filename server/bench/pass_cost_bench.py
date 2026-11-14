#!/usr/bin/env python3
"""Cost of one forward pass of m new tokens over the cached 1.4k-token static prefix.
    .venv/bin/python bench/pass_cost_bench.py --model mlx-community/Qwen2.5-14B-Instruct-4bit"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import server as s
from bench.latency_bench import SAMPLES

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
ap.add_argument("--reps", type=int, default=9)
a = ap.parse_args()
model, tok = load(a.model)
P = [tok.encode(s._chat_prompt(tok, t, s.SYSTEM, s.TECHNIQUE_FEWSHOT)) for t in SAMPLES]
L = min(next(i for i, (x, y) in enumerate(zip(P[0], p)) if x != y) for p in P[1:])
cache = make_prompt_cache(model); mx.eval(model(mx.array([P[0][:L]]), cache=cache))
res = {"model": a.model, "context": L, "ms": {}}
for m in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
    x = mx.array([P[1][L:L + 1] * m])
    ts = []
    for _ in range(a.reps + 1):
        t0 = time.perf_counter(); mx.eval(model(x, cache=cache)); ts.append(time.perf_counter() - t0)
        trim_prompt_cache(cache, m)
    res["ms"][m] = 1000 * statistics.median(ts[1:])
print(json.dumps(res))
