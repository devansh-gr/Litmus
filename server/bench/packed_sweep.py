#!/usr/bin/env python3
"""Packed vs sequential label scoring as the number of packed label tokens grows, with two broken
controls (causal mask across labels, consecutive RoPE positions) to show what a masking or position
bug looks like next to ordinary fp16 noise. Synthetic label sets: K labels of 5 tokens (4 packed each).
    .venv/bin/python bench/packed_sweep.py --model mlx-community/Llama-3.2-3B-Instruct-4bit"""
import argparse, json, random, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import server as s
from bench.latency_bench import SAMPLES
from bench.prefix_cache_bench import score_from
from bench.packed import install, packed_scores

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--inputs", type=int, default=5)
a = ap.parse_args()
model, tok = load(a.model); install(model)
enc = lambda t, sp=True: tok.encode(t, add_special_tokens=sp)
prompts = [enc(s._chat_prompt(tok, t, s.SYSTEM, s.TECHNIQUE_FEWSHOT)) for t in SAMPLES[:a.inputs]]
static_len = min(next(i for i, (x, y) in enumerate(zip(prompts[0], p)) if x != y) for p in prompts[1:])
cache = make_prompt_cache(model); mx.eval(model(mx.array([prompts[0][:static_len]]), cache=cache))
vocab = sorted({t for v in s.VECTORS for t in enc(v, False)} | {t for p in prompts for t in p[static_len:]})
rng = random.Random(0)

def run(fn, p):
    suf = p[static_len:]; last = model(mx.array([suf]), cache=cache)[0][-1]; mx.eval(last)
    t0 = time.perf_counter(); r = fn(last); mx.eval(r); dt = time.perf_counter() - t0
    trim_prompt_cache(cache, len(suf)); return r, dt

for T in (8, 16, 32, 48, 64, 96):
    labels = [[rng.choice(vocab) for _ in range(5)] for _ in range(T // 4)]
    row = {"model": a.model, "packed_tokens": T, "labels": len(labels), "seq_s": [], "packed_s": [],
           "max_diff": 0.0, "causal_diff": 0.0, "rope_diff": 0.0, "argmax_agree": 0}
    run(lambda l: packed_scores(model, cache, l, labels), prompts[0])           # warm-up
    for p in prompts:
        sq, ts = run(lambda l: score_from(model, cache, l, labels), p)
        pk, tp = run(lambda l: packed_scores(model, cache, l, labels), p)
        ca, _ = run(lambda l: packed_scores(model, cache, l, labels, "causal"), p)
        ro, _ = run(lambda l: packed_scores(model, cache, l, labels, "rope"), p)
        row["seq_s"].append(ts); row["packed_s"].append(tp)
        d = lambda x: mx.abs(sq - x).max().item()
        row["max_diff"] = max(row["max_diff"], d(pk)); row["causal_diff"] = max(row["causal_diff"], d(ca))
        row["rope_diff"] = max(row["rope_diff"], d(ro)); row["argmax_agree"] += int(mx.argmax(sq).item() == mx.argmax(pk).item())
    row["seq_med"], row["packed_med"] = statistics.median(row["seq_s"]), statistics.median(row["packed_s"])
    print(json.dumps(row), flush=True)
