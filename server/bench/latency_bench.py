#!/usr/bin/env python3
"""Latency benchmark for /classify.

Measures three regimes so you can see where time goes and whether a change helped:
  cold   — first call on novel text (full forward passes, no cache)
  warm   — novel text after the model is warm (steady-state cost)
  cached — exact repeat (should be ~0ms; hits _classify_cache)

    python bench/latency_bench.py                 # live server on :8765
    python bench/latency_bench.py --n 20

Reports p50 / p90 / p95 for each regime. Run it before and after a latency change
to quantify the win.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

# Novel-ish sentences so cold/warm calls actually do work (not cache hits).
SAMPLES = [
    "The committee will reconvene after the fiscal review in early autumn.",
    "You must claim this before midnight or the reward vanishes for good.",
    "Everyone in the building has already signed up for the new plan.",
    "This invention will reshape civilization and end scarcity forever.",
    "Leading economists insist the policy is beyond any reasonable dispute.",
    "They are coming for what you built, and no one is going to stop them.",
    "Do not question the process; the answer is plainly obvious to anyone.",
    "Behold the most staggering leap our species has ever dared to make.",
    "Only the loyal understand what those outsiders are truly plotting.",
    "The shipment of ceramic tiles arrives on the third of next month.",
]


def classify(endpoint, text, timeout=60.0):
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/classify",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return time.perf_counter() - t0


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def report(name, xs):
    print(f"{name:<8} p50 {pct(xs,.50)*1000:6.0f}ms   "
          f"p90 {pct(xs,.90)*1000:6.0f}ms   "
          f"p95 {pct(xs,.95)*1000:6.0f}ms   "
          f"(n={len(xs)}, min {min(xs)*1000:.0f}ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8765")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    print(f"benchmarking {args.endpoint}/classify\n")

    # Warm the model once (first-ever call pays load, not representative).
    classify(args.endpoint, "warming up the detector before timing.")

    cold, warm, cached = [], [], []
    for i in range(args.n):
        s = SAMPLES[i % len(SAMPLES)] + f" (variant {i})"   # force novelty -> no cache
        cold.append(classify(args.endpoint, s))
    for i in range(args.n):
        s = SAMPLES[i % len(SAMPLES)] + f" (warm {i})"
        warm.append(classify(args.endpoint, s))
    fixed = "This exact sentence is classified repeatedly to measure the cache path."
    classify(args.endpoint, fixed)                          # prime the cache
    for _ in range(args.n):
        cached.append(classify(args.endpoint, fixed))

    print("=" * 60)
    report("cold", cold)
    report("warm", warm)
    report("cached", cached)
    print("=" * 60)
    print(f"median non-cached classify: {statistics.median(cold + warm)*1000:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
