#!/usr/bin/env python3
"""Benchmark the detector against a labeled eval set.

Runs every example in tests/data/eval_set.jsonl through the live /classify endpoint
and reports overall accuracy, per-vector precision/recall, the worst confusions, and
confidence calibration (mean confidence on hits vs misses).

    python tests/run_eval.py                      # hit the live server on :8765
    python tests/run_eval.py --endpoint http://127.0.0.1:8765
    python tests/run_eval.py --json out.json      # also dump machine-readable results

This is the detector's honest report card: what it gets right, what it confuses, and
whether its confidence is trustworthy. Keep the eval set diverse and hard.
"""
import argparse
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "eval_set.jsonl"


def classify(endpoint: str, text: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/classify",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8765")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--json", default=None, help="dump results as JSON to this path")
    args = ap.parse_args()

    rows = load(Path(args.data))
    if not rows:
        print("no eval rows found", file=sys.stderr)
        return 2

    results = []
    per_label = defaultdict(lambda: {"support": 0, "correct": 0})
    predicted_count = Counter()
    confusions = Counter()
    conf_hit, conf_miss = [], []
    latencies = []

    print(f"running {len(rows)} examples against {args.endpoint} ...\n")
    for row in rows:
        text, gold = row["text"], row["label"]
        t0 = time.perf_counter()
        try:
            resp = classify(args.endpoint, text)
        except Exception as e:
            print(f"  ERROR on {text[:40]!r}: {e}", file=sys.stderr)
            return 1
        dt = time.perf_counter() - t0
        latencies.append(dt)

        pred = resp.get("vector", "?")
        conf = resp.get("confidence", 0)
        ok = pred == gold
        per_label[gold]["support"] += 1
        per_label[gold]["correct"] += int(ok)
        predicted_count[pred] += 1
        if not ok:
            confusions[(gold, pred)] += 1
        (conf_hit if ok else conf_miss).append(conf)
        results.append({"text": text, "gold": gold, "pred": pred, "confidence": conf, "ok": ok})

    total = len(results)
    correct = sum(r["ok"] for r in results)
    acc = correct / total

    # Per-label precision/recall.
    print("=" * 68)
    print(f"OVERALL ACCURACY: {correct}/{total} = {acc:.1%}")
    print("=" * 68)
    print(f"\n{'vector':<30} {'recall':>8} {'prec':>8} {'support':>8}")
    print("-" * 58)
    for label in sorted(per_label):
        support = per_label[label]["support"]
        rec = per_label[label]["correct"] / support if support else 0.0
        tp = per_label[label]["correct"]
        prec = tp / predicted_count[label] if predicted_count[label] else 0.0
        print(f"{label:<30} {rec:>7.0%} {prec:>8.0%} {support:>8}")

    if confusions:
        print("\nworst confusions (gold -> predicted):")
        for (gold, pred), n in confusions.most_common(8):
            print(f"  {gold:<28} -> {pred:<24} x{n}")

    mh = sum(conf_hit) / len(conf_hit) if conf_hit else 0
    mm = sum(conf_miss) / len(conf_miss) if conf_miss else 0
    print(f"\nconfidence calibration:")
    print(f"  mean confidence when CORRECT : {mh:.0f}%")
    print(f"  mean confidence when WRONG   : {mm:.0f}%")
    print(f"  (healthy detector: correct >> wrong)")

    lat = sorted(latencies)
    p50 = lat[len(lat) // 2]
    p95 = lat[int(len(lat) * 0.95)]
    print(f"\nlatency: p50 {p50*1000:.0f}ms  p95 {p95*1000:.0f}ms  "
          f"(cached repeats are near-instant)")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "accuracy": acc, "total": total, "correct": correct,
            "results": results,
        }, indent=2))
        print(f"\nwrote {args.json}")

    return 0 if acc >= 0.70 else 1


if __name__ == "__main__":
    sys.exit(main())
