#!/usr/bin/env python3
"""Show the detector's confusion pairs on the manipulation eval set.

A focused view for tuning the technique labels: which gold vector gets mislabeled as
what, and the exact texts. Complements run_eval.py (which gives the aggregate numbers).

    python tests/confusions.py                 # live server on :8765
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def classify(endpoint, text, timeout=60):
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/classify",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8765")
    ap.add_argument("--data", default=str(HERE / "data" / "eval_set.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    pairs = Counter()
    misses = []
    for row in rows:
        pred = classify(args.endpoint, row["text"])["vector"]
        if pred != row["label"]:
            pairs[(row["label"], pred)] += 1
            misses.append((row["label"], pred, row["text"]))

    correct = len(rows) - len(misses)
    print(f"accuracy: {correct}/{len(rows)} = {correct/len(rows):.1%}\n")
    print("confusion pairs (gold -> predicted):")
    for (g, p), n in pairs.most_common():
        print(f"  {g:>28} -> {p:<24} x{n}")
    print("\nmisclassified texts:")
    for g, p, t in misses:
        print(f"  [{g} -> {p}] {t[:56]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
