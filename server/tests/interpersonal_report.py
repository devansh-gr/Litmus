"""Interpersonal-manipulation family report.

Scores the curated interpersonal set (guilt-tripping / love-bombing / blame-shifting /
gaslighting→critical-thinking-suppression + benign controls) against the live server.

NOTE ON HONESTY: these examples are self-authored, so this is a BEHAVIORAL REGRESSION
set for the interpersonal cluster — NOT an accuracy benchmark. The honest accuracy
number lives in bench_full.py (external, independently-labeled). Use this to catch
cluster-boundary regressions (guilt vs blame vs love-bomb) as the taxonomy evolves.

    python tests/interpersonal_report.py
"""
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data" / "interpersonal_test.jsonl"
URL = "http://127.0.0.1:8765/classify"


def classify(text: str) -> dict:
    req = urllib.request.Request(
        URL, data=json.dumps({"text": text}).encode(),
        headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def main() -> int:
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    per = defaultdict(lambda: [0, 0])
    misses = []
    for r in rows:
        d = classify(r["text"])
        ok = d["vector"] == r["label"]
        per[r["label"]][0] += ok
        per[r["label"]][1] += 1
        if not ok:
            misses.append((r["label"], d["vector"], d["confidence"], r["text"]))
    correct = sum(c for c, _ in per.values())
    total = sum(n for _, n in per.values())
    print(f"Interpersonal regression: {correct}/{total} = {correct/total*100:.0f}%\n")
    for lab in sorted(per):
        c, n = per[lab]
        print(f"  {lab:32} {c}/{n}")
    if misses:
        print("\nConfusions:")
        for exp, got, conf, text in misses:
            print(f"  exp={exp:30} got={got:30} {conf:>3}%  {text[:55]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
