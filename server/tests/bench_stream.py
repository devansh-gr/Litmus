"""Streaming benchmark — resilient to being killed mid-run (e.g. a battery watchdog).

Unlike bench_full.py (which computes metrics only at the very end), this POSTs each example
to the live /classify, writes {gold, pred, ok} to an output JSONL and FLUSHES immediately, and
prints a running accuracy every 10 examples. If it's killed partway, the output file + the last
printed line still give a real accuracy on the examples that finished. Score a partial file later
with:  python tests/bench_stream.py --score <out.jsonl>

Run:  python tests/bench_stream.py --data tests/data/external_test.jsonl --out /tmp/pred.jsonl
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter


def score(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    correct = sum(r["ok"] for r in rows)
    print(f"N={n}  accuracy={100*correct/n:.1f}%  ({correct}/{n})" if n else "empty")
    # per-class recall
    gold_tot, gold_ok = Counter(), Counter()
    for r in rows:
        gold_tot[r["gold"]] += 1
        gold_ok[r["gold"]] += r["ok"]
    for lab in sorted(gold_tot):
        t = gold_tot[lab]
        print(f"  {lab:28s} recall {gold_ok[lab]/t:.2f}  ({gold_ok[lab]}/{t})")


def run(data, out, url, limit):
    rows = [json.loads(l) for l in open(data) if l.strip()]
    if limit:
        rows = rows[:limit]
    correct = n = 0
    with open(out, "w") as f:
        for r in rows:
            text, gold = r.get("text", ""), r.get("label")
            req = urllib.request.Request(
                url, data=json.dumps({"text": text}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                pred = json.loads(urllib.request.urlopen(req, timeout=180).read())["vector"]
            except Exception as e:  # noqa: BLE001
                print(f"  (request failed at n={n}: {e})", flush=True)
                break
            ok = int(pred == gold)
            correct += ok
            n += 1
            f.write(json.dumps({"gold": gold, "pred": pred, "ok": ok}) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"running acc: {correct}/{n} = {100*correct/n:.1f}%", flush=True)
    print(f"FINAL: {correct}/{n} = {100*correct/n:.1f}%" if n else "no examples", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default="/tmp/bench_stream_out.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8765/classify")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score", help="score an existing partial/full output file and exit")
    a = ap.parse_args()
    if a.score:
        score(a.score)
    elif a.data:
        run(a.data, a.out, a.url, a.limit)
    else:
        ap.error("need --data or --score")
