"""Gate recall diagnostic + threshold sweep.

The honest benchmark showed the manipulation gate misses ~44% of real manipulation
(recall 0.56 at GATE_NONE_THRESHOLD=0.60). This tool answers two questions:

  1. WHERE is the threshold's sweet spot? The gate's manip_prob is recorded ONCE per
     example, then the flag cutoff is swept in-memory — so we see the full
     recall/precision/benign-FP trade without re-hitting the server per threshold.
  2. WHAT is it missing? Gate false-negatives (manipulation the gate calls benign)
     broken down by gold class + the manip_prob distribution, so we can tell a loose
     cross-taxonomy mapping (noise) from a real gate weakness (fixable).

Tune on external_dev, report on external_holdout — never sweep on the set you quote.

    python tests/gate_diag.py --data tests/data/external_dev.jsonl
    python tests/gate_diag.py --data tests/data/external_dev.jsonl --cache dev_scores.json
"""
import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

URL = "http://127.0.0.1:8765/classify"


def classify(text: str) -> dict:
    req = urllib.request.Request(
        URL, data=json.dumps({"text": text}).encode(),
        headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def score_set(path: Path, cache: Path | None):
    """Return [{text, gold, manip_prob, pred}]. Uses cache if present (keyed by text)."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cached = {}
    if cache and cache.exists():
        cached = {r["text"]: r for r in json.loads(cache.read_text())}
    out = []
    for i, r in enumerate(rows):
        text, gold = r["text"], r["label"]
        if text in cached:
            out.append(cached[text])
            continue
        d = classify(text)
        out.append({"text": text, "gold": gold,
                    "manip_prob": d.get("manip_prob", 1.0 if d["vector"] != "none" else 0.0),
                    "pred": d["vector"]})
        if (i + 1) % 25 == 0:
            print(f"  scored {i+1}/{len(rows)}")
    if cache:
        cache.write_text(json.dumps(out, indent=0))
    return out


def sweep(recs):
    manip = [r for r in recs if r["gold"] != "none"]
    benign = [r for r in recs if r["gold"] == "none"]
    print(f"\nthreshold sweep  (manip={len(manip)}  benign={len(benign)})")
    print(f"{'GATE_NONE':>10} {'flag>':>6} {'recall':>7} {'precision':>10} {'F1':>6} {'F0.5':>6} {'benignFP':>9}")
    best = None
    for gnt in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        c = round(1 - gnt, 2)                       # flag if manip_prob > c
        tp = sum(r["manip_prob"] > c for r in manip)
        fp = sum(r["manip_prob"] > c for r in benign)
        recall = tp / len(manip) if manip else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
        f05 = 1.25 * prec * recall / (0.25 * prec + recall) if (prec + recall) else 0
        fpr = fp / len(benign) if benign else 0
        print(f"{gnt:>10.2f} {c:>6.2f} {recall:>7.2f} {prec:>10.2f} {f1:>6.2f} {f05:>6.2f} {fpr:>9.0%}")
        # pick max recall subject to benign-FP <= 30%
        if fpr <= 0.30 and (best is None or recall > best[1]):
            best = (gnt, recall, fpr)
    if best:
        print(f"\n-> recall-max @ benign-FP<=30%: GATE_NONE_THRESHOLD={best[0]:.2f} "
              f"(recall {best[1]:.2f}, benignFP {best[2]:.0%})")


def false_negatives(recs, gnt=0.60):
    c = 1 - gnt
    fns = [r for r in recs if r["gold"] != "none" and r["manip_prob"] <= c]
    manip = [r for r in recs if r["gold"] != "none"]
    by_class = defaultdict(lambda: [0, 0])
    for r in manip:
        by_class[r["gold"]][1] += 1
        if r["manip_prob"] <= c:
            by_class[r["gold"]][0] += 1
    print(f"\ngate false-negatives @ GATE_NONE_THRESHOLD={gnt} (manip called benign): {len(fns)}/{len(manip)}")
    for cls in sorted(by_class):
        miss, tot = by_class[cls]
        print(f"  {cls:32} missed {miss}/{tot}")
    print("\nworst misses (lowest manip_prob):")
    for r in sorted(fns, key=lambda r: r["manip_prob"])[:12]:
        print(f"  mp={r['manip_prob']:.3f}  {r['gold']:22} {r['text'][:60]!r}")


def false_positives(recs, gnt=0.65):
    """The precision side: benign examples the gate flags. If a recall fix started
    catching neutral stats / clocks / news, they show up here."""
    c = 1 - gnt
    benign = [r for r in recs if r["gold"] == "none"]
    fps = [r for r in benign if r["manip_prob"] > c]
    print(f"\ngate false-positives @ GATE_NONE_THRESHOLD={gnt} (benign flagged): {len(fps)}/{len(benign)}")
    for r in sorted(fps, key=lambda r: -r["manip_prob"])[:12]:
        print(f"  mp={r['manip_prob']:.3f}  -> {r['pred']:22} {r['text'][:60]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    recs = score_set(Path(args.data), Path(args.cache) if args.cache else None)
    sweep(recs)
    false_negatives(recs)
    false_positives(recs)


if __name__ == "__main__":
    main()
