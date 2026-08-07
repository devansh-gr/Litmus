"""Build a FRESH external test set from Mathur et al. 2019, "Dark Patterns at Scale".

Why this one: we tuned our fomo/social-proof/false-urgency behaviour partly on the *Yamana*
e-commerce dark-pattern corpus. Mathur is an INDEPENDENT collection (different sites, different
strings), so measuring on it is a real over-fitting probe on exactly our best-tuned vectors —
if accuracy craters vs our Yamana-derived `external_test`, we memorized phrasings, not patterns.

Source (Apache-licensed public repo, downloaded to a temp path, passed as arg):
  github.com/aruneshmathur/dark-patterns  data/final-dark-patterns/dark-patterns.csv
Mapping (same category names as Yamana, so the mapping is unambiguous):
  Scarcity -> fomo | Urgency -> false-urgency | Social Proof -> social-proof-conformity
  Confirmshaming (Pattern Type) -> guilt-tripping  (shame-worded decline options)

    python tests/build_mathur_set.py <dark-patterns.csv> > data/external_mathur.jsonl
"""
import csv
import json
import random
import re
import sys
from collections import defaultdict

random.seed(0)
csv.field_size_limit(10_000_000)

CAT_MAP = {"Scarcity": "fomo", "Urgency": "false-urgency", "Social Proof": "social-proof-conformity"}
PER_CLASS = 80  # cap per class so the bench stays quick + balanced


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def main(path):
    buckets = defaultdict(list)
    seen = set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            text = norm(r.get("Pattern String", ""))
            cat = (r.get("Pattern Category") or "").strip()
            typ = (r.get("Pattern Type") or "").strip()
            if len(text) < 4 or len(text) > 200:
                continue
            label = CAT_MAP.get(cat)
            if typ == "Confirmshaming":
                label = "guilt-tripping"
            if not label:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            buckets[label].append(text)

    rows = []
    for label, texts in buckets.items():
        random.shuffle(texts)
        for t in texts[:PER_CLASS]:
            rows.append({"text": t, "label": label})
    random.shuffle(rows)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    # summary to stderr (doesn't pollute the jsonl on stdout)
    dist = defaultdict(int)
    for r in rows:
        dist[r["label"]] += 1
    print("distribution:", dict(dist), f"total={len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
