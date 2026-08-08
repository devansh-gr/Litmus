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

    python tests/build_mathur_set.py <dark-patterns.csv>       > data/external_mathur.jsonl        # test (default)
    python tests/build_mathur_set.py <dark-patterns.csv> train > data/external_mathur_train.jsonl  # DISJOINT train split
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
TEST_PER_CLASS = 80    # first 80/class -> external_mathur.jsonl (the held-out fresh TEST)
TRAIN_PER_CLASS = 80   # next 80/class -> external_mathur_train.jsonl (DISJOINT, safe to fine-tune on)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def main(path, split="test"):
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
        random.shuffle(texts)  # seed(0) -> deterministic, so test and train slices are stable + disjoint
        sl = slice(0, TEST_PER_CLASS) if split == "test" else slice(TEST_PER_CLASS, TEST_PER_CLASS + TRAIN_PER_CLASS)
        for t in texts[sl]:
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
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "test")
