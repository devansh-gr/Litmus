#!/usr/bin/env python3
"""Build an EXTERNAL, independently-labeled test set from public datasets.

Escapes the self-authored-data trap: these examples were written and labeled by other
people, and we never tuned prompts on them — so it's a real generalization test.

Sources (downloaded to a temp dir, paths passed as args):
  - Yamana ec-darkpattern (e-commerce copy): Urgency→false-urgency, Scarcity→fomo,
    Social Proof→social-proof-conformity, Not Dark Pattern→none.
  - LOGIC logical-fallacy (Jin et al. 2022): ad populum→social-proof-conformity,
    fallacy of credibility→authority-appeal.

CAVEATS baked into the output (documented, not hidden):
  - Cross-taxonomy mapping: other people's label definitions ≈ but ≠ ours.
  - Domain shift: e-commerce / debate text, not arbitrary highlighted text.
  - Coverage: only 4 manipulation vectors + none have a clean external label;
    fear/outrage/tribal/crit-think/dopamine/hype/awe are NOT covered here.

    python tests/build_external_set.py <darkpattern.tsv> <logic_dir> > data/external_test.jsonl
"""
import csv
import json
import random
import sys
from pathlib import Path

random.seed(0)  # reproducible sample

DP_MAP = {
    "Urgency": "false-urgency",
    "Scarcity": "fomo",
    "Social Proof": "social-proof-conformity",
    "Not Dark Pattern": "none",
}
LOGIC_MAP = {
    "ad populum": "social-proof-conformity",
    "fallacy of credibility": "authority-appeal",
}
CAPS = {  # keep the eval runnable + roughly balanced
    "none": 80, "false-urgency": 40, "fomo": 40,
    "social-proof-conformity": 50, "authority-appeal": 40,
}


def clean(t: str) -> str:
    return " ".join(t.split()).strip()


def main():
    dp_path, logic_dir = sys.argv[1], sys.argv[2]
    buckets: dict[str, list[dict]] = {v: [] for v in CAPS}
    seen = set()

    def add(text, vector, source):
        text = clean(text)
        if len(text) < 15 or len(text) > 300:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        buckets[vector].append({"text": text, "label": vector, "source": source})

    # dark patterns
    with open(dp_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            v = DP_MAP.get((row.get("Pattern Category") or "").strip())
            if v:
                add(row.get("text", ""), v, "dark-patterns")

    # LOGIC
    for name in ("logic_edu_train.csv", "logic_edu_dev.csv", "logic_edu_test.csv"):
        p = Path(logic_dir) / name
        if not p.exists():
            continue
        for row in csv.DictReader(open(p)):
            v = LOGIC_MAP.get((row.get("updated_label") or "").strip())
            if v:
                add(row.get("source_article", ""), v, "logic")

    out = []
    for v, cap in CAPS.items():
        items = buckets[v]
        random.shuffle(items)
        out.extend(items[:cap])
    random.shuffle(out)
    for row in out:
        print(json.dumps(row))

    # summary to stderr
    from collections import Counter
    c = Counter(r["label"] for r in out)
    print(f"built {len(out)} external examples: {dict(c)}", file=sys.stderr)


if __name__ == "__main__":
    main()
