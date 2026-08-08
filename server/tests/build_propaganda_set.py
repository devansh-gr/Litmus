"""Build a FRESH external test set covering the vectors our benchmark LACKED.

Our external_test/mathur cover only marketing vectors (fomo/urgency/social-proof/dopamine).
This adds the missing PROPAGANDA vectors — fear, outrage, tribal, authority, critical-thinking,
hype — from an independent public corpus we never tuned on.

Source (found via the dataset-discovery workflow, then VERIFIED by hand: real files, flat
input->output labels, Apache-2.0, non-gated):
  HuggingFace `anismahmahi/10_techniques_test` (SemEval-2020/PTC propaganda spans, cleaned to
  one text + one technique label per row).
Mapping (SemEval technique -> our vector):
  appeal to fear            -> fear-mongering
  flag waving               -> tribal-in-group-bias
  appeal to authority       -> authority-appeal
  Doubt                     -> critical-thinking-suppression
  loaded language / name calling/labeling -> outrage
  exaggeration/minimisation -> hype-hope-mongering   (loose: exaggeration ~ hype)

CAVEATS (documented, not hidden): these are span fragments extracted from news articles, so
some are short/context-dependent (noise); we drop <5-word fragments. The exaggeration->hype
mapping is loose. Different domain (political news) from our e-commerce/debate sets.

    python tests/build_propaganda_set.py > data/external_propaganda.jsonl
"""
import csv
import json
import random
import re
import sys
from collections import defaultdict

from huggingface_hub import hf_hub_download

random.seed(0)
csv.field_size_limit(10_000_000)

MAP = {
    "appeal to fear fallacy": "fear-mongering",
    "flag waving fallacy": "tribal-in-group-bias",
    "appeal to authority fallacy": "authority-appeal",
    "Doubt fallacy": "critical-thinking-suppression",
    "loaded language fallacy": "outrage",
    "name calling/labeling fallacy": "outrage",
    "exaggeration/minimisation fallacy": "hype-hope-mongering",
}
PER_CLASS = 60
MIN_WORDS = 5


def main():
    p = hf_hub_download("anismahmahi/10_techniques_test",
                        "cleaned_hf_10_techniques.csv", repo_type="dataset")
    buckets = defaultdict(list)
    seen = set()
    with open(p, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            text = re.sub(r"\s+", " ", (r.get("input") or "")).strip()
            label = MAP.get((r.get("output") or "").strip())
            if not label or len(text.split()) < MIN_WORDS or len(text) > 300:
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
    dist = defaultdict(int)
    for r in rows:
        dist[r["label"]] += 1
    print("distribution:", dict(dist), f"total={len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
