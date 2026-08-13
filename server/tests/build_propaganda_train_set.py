"""Build a TRAINING set of the starved propaganda vectors (fear / outrage / tribal / authority /
critical-thinking / hype) to feed the LoRA fine-tune, which was data-starved on exactly these.

Source: HuggingFace `anismahmahi/10_techniques_train` (`cleaned_hf_10_techniques_train.csv`) — the
disjoint TRAIN sibling of `anismahmahi/10_techniques_test`, which we already use (as a held-out
BENCH) to build `external_propaganda.jsonl`. Same SemEval-2020/PTC source, provider-split into
train/test, so training on the train split does NOT leak into that propaganda bench, and it never
touches `external_test` at all.

Same technique->vector MAP as `build_propaganda_set.py`; higher PER_CLASS since this is training
data (the whole point is to lift the starved classes). Same noise caveats (span fragments; the
exaggeration->hype mapping is loose) — kept honest, and prepare_data.py still holds out the
external sets, so the honest external benchmark is unaffected.

    python tests/build_propaganda_train_set.py > data/external_propaganda_train.jsonl
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
PER_CLASS = 100
MIN_WORDS = 5


def main():
    p = hf_hub_download("anismahmahi/10_techniques_train",
                        "cleaned_hf_10_techniques_train.csv", repo_type="dataset")
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
