"""Build a LEAKAGE-FREE LoRA training set for the technique classifier.

Train/valid come from data we may tune on — self-authored sets + external_DEV. The
external_HOLDOUT (150) is NEVER touched here, so `bench_full.py --data external_holdout`
remains an honest generalization estimate after fine-tuning.

Output (mlx_lm chat format, one JSON object per line):
  {"messages": [{"role":"system",...},{"role":"user",text},{"role":"assistant",label}]}

We train with the SAME system prompt used at inference (definitions + tie-break) and
`--mask-prompt`, so the adapter only learns the label completion — consistent with the
log-prob label-scoring path. Run:  python lora/prepare_data.py
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "tests" / "data"
OUT = HERE / "data"

# Import the live SYSTEM prompt so training context == inference context.
import sys
sys.path.insert(0, str(HERE.parent))
from server import SYSTEM, VECTORS  # noqa: E402

# Sources we ARE allowed to fit on (self-authored + external DEV + the DISJOINT Mathur train
# split). external_HOLDOUT and external_mathur (test) are excluded → both stay honest tests.
TRAIN_SOURCES = ["eval_set", "realistic_set", "interpersonal_test", "external_dev", "external_mathur_train"]
SEED = 20260807  # fixed (Math.random/Date unavailable-equivalent discipline: deterministic split)


def load(name):
    p = DATA / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def to_chat(row):
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["text"]},
        {"role": "assistant", "content": row["label"]},
    ]}


def main():
    rows, seen = [], set()
    for src in TRAIN_SOURCES:
        for r in load(src):
            key = r["text"].strip()
            if key in seen:
                continue
            if r["label"] not in VECTORS:
                continue
            seen.add(key)
            rows.append(r)
    rng = random.Random(SEED)
    rng.shuffle(rows)

    n_valid = max(20, len(rows) // 10)
    valid, train = rows[:n_valid], rows[n_valid:]

    OUT.mkdir(exist_ok=True)
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(to_chat(r)) for r in train) + "\n")
    (OUT / "valid.jsonl").write_text("\n".join(json.dumps(to_chat(r)) for r in valid) + "\n")

    from collections import Counter
    dist = Counter(r["label"] for r in rows)
    print(f"train {len(train)}  valid {len(valid)}  (holdout EXCLUDED)")
    print("class distribution:")
    for k in sorted(dist):
        print(f"  {k:32} {dist[k]}")


if __name__ == "__main__":
    main()
