"""A5 — THE FAIR CONTROL.

A4 compared the brain map against TF-IDF, which was a strawman: bag-of-words
cannot generalise to unseen vocabulary, so it lost by 17 points. That told us
nothing.

The honest question is narrower and sharper:

    TRIBE v2's brain map is  text -> Llama embedding -> brain projection.

    So: does the BRAIN PROJECTION step add anything, or is all the decoding
    power already present in the raw Llama embedding it was built from?

If Llama embeddings decode emotion as well as (or better than) the brain map,
then the brain step contributes NO detection power, and its only value is
interpretability (telling you WHERE in cortex the content acts). That is still a
legitimate product -- but it must be sold as interpretation, not detection.

Same 60 sentences, same CV, same classifier as A3.
"""

import json
from pathlib import Path

import numpy as np
import torch

from a3_emotion_test import CONDITIONS

HERE = Path(__file__).parent
OUT = HERE / "a5_results.json"

BRAIN = {
    "4way": 0.750,
    "fear_vs_neutral": 0.833,
    "outrage_vs_neutral": 1.000,
    "reward_vs_neutral": 0.733,
    "fear_vs_outrage": 0.933,
    "fear_vs_reward": 0.767,
    "outrage_vs_reward": 0.867,
}

MODEL = "meta-llama/Llama-3.2-3B"


def embed(texts):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    vecs = []
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok(t, return_tensors="pt")
            out = mdl(**enc).last_hidden_state[0]   # (tokens, hidden)
            vecs.append(out.mean(dim=0).numpy())    # mean-pool, same as TRIBE's aggregation
            if (i + 1) % 15 == 0:
                print(f"  embedded {i+1}/{len(texts)}", flush=True)
    return np.stack(vecs)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    texts, y = [], []
    for cond, sents in CONDITIONS.items():
        texts.extend(sents)
        y.extend([cond] * len(sents))
    y = np.array(y)

    print(f"embedding {len(texts)} sentences with {MODEL}...", flush=True)
    E = embed(texts)
    print(f"embeddings: {E.shape}\n", flush=True)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    results = {}
    print("=== LLAMA EMBEDDING (text) vs TRIBE v2 BRAIN MAP ===\n")
    print(f"{'comparison':<22} {'LLAMA':>7}  {'BRAIN':>7}  {'delta':>7}")

    acc = cross_val_score(clf, E, y, cv=cv, scoring="accuracy").mean()
    results["4way"] = float(acc)
    print(f"{'4-way':<22} {acc*100:>6.1f}% {BRAIN['4way']*100:>6.1f}% "
          f"{(BRAIN['4way']-acc)*100:>+6.1f}")

    emos = ["fear", "outrage", "reward"]
    pairs = [(e, "neutral") for e in emos] + [
        ("fear", "outrage"), ("fear", "reward"), ("outrage", "reward")
    ]
    for a, b in pairs:
        m = np.isin(y, [a, b])
        s = cross_val_score(clf, E[m], y[m], cv=cv, scoring="accuracy").mean()
        key = f"{a}_vs_{b}"
        results[key] = float(s)
        print(f"{key:<22} {s*100:>6.1f}% {BRAIN[key]*100:>6.1f}% "
              f"{(BRAIN[key]-s)*100:>+6.1f}")

    delta = float(np.mean([BRAIN[k] - results[k] for k in BRAIN]))
    print(f"\nmean advantage of BRAIN over LLAMA: {delta*100:+.1f} points")
    if delta <= 0.02:
        print("\nVERDICT: the brain projection adds NO detection power.")
        print("         All signal was already in the Llama embedding.")
        print("         => Sell the brain map as INTERPRETATION (where it acts),")
        print("            never as a better detector.")
    else:
        print("\nVERDICT: brain map decodes better than the embedding it derives from.")
        print("         Cannot add information (data-processing inequality), so this")
        print("         means the brain projection is a useful REGULARISER /")
        print("         dimensionality reduction (20484 -> 75 ROIs) on tiny data.")

    OUT.write_text(json.dumps(
        {"llama_text": results, "brain_based": BRAIN, "brain_minus_llama": delta},
        indent=2,
    ))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
