"""A4 — THE CONTROL THAT DECIDES WHETHER THE BRAIN LAYER EARNS ITS PLACE.

TRIBE v2's brain map is a DETERMINISTIC FUNCTION OF THE TEXT. By the data-
processing inequality it cannot contain information the text does not already
carry. So a text-only classifier is an UPPER BOUND on what brain-based decoding
can achieve.

If a dumb bag-of-words model matches or beats the 75% 4-way accuracy we got from
the predicted brain maps, then the brain layer adds ZERO detection power, and its
only remaining value is *interpretability* (where in cortex the content acts).

Same 60 sentences, same CV scheme, same classifier family as A3.
"""

import json
from pathlib import Path

import numpy as np

from a3_emotion_test import CONDITIONS

HERE = Path(__file__).parent
OUT = HERE / "a4_results.json"

# Brain-based numbers from A3, for a like-for-like comparison.
BRAIN = {
    "4way": 0.750,
    "fear_vs_neutral": 0.833,
    "outrage_vs_neutral": 1.000,
    "reward_vs_neutral": 0.733,
    "fear_vs_outrage": 0.933,
    "fear_vs_reward": 0.767,
    "outrage_vs_reward": 0.867,
}


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline

    texts, y = [], []
    for cond, sents in CONDITIONS.items():
        texts.extend(sents)
        y.extend([cond] * len(sents))
    texts = np.array(texts)
    y = np.array(y)
    print(f"{len(texts)} sentences, {len(set(y))} conditions\n")

    clf = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words="english"),
        LogisticRegression(max_iter=5000, C=1.0),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    results = {}
    acc = cross_val_score(clf, texts, y, cv=cv, scoring="accuracy").mean()
    results["4way"] = float(acc)

    print("=== TEXT-ONLY (TF-IDF bag-of-words) vs BRAIN-BASED (TRIBE v2) ===\n")
    print(f"{'comparison':<24} {'TEXT':>7}  {'BRAIN':>7}   {'chance':>6}")
    print(f"{'4-way':<24} {acc*100:>6.1f}% {BRAIN['4way']*100:>6.1f}%   {25:>5}%")

    emos = ["fear", "outrage", "reward"]
    pairs = [(e, "neutral") for e in emos] + [
        ("fear", "outrage"), ("fear", "reward"), ("outrage", "reward")
    ]
    for a, b in pairs:
        m = np.isin(y, [a, b])
        s = cross_val_score(clf, texts[m], y[m], cv=cv, scoring="accuracy").mean()
        key = f"{a}_vs_{b}"
        results[key] = float(s)
        print(f"{key:<24} {s*100:>6.1f}% {BRAIN[key]*100:>6.1f}%   {50:>5}%")

    deltas = [results[k] - BRAIN[k] for k in BRAIN]
    mean_delta = float(np.mean(deltas))
    print(f"\nmean advantage of TEXT over BRAIN: {mean_delta*100:+.1f} points")
    print(
        "\nVERDICT: "
        + (
            "text-only MATCHES/BEATS brain decoding => the brain layer adds NO\n"
            "         detection power. Its only value is interpretability."
            if mean_delta >= -0.02
            else "brain decoding beats text-only => unexpected; investigate\n"
            "         (should be impossible: brain map is a function of the text)."
        )
    )

    OUT.write_text(json.dumps(
        {"text_only": results, "brain_based": BRAIN, "mean_delta": mean_delta}, indent=2
    ))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
