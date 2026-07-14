"""Build the per-vertex baseline needed to make /brainmap meaningful.

THE PROBLEM: TRIBE v2's text path synthesises speech, so the raw predicted
activation for ANY sentence is dominated by primary auditory cortex simply
responding to "there is speech" -- an artifact of our TTS step, not of the
content. Reporting it would be a lie (the user READS the text).

THE FIX: compute mean/std per vertex across a reference corpus. At inference we
z-score a new sentence against that baseline, which cancels the generic speech
response and leaves only CONTENT-driven deviation. This is exactly what made the
A2/A3 experiments valid.

Writes: baseline.npz  (mean, std over 20484 fsaverage5 vertices)
"""

import time
from pathlib import Path

import numpy as np
import torch

from a3_emotion_test import CONDITIONS

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE / "baseline.npz"


def main() -> None:
    from tribev2.demo_utils import TribeModel

    CACHE.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(CACHE), device=device
    )

    sentences = [s for sents in CONDITIONS.values() for s in sents]
    print(f"building baseline from {len(sentences)} reference sentences", flush=True)

    vecs = []
    t0 = time.time()
    for i, sent in enumerate(sentences):
        txt = CACHE / f"a3_baseline_{i}.txt"
        txt.write_text(sent)
        events = model.get_events_dataframe(text_path=str(txt))
        preds, _ = model.predict(events=events)
        vecs.append(np.asarray(preds).mean(axis=0))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sentences)}  ({time.time()-t0:.0f}s)", flush=True)

    X = np.stack(vecs)
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-9)
    np.savez(OUT, mean=mean, std=std, n=len(sentences))
    print(f"\n[saved] {OUT}  mean={mean.shape} std={std.shape} n={len(sentences)}")


if __name__ == "__main__":
    main()
