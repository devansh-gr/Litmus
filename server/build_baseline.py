"""Build the per-vertex baseline that makes /brainmap meaningful.

THE PROBLEM: TRIBE v2's text path synthesises speech, so raw activation for ANY
sentence is dominated by primary auditory cortex responding to "there is speech"
-- an artifact of our TTS step, not the content. We z-score new text against this
baseline so only CONTENT-driven deviation survives.

CRASH-RESILIENT: predictions are checkpointed to baseline_partial.npz after every
sentence. On restart it skips sentences already done. A hang (see the num_workers
deadlock) can now cost at most one sentence, not the whole run.
"""

import time
from pathlib import Path

import numpy as np
import torch

from a3_emotion_test import CONDITIONS

HERE = Path(__file__).parent
CACHE = HERE / "cache"
PARTIAL = HERE / "baseline_partial.npz"
FINAL = HERE / "baseline.npz"


def main() -> None:
    from tribev2.demo_utils import TribeModel

    CACHE.mkdir(exist_ok=True)
    sentences = [s for sents in CONDITIONS.values() for s in sents]

    done: dict[int, np.ndarray] = {}
    if PARTIAL.exists():
        d = np.load(PARTIAL)
        for k in d.files:
            if k.startswith("v"):
                done[int(k[1:])] = d[k]
        print(f"resuming: {len(done)}/{len(sentences)} already computed", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(CACHE), device=device
    )

    t0 = time.time()
    for i, sent in enumerate(sentences):
        if i in done:
            continue
        txt = CACHE / f"a3_baseline_{i}.txt"
        txt.write_text(sent)
        events = model.get_events_dataframe(text_path=str(txt))
        preds, _ = model.predict(events=events, verbose=False)
        done[i] = np.asarray(preds).mean(axis=0)
        # checkpoint immediately
        np.savez(PARTIAL, **{f"v{k}": v for k, v in done.items()})
        print(f"  {len(done)}/{len(sentences)}  ({time.time()-t0:.0f}s)", flush=True)

    X = np.stack([done[i] for i in range(len(sentences))])
    np.savez(FINAL, mean=X.mean(axis=0), std=np.maximum(X.std(axis=0), 1e-9), n=len(sentences))
    print(f"\n[saved] {FINAL}  from {len(sentences)} sentences", flush=True)


if __name__ == "__main__":
    main()
