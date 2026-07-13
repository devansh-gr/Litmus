"""Measure the TRUE cold per-sentence cost (novel text => no cache hit).

This decides whether A2 (the fear-vs-neutral signal test) is feasible on this
Mac, since A2 needs dozens of unseen sentences.

NOTE: everything must live under a __main__ guard — the DataLoader spawns
worker processes that re-import this module.
"""

import time
from pathlib import Path

import torch

CACHE = Path(__file__).parent / "cache"

# Deliberately novel sentences so nothing is cached.
SENTENCES = [
    "The council approved the new drainage plan for the eastern district on Tuesday.",
    "Terrifying new evidence proves the outbreak will devastate every family you love.",
]


def main() -> None:
    from tribev2.demo_utils import TribeModel

    CACHE.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}", flush=True)

    t0 = time.time()
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(CACHE), device=device
    )
    print(f"model load: {time.time()-t0:.1f}s\n", flush=True)

    for i, sent in enumerate(SENTENCES):
        txt = CACHE / f"timing_{i}.txt"
        txt.write_text(sent)

        t0 = time.time()
        events = model.get_events_dataframe(text_path=str(txt))
        t_events = time.time() - t0

        t0 = time.time()
        preds, _segments = model.predict(events=events)
        t_pred = time.time() - t0

        print(
            f"[{i}] events={t_events:6.1f}s  predict={t_pred:6.1f}s  "
            f"TOTAL={t_events + t_pred:6.1f}s  shape={preds.shape}",
            flush=True,
        )


if __name__ == "__main__":
    main()
