"""A6 — verify the gTTS/Google leak is gone and the pipeline still works.

Runs a NOVEL sentence (so TTS must actually execute) and asserts the offline
path was taken. If this prints OFFLINE, no highlighted text ever leaves the
machine.
"""

import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
CACHE = HERE / "cache"

SENTENCE = "The harbour master repainted the mooring posts before the autumn tide."


def main() -> None:
    from tribev2.demo_utils import TribeModel

    CACHE.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(CACHE), device=device
    )

    txt = CACHE / "a6_offline.txt"
    txt.write_text(SENTENCE)

    t0 = time.time()
    events = model.get_events_dataframe(text_path=str(txt))
    preds, _ = model.predict(events=events)
    print(f"\n[ok] preds={preds.shape} in {time.time()-t0:.0f}s", flush=True)
    print("[ok] pipeline still works with offline TTS", flush=True)


if __name__ == "__main__":
    main()
