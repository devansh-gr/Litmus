"""A1 smoke test: can TRIBE v2 load and predict from text on this Mac?

Tries MPS (Apple GPU) first, falls back to CPU. Prints prediction shape.
NOTE: the text path still uses gTTS (Google) — benign test sentence only.
"""

import time
import traceback
from pathlib import Path

import torch

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

TEXT = "The storm is coming and everyone must prepare for what happens next."


def run(device: str):
    from tribev2.demo_utils import TribeModel

    print(f"\n=== attempting device={device} ===", flush=True)

    t0 = time.time()
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(CACHE), device=device
    )
    print(f"[ok] model loaded in {time.time()-t0:.1f}s", flush=True)

    txt = CACHE / "sample.txt"
    txt.write_text(TEXT)

    t0 = time.time()
    events = model.get_events_dataframe(text_path=str(txt))
    print(f"[ok] events: {events.shape} in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    preds, segments = model.predict(events=events)
    print(f"[ok] PREDICTIONS: shape={preds.shape} in {time.time()-t0:.1f}s", flush=True)
    print(f"[ok] n_segments={len(segments)}", flush=True)
    print(f"\n*** SUCCESS on {device} ***", flush=True)
    return True


if __name__ == "__main__":
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")

    for dev in devices:
        try:
            run(dev)
            break
        except Exception:
            print(f"\n!!! FAILED on {dev} !!!", flush=True)
            traceback.print_exc()
            if dev == devices[-1]:
                print("\n*** all devices failed ***", flush=True)
