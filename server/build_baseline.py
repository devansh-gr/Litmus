"""Build the per-vertex baseline that makes /brainmap meaningful.

THE PROBLEM: even TRIBE v2's TEXT path (word events, no audio) predicts strong
primary-auditory activation for ANY sentence -- that dominance is intrinsic to
the model (trained on speech), not the input (I1). We z-score new text against
this baseline so only CONTENT-driven deviation survives.

THE BASELINE MUST BE NEUTRAL TEXT. If the baseline contains emotional sentences,
z-scoring subtracts the emotional signal too, leaving only sensorimotor noise
(learned the hard way). A neutral baseline makes emotional text deviate in the
fronto-orbital / value regions that A2-A3 validated against published fMRI.

THE BASELINE MUST MATCH THE QUERY-TIME MODE. Set CPD_BRAINMAP_MODE the same way
here as when serving (default "text"). The partial checkpoint and final file are
mode-tagged so an audio baseline is never silently mixed with a text one.

CRASH-RESILIENT: checkpoints after every sentence and resumes on restart, so a
hang costs at most one sentence.
"""

import time
from pathlib import Path

import numpy as np
import torch

from tribe_events import BRAINMAP_MODE, build_events, harden_tribe

HERE = Path(__file__).parent
CACHE = HERE / "cache"
PARTIAL = HERE / f"baseline_partial_{BRAINMAP_MODE}.npz"
FINAL = HERE / "baseline.npz"

# Neutral / informational reference corpus. Deliberately varied in topic and
# syntax, no emotional or persuasive content. The first 15 are the A3 neutral set
# (already cached), the rest are fresh.
NEUTRAL = [
    "The council approved the new drainage plan for the eastern district on Tuesday.",
    "The library will extend its opening hours during the spring term this year.",
    "She placed the folded map back inside the glove compartment of the car.",
    "The train arrives at the central station every twenty minutes during the day.",
    "He watered the plants on the balcony before sitting down to read.",
    "The museum has rearranged its collection of pottery in the eastern wing.",
    "Rainfall this month was slightly above the seasonal average for the region.",
    "The committee will publish its annual report at the end of the quarter.",
    "The bakery on the corner opens at six every morning except Sunday.",
    "A new bicycle lane was added along the river path last autumn.",
    "The lecture covered the basic principles of sedimentary rock formation.",
    "She filed the documents alphabetically in the cabinet beside her desk.",
    "The ferry crosses the harbour four times a day during the summer.",
    "The software update adds support for two additional keyboard layouts.",
    "A wooden bench was installed near the entrance of the public garden.",
    "The recipe suggests letting the dough rest for about thirty minutes.",
    "Their office moved to the third floor of the building last month.",
    "The bus timetable changes slightly on public holidays and weekends.",
    "He labelled each box before stacking them neatly in the storage room.",
    "The report summarises rainfall and temperature data for the past decade.",
    "The gardener trimmed the hedges along the northern edge of the lawn.",
    "The store restocks fresh produce on Monday and Thursday mornings.",
    "A short footbridge connects the two halves of the campus over the stream.",
    "The printer on the second floor supports double-sided colour printing.",
    "She catalogued the samples by date and stored them in the freezer.",
    "The seminar will be held in the annex next to the main lecture hall.",
    "The road resurfacing work is scheduled to finish before the end of March.",
    "He adjusted the thermostat and closed the window before leaving the room.",
    "The archive contains maps of the town dating back to the last century.",
    "The workshop covers basic maintenance for common household appliances.",
]


def main() -> None:
    from tribev2.demo_utils import TribeModel

    CACHE.mkdir(exist_ok=True)

    done: dict[int, np.ndarray] = {}
    if PARTIAL.exists():
        d = np.load(PARTIAL)
        for k in d.files:
            if k.startswith("v"):
                done[int(k[1:])] = d[k]
        print(f"resuming: {len(done)}/{len(NEUTRAL)} already computed", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(CACHE), device=device
    )
    harden_tribe(model)  # num_workers=0 -> no forked-worker deadlock (froze at 2/30)
    print(f"[mode] CPD_BRAINMAP_MODE={BRAINMAP_MODE} | num_workers={model.data.num_workers}",
          flush=True)

    t0 = time.time()
    for i, sent in enumerate(NEUTRAL):
        if i in done:
            continue
        txt = CACHE / f"neutral_baseline_{i}.txt"
        txt.write_text(sent)
        events = build_events(model, sent, str(txt))  # mode-dispatched (text|audio)
        preds, _ = model.predict(events=events, verbose=False)
        done[i] = np.asarray(preds).mean(axis=0)
        np.savez(PARTIAL, **{f"v{k}": v for k, v in done.items()})
        print(f"  {len(done)}/{len(NEUTRAL)}  ({time.time()-t0:.0f}s)", flush=True)

    X = np.stack([done[i] for i in range(len(NEUTRAL))])
    np.savez(FINAL, mean=X.mean(axis=0), std=np.maximum(X.std(axis=0), 1e-9),
             n=len(NEUTRAL), mode=BRAINMAP_MODE)
    print(f"\n[saved] {FINAL}  from {len(NEUTRAL)} NEUTRAL sentences ({BRAINMAP_MODE} mode)",
          flush=True)


if __name__ == "__main__":
    main()
