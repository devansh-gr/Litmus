"""I1b -- validate the SWITCHED text-only /brainmap end-to-end.

After switching /brainmap to CPD_BRAINMAP_MODE=text and rebuilding baseline.npz
text-only, confirm the product's core honesty claim still holds: emotional text
must still separate from neutral in the CURATED semantic ROIs that /brainmap
reports, at roughly A7's audio-path d=0.95. If it doesn't, the switch broke the
interpretation and must be reverted.

This is the REAL pipeline, not a probe: same text_only_events(), the same
committed baseline.npz z-scoring, and the same SEMANTIC_ROIS / CORTICAL_SYSTEMS
that server.py serves. Test sentences are HELD OUT (none are in the baseline
corpus). Run from the server/ dir:  .venv/bin/python -u experiments/verify_text_only_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from tribe_events import text_only_events, harden_tribe
from server import SEMANTIC_ROIS, CORTICAL_SYSTEMS, get_atlas

HERE = Path(__file__).resolve().parent.parent

# Held-out fear sentences (from the I1 probe).
FEAR = [
    "Terrifying new evidence proves the outbreak will devastate every family you love.",
    "The killer is still out there and the police cannot protect you.",
    "A deadly toxin has been found in the water supply of your neighbourhood.",
    "The economy is on the brink of a catastrophe that will ruin everyone.",
    "Your children are being targeted by predators hiding in plain sight online.",
]
# Held-out NEUTRAL sentences -- deliberately NOT any of the 30 baseline sentences.
NEUTRAL = [
    "The delivery van stops at the depot twice a day to collect parcels.",
    "A new coat of paint was applied to the fence behind the parking lot.",
    "The spreadsheet lists the room numbers for each of the afternoon sessions.",
    "He replaced the batteries in the wall clock above the kitchen counter.",
    "The catalogue groups the tools by size and lists their part numbers.",
]


def main():
    from tribev2.demo_utils import TribeModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = harden_tribe(TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(HERE / "cache"), device=device))
    print(f">> TRIBE loaded (num_workers={model.data.num_workers})", flush=True)

    b = np.load(HERE / "baseline.npz")
    assert "mode" in b.files and str(b["mode"]) == "text", \
        f"baseline.npz is not text-mode: {b.files}"
    print(f">> baseline: n={int(b['n'])} mode={str(b['mode'])}", flush=True)

    labels, annot = get_atlas()
    idx = {n: i for i, n in enumerate(labels)}

    def region_z(zvert, name):
        i = idx.get(name)
        if i is None:
            return None
        mask = annot == i
        return float(zvert[mask].mean()) if mask.sum() else None

    def semantic_engagement(sentence):
        preds, _ = model.predict(events=text_only_events(sentence), verbose=False)
        zvert = (np.asarray(preds).mean(axis=0) - b["mean"]) / b["std"]
        zs = [region_z(zvert, n) for n in SEMANTIC_ROIS]
        zs = [z for z in zs if z is not None]
        sysvals = {sysname: float(np.mean([region_z(zvert, n) for n in rois]))
                   for sysname, rois in CORTICAL_SYSTEMS.items()}
        return float(np.mean(zs)), sysvals

    fe, fsys = zip(*[semantic_engagement(s) for s in FEAR])
    ne, nsys = zip(*[semantic_engagement(s) for s in NEUTRAL])
    fe, ne = np.array(fe), np.array(ne)

    pooled = np.sqrt((fe.var(ddof=1) + ne.var(ddof=1)) / 2)
    d = (fe.mean() - ne.mean()) / max(pooled, 1e-9)

    print("\n=== semantic-ROI engagement (z vs neutral text-only baseline) ===")
    print(f"  fear    mean = {fe.mean():+.3f}   (per-sentence: {np.round(fe,2)})")
    print(f"  neutral mean = {ne.mean():+.3f}   (per-sentence: {np.round(ne,2)})")
    print(f"  Cohen's d (fear - neutral) = {d:+.2f}   [A7 audio-path was +0.95]")

    print("\n=== per-system engagement (fear vs neutral) ===")
    for sysname in CORTICAL_SYSTEMS:
        f = np.mean([s[sysname] for s in fsys])
        n = np.mean([s[sysname] for s in nsys])
        print(f"  {sysname:42s} fear {f:+.3f}  neutral {n:+.3f}  Δ {f-n:+.3f}")

    verdict = "PASS" if d >= 0.6 else "FAIL"
    print(f"\nVERIFY_TEXT_ONLY: {verdict} (d={d:+.2f})")


if __name__ == "__main__":
    main()
