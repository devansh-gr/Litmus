"""A2 — THE MAKE-OR-BREAK TEST.

Question: does a single highlighted SENTENCE produce a *systematic* brain signal,
or is it noise? TRIBE v2 was trained on hour-long naturalistic stimuli; a
decontextualised sentence is off-distribution.

Method:
  1. 20 fear-laden vs 20 neutral sentences (matched in length).
  2. Each -> TRIBE v2 -> (timesteps, 20484 fsaverage5 vertices); mean over time.
  3. z-score each vertex ACROSS sentences (removes per-vertex baseline, which
     otherwise dominates and would fake a signal).
  4. Reduce 20484 vertices -> 75 named Destrieux regions.
  5. Per region: fear vs neutral t-test + Cohen's d.

Two ways this can fail, and we check both:
  - NO DISCRIMINATION: predictions are near-identical regardless of input
    (mean pairwise correlation ~1.0) => the model ignores our content.
  - NO SYSTEMATIC EFFECT: predictions vary, but fear vs neutral doesn't separate
    in any consistent region => sentence-level input is noise.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE / "a2_results.json"

FEAR = [
    "Terrifying new evidence proves the outbreak will devastate every family you love.",
    "The killer is still out there and the police admit they cannot protect you.",
    "Experts warn the collapse will wipe out your savings within a matter of weeks.",
    "A deadly toxin has been found in the water supply of your neighbourhood.",
    "The disease spreads silently and by the time you notice it is far too late.",
    "Violent crime is exploding and no one on your street is safe after dark.",
    "The economy is on the brink of a catastrophe that will ruin an entire generation.",
    "Doctors are terrified by the surge of a virus that resists every treatment.",
    "Your children are being targeted by predators hiding in plain sight online.",
    "The fire is spreading fast and there may be no time left to evacuate.",
    "A devastating earthquake could level the entire city without any warning at all.",
    "Radiation from the plant is leaking and officials are desperately hiding the truth.",
    "Millions will lose everything when the housing market finally implodes this year.",
    "The poison was in the food for months before anybody realised the danger.",
    "Terrorists are planning an attack and the security services have completely failed.",
    "A monstrous storm is barreling toward the coast and will destroy everything.",
    "The infection rate is soaring and hospitals are already turning patients away.",
    "Criminals now know where you live and the authorities refuse to intervene.",
    "This chemical is quietly destroying your brain and nobody is being warned.",
    "The dam is failing and the flood will drown the valley before dawn.",
]

NEUTRAL = [
    "The council approved the new drainage plan for the eastern district on Tuesday.",
    "The library will extend its opening hours during the spring term this year.",
    "She placed the folded map back inside the glove compartment of the car.",
    "The recipe calls for two cups of flour and a teaspoon of salt.",
    "The train arrives at the central station every twenty minutes during the day.",
    "He watered the plants on the balcony before sitting down to read.",
    "The museum has rearranged its collection of pottery in the eastern wing.",
    "Rainfall this month was slightly above the seasonal average for the region.",
    "The committee will publish its annual report at the end of the quarter.",
    "They repainted the fence a pale grey colour over the weekend.",
    "The bakery on the corner opens at six every morning except Sunday.",
    "A new bicycle lane was added along the river path last autumn.",
    "The lecture covered the basic principles of sedimentary rock formation.",
    "She filed the documents alphabetically in the cabinet beside her desk.",
    "The ferry crosses the harbour four times a day during the summer.",
    "He measured the length of the table before ordering the new cloth.",
    "The software update adds support for two additional keyboard layouts.",
    "Local farmers reported an ordinary yield of barley for the season.",
    "The meeting was moved to the smaller conference room on the third floor.",
    "A wooden bench was installed near the entrance of the public garden.",
]

ROIS_OF_INTEREST = {
    "ACC (outrage/conflict)": ["G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant"],
    "Insula (tribal/disgust)": [
        "G_Ins_lg_and_S_cent_ins",
        "G_insular_short",
        "S_circular_insula_ant",
        "S_circular_insula_sup",
    ],
    "dlPFC (reasoning load)": ["G_front_middle"],
    "vmPFC (social/value)": ["G_rectus", "S_suborbital", "G_subcallosal"],
    "Orbitofrontal": ["G_orbital"],
}


def predict_all(model, sentences, tag):
    vecs = []
    for i, sent in enumerate(sentences):
        txt = CACHE / f"a2_{tag}_{i}.txt"
        txt.write_text(sent)
        t0 = time.time()
        events = model.get_events_dataframe(text_path=str(txt))
        preds, _ = model.predict(events=events)
        vecs.append(np.asarray(preds).mean(axis=0))  # mean over timesteps
        print(f"  [{tag} {i+1}/{len(sentences)}] {time.time()-t0:.0f}s", flush=True)
    return np.stack(vecs)


def main() -> None:
    from scipy import stats
    from nilearn import datasets
    from tribev2.demo_utils import TribeModel

    CACHE.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}\n", flush=True)

    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(CACHE), device=device
    )

    t_start = time.time()
    X_fear = predict_all(model, FEAR, "fear")
    X_neut = predict_all(model, NEUTRAL, "neutral")
    print(f"\nall predictions in {(time.time()-t_start)/60:.1f} min", flush=True)

    X = np.concatenate([X_fear, X_neut])            # (40, 20484)
    y = np.array([1] * len(FEAR) + [0] * len(NEUTRAL))

    # --- FAILURE CHECK 1: does the model discriminate inputs at all? ---
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    corr = (Xc @ Xc.T) / np.maximum(norms * norms.T, 1e-9)
    off = corr[~np.eye(len(X), dtype=bool)]
    print(f"\n[check] mean pairwise corr of centred predictions: {off.mean():.3f}")
    print(f"[check] std of raw predictions across sentences: {X.std(axis=0).mean():.5f}")

    # --- z-score each vertex across sentences ---
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)

    # --- reduce vertices -> Destrieux regions ---
    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    annot = np.concatenate(
        [np.asarray(atlas["map_left"]), np.asarray(atlas["map_right"])]
    )
    assert annot.shape[0] == Z.shape[1], (annot.shape, Z.shape)

    results = []
    for idx, name in enumerate(labels):
        if name == "Unknown":
            continue
        mask = annot == idx
        if mask.sum() == 0:
            continue
        roi = Z[:, mask].mean(axis=1)
        t, p = stats.ttest_ind(roi[y == 1], roi[y == 0])
        n1, n0 = (y == 1).sum(), (y == 0).sum()
        pooled = np.sqrt(
            ((n1 - 1) * roi[y == 1].var(ddof=1) + (n0 - 1) * roi[y == 0].var(ddof=1))
            / (n1 + n0 - 2)
        )
        d = (roi[y == 1].mean() - roi[y == 0].mean()) / max(pooled, 1e-9)
        results.append(
            {"region": name, "t": float(t), "p": float(p), "cohens_d": float(d),
             "n_vertices": int(mask.sum())}
        )

    results.sort(key=lambda r: -abs(r["cohens_d"]))

    print("\n=== TOP 10 REGIONS separating FEAR vs NEUTRAL (by |Cohen's d|) ===")
    for r in results[:10]:
        star = "*" if r["p"] < 0.05 else " "
        print(f" {star} {r['region']:<34} d={r['cohens_d']:+.2f}  p={r['p']:.4f}")

    print("\n=== REGIONS WE CARE ABOUT (the taxonomy) ===")
    by_name = {r["region"]: r for r in results}
    for pretty, names in ROIS_OF_INTEREST.items():
        print(f" {pretty}")
        for n in names:
            r = by_name.get(n)
            if r:
                star = "*" if r["p"] < 0.05 else " "
                print(f"   {star} {n:<32} d={r['cohens_d']:+.2f}  p={r['p']:.4f}")

    n_sig = sum(1 for r in results if r["p"] < 0.05)
    print(f"\n[summary] {n_sig}/{len(results)} regions p<0.05 "
          f"(~{0.05*len(results):.1f} expected by chance alone)")

    OUT.write_text(json.dumps(
        {"mean_pairwise_corr": float(off.mean()), "n_significant": int(n_sig),
         "n_regions": len(results), "results": results}, indent=2))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
