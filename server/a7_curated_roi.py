"""A7 — is there an HONEST per-sentence brain visual, or does the brain layer get cut?

Finding so far: ranking ALL regions of a single sentence's z-map (vs neutral
baseline) surfaces sensorimotor/auditory cortex -- low-level TTS acoustics, not
content. But A2 showed a CURATED set of semantic/affective ROIs (IFG, orbital,
vmPFC) DOES separate emotional from neutral at the group level (d=1.1-1.6).

Question: for a single sentence, does the mean z over those CURATED semantic ROIs
still separate emotional (fear+reward) from neutral? If yes, /brainmap can honestly
report "value/language-evaluation cortex engagement" instead of acoustic noise. If
no, the brain layer is not a per-sentence signal and gets cut from product claims.

Uses cached predictions (fast). Loads its own model; run with the server stopped.
"""

import numpy as np
import torch

from a3_emotion_test import CONDITIONS
from pathlib import Path

HERE = Path(__file__).parent
CACHE = HERE / "cache"

# Curated interpretable ROIs (value + language-evaluation), from A2's significant
# fear-vs-neutral regions. NOT sensorimotor, NOT auditory.
SEMANTIC_ROIS = [
    "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_inf-Opercular",
    "S_orbital-H_Shaped", "S_orbital_med-olfact", "S_orbital_lateral",
    "G_rectus", "G_subcallosal", "G_front_middle", "G_front_sup",
    "G_orbital",
]
SENSORIMOTOR = [
    "G_precentral", "G_postcentral", "S_central", "G_and_S_subcentral",
    "G_and_S_paracentral",
]


def main() -> None:
    from scipy import stats
    from nilearn import datasets
    from tribev2.demo_utils import TribeModel

    b = np.load(HERE / "baseline.npz")
    mean, std = b["mean"], b["std"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(CACHE), device=device)

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    annot = np.concatenate([np.asarray(atlas["map_left"]), np.asarray(atlas["map_right"])])
    idx = {name: i for i, name in enumerate(labels)}

    def roi_z(zvert, names):
        vals = []
        for n in names:
            if n in idx:
                m = annot == idx[n]
                if m.sum():
                    vals.append(zvert[m].mean())
        return float(np.mean(vals)) if vals else np.nan

    # 10 each (cached), held-out neutral not in the 30-sentence baseline overlap-safe
    fear = CONDITIONS["fear"][:10]
    reward = CONDITIONS["reward"][:10]
    neutral = CONDITIONS["neutral"][:10]

    def zmap(sent, tag, i):
        txt = CACHE / f"a7_{tag}_{i}.txt"
        txt.write_text(sent)
        ev = model.get_events_dataframe(text_path=str(txt))
        preds, _ = model.predict(events=ev, verbose=False)
        vmean = np.asarray(preds).mean(axis=0)
        return (vmean - mean) / std

    rows = {"fear": [], "reward": [], "neutral": []}
    for tag, sents in [("fear", fear), ("reward", reward), ("neutral", neutral)]:
        for i, s in enumerate(sents):
            z = zmap(s, tag, i)
            rows[tag].append((roi_z(z, SEMANTIC_ROIS), roi_z(z, SENSORIMOTOR)))
            print(f"  {tag} {i+1}/10", flush=True)

    sem = {k: np.array([r[0] for r in v]) for k, v in rows.items()}
    mot = {k: np.array([r[1] for r in v]) for k, v in rows.items()}
    emo_sem = np.concatenate([sem["fear"], sem["reward"]])
    emo_mot = np.concatenate([mot["fear"], mot["reward"]])

    def cohen(a, b):
        p = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        return (a.mean()-b.mean())/max(p, 1e-9)

    print("\n=== SEMANTIC ROIs (IFG/orbital/vmPFC) — emotional vs neutral ===")
    t, p = stats.ttest_ind(emo_sem, sem["neutral"])
    print(f"  emotional mean z={emo_sem.mean():+.3f}   neutral mean z={sem['neutral'].mean():+.3f}")
    print(f"  Cohen's d={cohen(emo_sem, sem['neutral']):+.2f}   p={p:.4f}")

    print("\n=== SENSORIMOTOR (the acoustic confound) — emotional vs neutral ===")
    t2, p2 = stats.ttest_ind(emo_mot, mot["neutral"])
    print(f"  emotional mean z={emo_mot.mean():+.3f}   neutral mean z={mot['neutral'].mean():+.3f}")
    print(f"  Cohen's d={cohen(emo_mot, mot['neutral']):+.2f}   p={p2:.4f}")

    print("\nVERDICT:")
    d_sem = cohen(emo_sem, sem["neutral"])
    if p < 0.05 and abs(d_sem) > 0.5:
        print("  SEMANTIC ROIs separate emotional from neutral per-sentence.")
        print("  => /brainmap can honestly report curated value/language-cortex engagement.")
    else:
        print("  Semantic ROIs do NOT separate per-sentence.")
        print("  => brain map is not a per-sentence signal; cut it from product claims.")
    print("A7_DONE")


if __name__ == "__main__":
    main()
