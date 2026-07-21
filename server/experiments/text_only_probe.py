"""I1 — does a TEXT-ONLY TRIBE path remove the TTS/acoustic confound?

The normal path is text -> TTS -> audio -> whisperx word timings -> predict, so
the prediction is "the brain LISTENING," dominated by auditory cortex (an artifact
of our TTS step). Here we inject Word events directly (synthetic timings, NO audio
event) so only the language branch runs.

Test: for fear vs neutral, text-only predictions should
  (a) NOT light up auditory cortex (no audio), and
  (b) still separate fear/neutral in fronto-orbital / language regions.
If both hold, text-only is the cleaner path (and kills the last TTS dependency).
"""

import numpy as np
import pandas as pd
import torch

FEAR = [
    "Terrifying new evidence proves the outbreak will devastate every family you love.",
    "The killer is still out there and the police cannot protect you.",
    "A deadly toxin has been found in the water supply of your neighbourhood.",
    "The economy is on the brink of a catastrophe that will ruin everyone.",
    "Your children are being targeted by predators hiding in plain sight online.",
]
NEUTRAL = [
    "The council approved the new drainage plan for the eastern district on Tuesday.",
    "She placed the folded map back inside the glove compartment of the car.",
    "The museum has rearranged its collection of pottery in the eastern wing.",
    "The committee will publish its annual report at the end of the quarter.",
    "A new bicycle lane was added along the river path last autumn.",
]

AUDITORY = ["S_temporal_transverse", "G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo",
            "G_temp_sup-Plan_polar", "G_temp_sup-Lateral"]
SEMANTIC = ["G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_inf-Opercular",
            "S_orbital-H_Shaped", "G_rectus", "G_front_middle"]


def main():
    from tribev2.demo_utils import TribeModel, get_audio_and_text_events
    from nilearn import datasets
    from scipy import stats

    print(">> loading model…", flush=True)
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder="cache",
        device="mps" if torch.backends.mps.is_available() else "cpu")
    print(">> MODEL LOADED", flush=True)

    def text_only_events(sentence: str) -> pd.DataFrame:
        words = sentence.replace(",", "").replace(".", "").split()
        dt = 0.35
        rows = [{"type": "Word", "text": w, "start": i * dt, "duration": dt,
                 "timeline": "default", "subject": "default", "filepath": ""}
                for i, w in enumerate(words)]
        print(">> building events (get_audio_and_text_events)…", flush=True)
        ev = get_audio_and_text_events(pd.DataFrame(rows))
        print(f">> events built: {ev.shape}, types={sorted(ev.type.unique())}", flush=True)
        return ev

    def predict_text_only(sentence: str) -> np.ndarray:
        events = text_only_events(sentence)
        print(">> predicting…", flush=True)
        preds, _ = model.predict(events=events, verbose=False)
        print(">> predicted", flush=True)
        return np.asarray(preds).mean(axis=0)   # (20484,)

    print("=== smoke test: does text-only predict at all? ===", flush=True)
    v0 = predict_text_only(FEAR[0])
    print(f"[ok] text-only prediction shape {v0.shape}\n", flush=True)

    X, y = [], []
    for s in FEAR:
        X.append(predict_text_only(s)); y.append(1)
    for s in NEUTRAL:
        X.append(predict_text_only(s)); y.append(0)
    X = np.stack(X); y = np.array(y)

    # z-score each vertex across THIS set (self-relative, like A2)
    Z = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    annot = np.concatenate([np.asarray(atlas["map_left"]), np.asarray(atlas["map_right"])])
    idx = {n: i for i, n in enumerate(labels)}

    def roi_mean_abs_z(names):  # how strongly a region is engaged at all (|z| over all sentences)
        vals = [np.abs(Z[:, annot == idx[n]]).mean() for n in names if n in idx and (annot == idx[n]).any()]
        return float(np.mean(vals)) if vals else float("nan")

    def roi_fear_vs_neutral_d(names):
        ds = []
        for n in names:
            if n in idx and (annot == idx[n]).any():
                roi = Z[:, annot == idx[n]].mean(1)
                a, b = roi[y == 1], roi[y == 0]
                p = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                ds.append((a.mean() - b.mean()) / max(p, 1e-9))
        return float(np.mean(ds)) if ds else float("nan")

    print("=== is auditory cortex quiet in TEXT-ONLY? (should be, no audio) ===")
    print(f"  auditory  mean|z| = {roi_mean_abs_z(AUDITORY):.3f}")
    print(f"  semantic  mean|z| = {roi_mean_abs_z(SEMANTIC):.3f}")
    print("=== does semantic cortex still separate fear vs neutral? ===")
    print(f"  semantic  Cohen's d (fear-neutral) = {roi_fear_vs_neutral_d(SEMANTIC):+.2f}")
    print(f"  auditory  Cohen's d (fear-neutral) = {roi_fear_vs_neutral_d(AUDITORY):+.2f}")
    print("TEXT_ONLY_DONE")


if __name__ == "__main__":
    main()
