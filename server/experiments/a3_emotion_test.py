"""A3 — Do DIFFERENT emotions hit DIFFERENT regions, or does it all just hit language cortex?

A2 showed fear vs neutral separates strongly (34/75 regions), but the separating
regions looked like speech/language areas — consistent with a linguistic confound
rather than affect. This test disentangles the two.

Conditions: fear | outrage | reward (dopamine bait) | neutral   (15 sentences each)

The decisive analyses:
  1. PROFILE SIMILARITY. Compute each emotion's effect map vs neutral (Cohen's d
     across 75 regions), then correlate those maps with each other.
       - r ~ +1  => every emotion moves the SAME regions => one generic
         "different-from-neutral" axis => LINGUISTIC CONFOUND, brain-region story dies.
       - r ~ 0   => emotions move DISTINCT regions => genuine affective
         differentiation => the product thesis survives.
  2. CROSS-VALIDATED DECODING. Can we classify emotion from the ROI pattern on
     held-out sentences? Accuracy vs 25% chance. This is the "trained readout"
     that replaces the hand-made taxonomy.
  3. Which regions actually drive each emotion (data-driven, not asserted).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
CACHE = HERE / "cache"
OUT = HERE / "a3_results.json"

CONDITIONS = {
    "fear": [
        "Terrifying new evidence proves the outbreak will devastate every family you love.",
        "The killer is still out there and the police admit they cannot protect you.",
        "A deadly toxin has been found in the water supply of your neighbourhood.",
        "The disease spreads silently and by the time you notice it is far too late.",
        "The economy is on the brink of a catastrophe that will ruin an entire generation.",
        "Your children are being targeted by predators hiding in plain sight online.",
        "The fire is spreading fast and there may be no time left to evacuate.",
        "A devastating earthquake could level the entire city without any warning at all.",
        "Millions will lose everything when the housing market finally implodes this year.",
        "A monstrous storm is barreling toward the coast and will destroy everything.",
        "The infection rate is soaring and hospitals are already turning patients away.",
        "This chemical is quietly destroying your brain and nobody is being warned.",
        "The dam is failing and the flood will drown the valley before dawn.",
        "Radiation from the plant is leaking and officials are hiding the truth.",
        "Violent crime is exploding and no one on your street is safe after dark.",
    ],
    "outrage": [
        "These corrupt officials stole your money and laughed about it in private.",
        "It is disgusting that they betrayed every promise they ever made to you.",
        "How dare they lecture us while breaking the very rules they wrote.",
        "The scandal proves they have contempt for ordinary working people like you.",
        "They shredded the documents to bury the truth and nobody was punished.",
        "It is outrageous that the guilty walked free while the victims got nothing.",
        "They took your taxes and handed them straight to their wealthy donors.",
        "The hypocrisy is sickening and they expect you to simply accept it.",
        "They lied under oath and the establishment closed ranks to protect them.",
        "Shameful behaviour like this should end a career, yet he was promoted.",
        "They mocked the families of the victims during a private fundraising dinner.",
        "The betrayal was deliberate and they have never once apologised for it.",
        "It is unacceptable that they raised their own pay while cutting your services.",
        "They rigged the process and then had the nerve to call it fair.",
        "The cover-up was brazen and every single one of them was complicit.",
    ],
    "reward": [
        "You just unlocked an exclusive prize that almost nobody ever gets to claim.",
        "Claim your free upgrade instantly and enjoy unlimited access starting today.",
        "This incredible jackpot could be yours with a single lucky click right now.",
        "An amazing limited drop is live and the rewards are absolutely enormous.",
        "You have been selected to receive a spectacular bonus worth thousands.",
        "Spin once and you could instantly win the grand prize of the year.",
        "Get an insane discount today and keep the savings for yourself forever.",
        "Your reward is waiting and it only takes one tap to collect it.",
        "Unlock the secret bonus level and collect treasure beyond your wildest dreams.",
        "A fantastic gift is reserved in your name and ready to be claimed.",
        "Double your money instantly with this one incredible opportunity today.",
        "You won the exclusive draw and the fabulous prize ships out tomorrow.",
        "Collect your daily bonus now and watch your rewards multiply every hour.",
        "This thrilling offer gives you everything you ever wanted for absolutely free.",
        "Tap once to claim a jackpot that could change your entire life.",
    ],
    "neutral": [
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
    ],
}


def main() -> None:
    from nilearn import datasets
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from tribev2.demo_utils import TribeModel

    CACHE.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}\n", flush=True)
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=str(CACHE), device=device
    )

    X, y = [], []
    t_start = time.time()
    for cond, sents in CONDITIONS.items():
        for i, sent in enumerate(sents):
            txt = CACHE / f"a3_{cond}_{i}.txt"
            txt.write_text(sent)
            t0 = time.time()
            events = model.get_events_dataframe(text_path=str(txt))
            preds, _ = model.predict(events=events)
            X.append(np.asarray(preds).mean(axis=0))
            y.append(cond)
            print(f"  [{cond} {i+1}/{len(sents)}] {time.time()-t0:.0f}s", flush=True)
    print(f"\nall predictions in {(time.time()-t_start)/60:.1f} min\n", flush=True)

    X = np.stack(X)
    y = np.array(y)

    # z-score each vertex across sentences, then reduce to Destrieux regions
    Z = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)
    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    annot = np.concatenate(
        [np.asarray(atlas["map_left"]), np.asarray(atlas["map_right"])]
    )
    names, cols = [], []
    for idx, nm in enumerate(labels):
        if nm == "Unknown":
            continue
        m = annot == idx
        if m.sum():
            names.append(nm)
            cols.append(Z[:, m].mean(axis=1))
    R = np.stack(cols, axis=1)  # (n_sentences, n_regions)
    print(f"ROI matrix: {R.shape}\n", flush=True)

    emos = ["fear", "outrage", "reward"]

    # --- 1. effect map (Cohen's d vs neutral) per emotion ---
    neu = R[y == "neutral"]
    dmaps = {}
    for e in emos:
        cur = R[y == e]
        pooled = np.sqrt((cur.var(axis=0, ddof=1) + neu.var(axis=0, ddof=1)) / 2)
        dmaps[e] = (cur.mean(axis=0) - neu.mean(axis=0)) / np.maximum(pooled, 1e-9)

    print("=== TOP 5 REGIONS PER EMOTION (vs neutral, by |d|) ===")
    for e in emos:
        order = np.argsort(-np.abs(dmaps[e]))[:5]
        print(f" {e}:")
        for i in order:
            t, p = stats.ttest_ind(R[y == e][:, i], neu[:, i])
            print(f"    {names[i]:<32} d={dmaps[e][i]:+.2f}  p={p:.4f}")

    # --- 2. THE DECISIVE TEST: are the effect maps the same? ---
    print("\n=== PROFILE SIMILARITY (correlation between emotion effect-maps) ===")
    print("    r~+1 => all emotions move the SAME regions => LINGUISTIC CONFOUND")
    print("    r~0  => emotions move DISTINCT regions => real affective signal\n")
    sims = {}
    for a in range(len(emos)):
        for b in range(a + 1, len(emos)):
            r = float(np.corrcoef(dmaps[emos[a]], dmaps[emos[b]])[0, 1])
            sims[f"{emos[a]}_vs_{emos[b]}"] = r
            print(f"    r({emos[a]:<8}, {emos[b]:<8}) = {r:+.3f}")

    # --- 3. cross-validated decoding (the "trained readout") ---
    print("\n=== CROSS-VALIDATED DECODING (the learned readout) ===")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc4 = cross_val_score(clf, R, y, cv=cv, scoring="accuracy")
    print(f"    4-way (fear/outrage/reward/neutral): {acc4.mean()*100:.1f}%  (chance 25%)")

    pair_scores = {}
    for e in emos:
        mask = np.isin(y, [e, "neutral"])
        a = cross_val_score(clf, R[mask], y[mask], cv=cv, scoring="accuracy")
        pair_scores[f"{e}_vs_neutral"] = float(a.mean())
        print(f"    {e} vs neutral: {a.mean()*100:.1f}%  (chance 50%)")
    # emotion-vs-emotion: can it tell fear from reward? (the hard one)
    for a_i in range(len(emos)):
        for b_i in range(a_i + 1, len(emos)):
            e1, e2 = emos[a_i], emos[b_i]
            mask = np.isin(y, [e1, e2])
            a = cross_val_score(clf, R[mask], y[mask], cv=cv, scoring="accuracy")
            pair_scores[f"{e1}_vs_{e2}"] = float(a.mean())
            print(f"    {e1} vs {e2}: {a.mean()*100:.1f}%  (chance 50%)")

    OUT.write_text(json.dumps({
        "profile_similarity": sims,
        "decoding_4way_acc": float(acc4.mean()),
        "decoding_pairs": pair_scores,
        "regions": names,
        "effect_maps": {e: dmaps[e].tolist() for e in emos},
    }, indent=2))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
