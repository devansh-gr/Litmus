# Benchmarking the detector — methodology research & roadmap

Research synthesis (2026-08-01) on how to *properly* benchmark the two-stage persuasion
detector (binary **gate** → 11-class **technique** classifier, on-device Llama-3.2-3B scored by
label log-probabilities). This is a findings + roadmap document for discussion — **not** a set of
changes yet.

---

## 0. TL;DR — the five things that matter most

1. **Our test data is self-authored and we tuned prompts on it.** That's the biggest methodological
   hole: the numbers measure "did the author's examples match the author's prompt," not
   generalization. Fix = borrow **real, externally-labeled** data.
2. **The gate bug we hit (calling everything "manipulation") was a textbook LLM failure** —
   *surface-form competition / common-token priming*. Our yes/no fix was luck; the principled fix is
   **contextual calibration** + single-token verbalizers.
3. **We never measure the "84%."** A confidence number the UI shows must be **calibrated** (when it
   says 84%, it should be right ~84% of the time) — measured with ECE / Brier / reliability diagrams.
4. **N≈50 means ±14-point error bars.** Every headline number needs a **confidence interval**, and
   test sets must be sized by *minority-class count*, not total.
5. **A two-stage pipeline needs 3–4 numbers, not one** (gate-isolated, classifier-on-gold-passes,
   end-to-end, oracle-gate upper bound) or you can't tell which stage is failing.

---

## Status (2026-08-05) — Tier 0 + 1 done, Tier 2 partial, interpersonal family, gate recall recovered

- ✅ **Tier 0 (metrics/hygiene):** `bench_full.py` reports macro-F1, per-class, confusion, MCC,
  Wilson/bootstrap CIs, gate PR-AUC, and ECE calibration. Dev/holdout split of the external set.
- ✅ **Gate recall recovery (dark patterns):** the 08-04 run showed the gate missing ~44% of real
  manipulation. `tests/gate_diag.py` localized the miss to e-commerce dark patterns that read like
  plain facts — social-proof activity nudges (22/31 missed), scarcity (10/20), countdown timers,
  clickbait. Re-swept the threshold (0.60→0.70) and taught the gate each dark-pattern family, each
  guarded so its benign look-alike (neutral stats, clocks, news) stays `none`. Result on the same
  300-ex external set: **gate recall 0.56 → 0.77, precision 0.84 → 0.87, PR-AUC 0.89 → 0.93**,
  accuracy 38% → 44% — **+21 recall points with no precision cost** (benign FP 23→25). Locked in
  by `tests/test_dark_patterns.py`.
- ✅ **Interpersonal family (taxonomy expansion):** added guilt-tripping, love-bombing, and
  blame-shifting/DARVO, and folded gaslighting + minimization into critical-thinking-suppression,
  because the marketing-only taxonomy misfiled one-on-one manipulation. Behavioral regression set
  `interpersonal_test.jsonl` (24 curated ex — self-authored, so a REGRESSION set, NOT an accuracy
  claim; see `interpersonal_report.py`): **21/24 (88%)**, love-bombing 5/5, benign controls 5/5,
  with 3 cluster-boundary confusions (guilt↔blame↔love-bomb) logged. No external labeled analog for
  interpersonal tactics yet — a Tier-1 gap to source.
- ✅ **Tier 1 (real data):** `external_test.jsonl` — 300 independently-labeled examples from
  dark-patterns + LOGIC + clickbait (6 vectors + none), never authored or tuned on. **Honest number
  (2026-08-05, after the dark-pattern gate fixes): 44% accuracy [38.2–49.3], macro-F1 0.18, MCC
  0.33, gate PR-AUC 0.93 / recall 0.77 / precision 0.87** vs the ~80% self-authored mirage.
  Calibration is OFF by product choice (assertive %), so displayed **ECE is 0.26**; turning it on
  (`CPD_CALIB_A=0.56 CPD_CALIB_B=-1.10`) restores ECE ≈ 0.10 at the cost of lower shown %. The gate
  now catches manipulation well; the remaining low per-class *technique* accuracy (dopamine 0.00,
  authority 0.15) is the **cross-taxonomy mapping difficulty** — clickbait↔dopamine and
  anecdotal-authority are loose analogs — not the gate failing to flag them.
- ✅ **Tier 2 (partial):** gate un-sharpened + threshold re-swept on external data;
  **Platt-calibrated confidence** (ECE 0.37→0.10); **abstain** on low confidence; fomo/false-urgency
  **technique blur fixed** (fomo recall 0.05→0.70). **Contextual calibration evaluated → no gain**
  (the yes/no gate already avoids the priming it targets) — kept off.
- ⏳ **Remaining (gate recall):** two known misses were left on the table on purpose. (1)
  **Anecdotal/weak authority** ("my friend tweeted about the health benefits, so…") — 6/19 missed;
  flagging it risks catching ordinary "my friend said…" speech, so it needs a careful intent guard,
  not another blanket clause. (2) **Bare aggregate counts** with no live-activity cue ("111 people
  have purchased this item") sit in the same manip_prob band as ordinary praise — catching them
  would re-flag "you did a wonderful job", so they stay borderline. Both are recall, not ranking:
  the gate PR-AUC (0.93) already separates them; only the operating threshold declines to commit.
- ⏳ **Remaining (coverage / rigor):** external coverage for fear / outrage / tribal /
  critical-thinking / hype / manufactured-awe and the whole interpersonal family (no clean dataset —
  needs sourcing or self-authored + annotators); the dopamine clickbait mapping is loose; CheckList
  robustness suite; a holdout-only re-run to de-bias the in-sample half of the 0.56→0.77 number;
  Tier 3 (multi-annotator ground truth / α). Deep-scan latency re-measure is still **reboot-blocked**
  (swap maxed during this sprint).

---

## 1. Honest assessment of what we have today

| Aspect | Current state | Problem |
|---|---|---|
| Data | `eval_set` (72), `realistic_set` (50), `benign_set` (16) — **all written by me** | Self-authored → author bias, prototypical/clean, unrepresentative distribution |
| Tuning | Gate prompt tuned **against `realistic_set`**, then reported on it | **Test-set leakage** — the score measures memorization, not generalization |
| Metric | Overall accuracy + per-vector recall/precision | Accuracy is misleading under imbalance; no macro-F1, no confusion matrix, no MCC |
| Gate metric | Pass/fail counts | No PR-AUC, no threshold chosen on a validation set, no cost model |
| Confidence | Mean-conf-when-correct vs wrong (83% vs 66%) | **Not calibration** — no ECE, Brier, or reliability diagram |
| Uncertainty | None | No confidence intervals; N is tiny so numbers are very noisy |
| Ground truth | One labeler (me) | No inter-annotator agreement; can't separate "correct" from "what I intended" |
| Scoring | Length-normalized label log-prob, `argmax` | Uncalibrated; vulnerable to surface-form / priming / length bias |

**Net:** what we have is fine as smoke-tests and regression guards. It is **not** a valid benchmark
of real-world accuracy. Everything below is how to make it one.

---

## 2. The data problem (highest leverage)

Self-authored examples avoid pretraining *contamination* but introduce **author bias**: narrow
vocabulary, prototypical phrasing, clean/balanced distribution unlike real traffic, and no
independent ground truth. Tuning prompts on the same set you report on is **leakage** — effectively
reporting on training data (Kaufman 2012; Cawley & Talbot 2010).

**Escape route — borrow real, externally-labeled data.** The research maps our marketing/
propaganda vectors to established corpora (the interpersonal family below has no such analog):

| Vector | External labeled analog | Status |
|---|---|---|
| fear-mongering | Appeal to Fear/Prejudice — SemEval-2023 T3, SemEval-2020 | **DIRECT** |
| authority-appeal | Appeal to Authority — SemEval-2023/2020, LOGIC, Argotario | **DIRECT** (best covered) |
| social-proof-conformity | Appeal to Popularity / Bandwagon — SemEval + Ad Populum | **DIRECT** |
| critical-thinking-suppression | Conversation Killer / Thought-terminating Cliché — SemEval | **DIRECT** |
| tribal-in-group-bias | Flag-Waving — SemEval | direct-ish (closest analog) |
| false-urgency | Appeal to Time — SemEval-2023; + dark-patterns *urgency* | **DIRECT** + marketing |
| dopamine-bait | Clickbait — Webis-Clickbait-17 (38,517 headlines) | marketing only |
| fomo | dark-patterns *scarcity/urgency* — Mathur 2019 / Yamana 2022 | marketing only |
| outrage | Loaded Language + Name Calling (composite) | partial |
| hype-hope-mongering | Exaggeration/Minimisation + positive Loaded Language | partial |
| **manufactured-awe** | **none anywhere** | **self-authored only** |
| **guilt-tripping / love-bombing / blame-shifting** | **none in surveyed corpora** | **self-authored only** (interpersonal family — Tier-1 sourcing gap) |

- **`SemEval-2023 Task 3`** is the flagship: 2,049 docs, 9 languages, span + paragraph labels, a
  23-technique taxonomy that covers **6 of our vectors** directly, official **micro-F1** protocol,
  research license. (https://aclanthology.org/2023.semeval-1.317/)
- **Webis-Clickbait-17** and **Mathur/Yamana dark-patterns** cover the *marketing* vectors
  (dopamine-bait, fomo, urgency) that propaganda corpora don't.
- **Reality check:** academic taxonomies are overwhelmingly *negative-valence* (news/political
  disinformation). Our *positive-valence* vectors (hype, awe, dopamine-bait) are under-represented —
  **manufactured-awe has no dataset in any surveyed corpus**, so it will always lean on
  self-authored data. Worth stating this limitation openly.
- **Contamination check** before trusting any public set: Oren et al. 2023 "Proving Test-Set
  Contamination in Black-Box LMs" — an exchangeability test we can run against Llama-3.2-3B without
  needing its training corpus. (https://arxiv.org/abs/2310.17623)

---

## 3. The right metrics (for an imbalanced, subjective, multi-class + gate system)

**Never headline plain accuracy** — under a mostly-benign stream, "always benign" scores high while
catching nothing (balanced accuracy would drop to 1/n). Report instead:

- **11-class:** **macro-F1** (headline, every technique counts equally) + **per-class P/R/F1**
  (`average=None`, so rare vectors are visible) + a **normalized confusion matrix** + **MCC** /
  **balanced accuracy**. (Grandini 2020; Chicco & Jurman 2020)
- **Gate (binary):** **PR-AUC / average precision** as the primary summary (ROC-AUC hides the
  false-positive burden when positives are rare), with ROC-AUC secondary. Pick the threshold on a
  **validation** split by target precision/recall or **expected cost** (Elkan 2001), then lock it.
  (Saito & Rehmsmeier 2015)
- **Report precision at the real base rate.** By Bayes, at 5% prevalence a gate with 90% recall +
  90% specificity still yields only **~32% precision**. Our balanced test sets make precision look
  better than production — flag this explicitly.

---

## 4. Calibration — making the "84%" mean something

A user-facing confidence % is a *promise*: 84% should be right ~84% of the time. That's separate
from accuracy, and LLM log-prob "confidences" are usually **overconfident**.

- **Measure:** reliability diagram (bin confidence vs actual accuracy) + **ECE** (expected
  calibration error) + **Brier score** (a proper scoring rule).
- **Fix:** **temperature scaling** — one scalar T dividing the logits, fit on a **validation** set;
  monotonic, so it changes the confidence numbers but **not** accuracy/ranking (Guo et al. 2017).
  Maps directly onto our `CONFIDENCE_TEMP` — but our 0.25 was tuned for *assertiveness*, not
  calibration. We should measure ECE and set T to minimize it, then decide how much to sharpen.
- Calibrate the **gate** and the **11-class** stages **separately** (different prompts, different
  bias profiles).

---

## 5. LLM label-scoring pitfalls (explains our gate bug directly)

Raw `argmax log P(label | prompt)` is **not** a sound scorer. Four biases:

1. **Surface-form competition** (Holtzman 2021) — probability mass splits across synonymous strings;
   the right concept can lose to a competitor surface form.
2. **Prompt-token / common-token priming** (Zhao 2021) — a word repeated in the prompt gets its
   label probability inflated by copying/frequency, not evidence. **This is exactly the bug we hit:**
   the gate prompt said "manipulation" repeatedly → the label "manipulation" won for *everything*
   (0/16 benign). Switching to "yes/no" accidentally dodged it.
3. **Length/tokenization bias** — summed log-probs favor shorter labels; our per-token normalization
   helps but isn't tokenization-agnostic (byte-length is better).
4. **The principled fixes:** single-token verbalizers where possible; **contextual calibration**
   ("Calibrate Before Use", Zhao 2021 — feed a content-free input like "N/A", read the model's bias,
   subtract it out; up to +30% absolute) and/or **domain-conditional PMI** (Holtzman 2021). Apply to
   **both** stages independently.

This is arguably the highest-value *algorithmic* upgrade: it would make the gate robust by design
instead of by lucky wording.

---

## 6. Evaluating the two-stage pipeline

Measure **all** of these or you can't localize failures (errors compound: 85% × 85% ≈ 72%):

1. **Gate in isolation** — P/R/PR-AUC on gold gate labels.
2. **Classifier on gold-passes** — 11-class macro-F1 on only the items that *truly* are manipulation
   (so upstream errors don't contaminate it).
3. **End-to-end cascade** — real input → predicted gate → classifier → final label. The deployment
   number.
4. **Oracle-gate upper bound** — replace the gate with gold; the gap vs end-to-end = the exact cost
   of gate errors. Cleanly attributes the error budget between the two stages.

---

## 7. Robustness / behavioral testing (beyond accuracy)

Adopt **CheckList** (Ribeiro 2020): a matrix of capabilities × test types, run separately for gate
and classifier:

- **MFT** (minimum functionality) — targeted unit tests per behavior (e.g. negation: "this is *not*
  urgent at all").
- **INV** (invariance) — label-preserving perturbations must not change the verdict: paraphrase,
  typos, swapping neutral names, adding neutral clauses.
- **DIR** (directional) — a known-direction edit moves the prediction the expected way (adding a
  clear fear clause should raise, not lower, manipulation).
- Perturbation families to cover: **negation** (models are notoriously brittle), sarcasm,
  mixed-signal ("great service, terrible food"), typos, length extremes. Tools: TextAttack.

Our `test_robustness.py` (long/emoji/unicode) is a start but only tests "doesn't crash," not
behavioral correctness.

---

## 8. Statistical rigor

- **Confidence intervals on every headline number** — Wilson score interval (small-N safe) or
  bootstrap (for F1/AUC/ECE with no closed form). At N=50, accuracy ±~8pts near 0.9, **±14pts near
  0.5**. A 10× bigger set only halves the interval.
- **Size the test set by minority-class count** — recall CIs depend on the number of *positive*
  examples, not total N.
- **Compare versions with McNemar's test** on the same items (paired), not two independent
  accuracies. Require an improvement to be significant / exceed the CI, not just numerically higher.
- **Stop p-hacking across prompt iterations** — we've now iterated the gate prompt ~6 times against
  the same set. Tune on **dev**, confirm once on a **locked test set** the author never inspects.
- **Report prompt-sensitivity spread** — meaning-preserving format changes can move LLM accuracy by
  tens of points (Sclar 2024); report min/max/std over a few prompt variants, not one lucky prompt.

---

## 9. Ground truth for subjective labels

Persuasion labels are subjective, so "gold" must be *constructed*: ≥3 annotators, a written codebook
with per-vector definitions + examples, a pilot round, then adjudication. Report **Krippendorff's α**
(handles multi-label, missing data) as the **reliability ceiling** — a classifier can't beat human
agreement. Expect modest α: **SemEval-2023 T3 reached only α ≈ 0.34** on fine-grained persuasion;
SemEval-2021 averaged ≈ 0.77. Translation: **this task is genuinely hard to agree on**, so some of
our "errors" are really label ambiguity — which caps achievable accuracy and must be reported.

---

## 10. Proposed roadmap (tiered by effort — for discussion)

**Tier 0 — hygiene (hours, high value):**
- Split data into **dev** (tune) vs **locked test** (report once). Stop reporting on the set we tuned on.
- Add **Wilson/bootstrap CIs** to `run_eval.py`; report N alongside every number.
- Add **macro-F1, per-class F1, confusion matrix, MCC**; treat gate as binary → **PR-AUC**.
- Add **ECE + reliability diagram + Brier** for the confidence %.

**Tier 1 — real data (days, highest value):**
- Pull **SemEval-2023 T3** (+ clickbait / dark-patterns) → build a real, externally-labeled eval set
  covering ~6–8 vectors. Run our contamination check first.
- Keep a small self-authored set only for **manufactured-awe** (no external data exists) — labeled clearly.

**Tier 2 — robustness & scoring (days):**
- **Contextual calibration** on both stages (fixes the surface-form/priming class of bugs by design).
- A **CheckList** suite (MFT/INV/DIR) for gate + classifier.
- Report the **4 two-stage numbers** (gate-isolated / classifier-on-gold / end-to-end / oracle-gate).

**Tier 3 — ground truth (weeks, if we want publishable rigor):**
- Recruit ≥3 annotators, codebook, adjudication, report **Krippendorff's α** as the ceiling.
- Temperature-scale for calibration; report prompt-sensitivity spread.

---

## Key decisions to make together
1. **How much rigor do we actually want?** Tier 0–1 (a credible internal benchmark) vs Tier 3
   (publishable/defensible). Depends on whether this is a demo or a claim we'll defend.
2. **Do we adopt external SemEval/clickbait data**, accepting a taxonomy-mapping step + domain shift
   (news/e-commerce vs whatever users highlight)?
3. **Is the confidence % a real promise we calibrate**, or a deliberately-sharpened UX signal we
   stop calling a probability?
4. **Do we invest in contextual calibration** (fixes the gate's fragility structurally) now, or keep
   the working-but-lucky yes/no gate?
5. **manufactured-awe has no external ground truth** — keep it as a vector (self-authored only), or
   reconsider it?

## Sources
scikit-learn model evaluation; Grandini 2020 (arxiv 2008.05756); Chicco & Jurman 2020; Saito &
Rehmsmeier 2015; Elkan 2001; Guo 2017 (arxiv 1706.04599); Brown/Cai/DasGupta 2001; Dietterich 1998
(McNemar); Kaufman 2012 / Cawley & Talbot 2010 (leakage); Artstein & Poesio 2008 (agreement);
Holtzman 2021 (arxiv 2104.08315, surface form); Zhao 2021 (arxiv 2102.09690, Calibrate Before Use);
EleutherAI MC-normalization; Oren 2023 (arxiv 2310.17623, contamination); Ribeiro 2020 (CheckList);
Sclar 2024 (arxiv 2310.11324, prompt sensitivity); SemEval-2023 T3 (aclanthology 2023.semeval-1.317);
SemEval-2020 T11 (2020.semeval-1.186); Da San Martino 2019 (D19-1565); LOGIC (2022.findings-emnlp.532);
Webis-Clickbait-17; Mathur 2019 / Yamana 2022 (dark patterns).
