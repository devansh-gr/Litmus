# Eval data — dev vs. locked test

Data hygiene matters: if you tune prompts on the same examples you report on, the number
measures memorization, not generalization (Cawley & Talbot 2010). So the sets have roles.

## DEV (self-authored — used to design/tune prompts; NOT a generalization estimate)
- `eval_set.jsonl` (72) — 6 hand-written examples per vector. Prompt-design + regression.
- `realistic_set.jsonl` (50) — 60% benign mix; the gate prompt was **tuned against this**.
- `benign_set.jsonl` (16) — greetings/helpful text; the reported-bug regression guard.

These are fine as smoke-tests and CI guards. They **overstate** real accuracy because the
author wrote the examples AND tuned the model to them (circular).

## TEST (external — independently labeled, never tuned on → the honest number)
- `external_test.jsonl` (250) — built by `build_external_set.py` from two public datasets:
  - **Yamana ec-darkpattern** (e-commerce copy): Urgency→false-urgency, Scarcity→fomo,
    Social Proof→social-proof-conformity, Not Dark Pattern→none.
  - **LOGIC logical-fallacy** (Jin et al. 2022): ad populum→social-proof-conformity,
    fallacy of credibility→authority-appeal.

Report the **external** number as the real one; use the dev sets only for iteration.

### External-set caveats (documented, not hidden)
- **Cross-taxonomy:** other people's label definitions ≈ but ≠ ours (e.g. dark-pattern
  "Scarcity" ≈ our fomo, but overlaps false-urgency).
- **Domain shift:** e-commerce / debate text, not arbitrary highlighted text.
- **Coverage:** only false-urgency, fomo, social-proof-conformity, authority-appeal, and
  none have a clean external label. fear / outrage / tribal / critical-thinking /
  dopamine-bait / hype / **manufactured-awe have NO external dataset** and remain
  self-authored only (manufactured-awe is genuinely novel — see BENCHMARKING.md).
- **Contamination:** not formally checked (Oren et al. 2023 exchangeability test is a
  follow-up); these are small public sets so some overlap with pretraining is possible.

Run: `python tests/bench_full.py --data tests/data/external_test.jsonl --name external`
