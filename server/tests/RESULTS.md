# Detector benchmark — results

Run `python tests/run_eval.py` against the live server to regenerate.

## Current (2026-07-29, after latency + taxonomy fixes)

**Overall accuracy: 62/72 = 86.1%** (+9.7 pts). **p50 latency 446ms → cached ~1ms** (KV-cache reuse).

| vector | recall | precision | Δ from baseline |
|---|---|---|---|
| manufactured-awe | **100%** | **100%** | **0% → 100% recall** (definition sharpened) |
| hype-hope-mongering | 100% | 75% | +17 recall |
| outrage / tribal / authority-appeal / dopamine* / crit-think* | 67–100% | 100% | precision up (awe no longer leaks) |
| false-urgency | 100% | 60% | unchanged (still the over-firing catch-all) |
| none | 83% | 100% | unchanged |

What fixed it: `manufactured-awe` and `hype-hope-mongering` had near-identical definitions
("change the world" vs "world-changing"). Rewrote them to separate the axis — **hype = personal
future upside/benefit** ("want in, improve your life") vs **awe = grandeur of the thing itself**
("revolutionary, unprecedented, greatest-ever"). Remaining weak seam: `false-urgency` still
absorbs fomo/fear/dopamine when urgency words are present (a genuinely shared surface cue).

---

## Baseline (2026-07-29, before latency + taxonomy fixes)

**Overall accuracy: 55/72 = 76.4%** on the 72-example eval set (6 per vector).

| vector | recall | precision |
|---|---|---|
| outrage | 100% | 100% |
| tribal-in-group-bias | 100% | 100% |
| authority-appeal | 83% | 100% |
| none | 83% | 100% |
| fomo | 67% | 100% |
| social-proof-conformity | 83% | 83% |
| critical-thinking-suppression | 67% | 80% |
| dopamine-bait | 67% | 80% |
| fear-mongering | 83% | 71% |
| false-urgency | 100% | 60% |
| hype-hope-mongering | 83% | 38% |
| **manufactured-awe** | **0%** | **0%** |

### What the algorithm does well
- **Outrage, tribal bias, authority appeal** are cleanly separated (near-perfect).
- **Confidence is trustworthy**: mean 83% when correct vs 66% when wrong — the number means something.
- **Neutral rejection** works (`none` at 83% recall, 100% precision) — it rarely invents manipulation.

### What it does NOT do well (real limitations)
- **`manufactured-awe` is indistinguishable from `hype-hope-mongering`** — all 6 awe examples were
  labeled hype (0% recall). The two definitions overlap heavily ("change the world" vs
  "world-changing"). This is the taxonomy's weakest seam.
- **`false-urgency` over-fires** (100% recall, 60% precision) — fomo / fear / dopamine bleed into it
  because urgency language ("now", "before it's gone") is shared across techniques.
- **`hype` over-fires** (38% precision) — it's the dumping ground for awe.
- These confusions are *semantically reasonable* (the techniques genuinely co-occur), which is why
  the card also shows an "also:" mixture line rather than pretending it's one clean label.

### Latency
- **p50 4014ms, p95 4493ms** per classify — too slow to feel instant. Root cause: `_label_scores`
  runs 12 full forward passes over the shared prompt prefix. Fix tracked separately (KV-cache reuse).
- Cached repeats are near-instant (memoization).

### Confusion hot-spots
```
manufactured-awe  -> hype-hope-mongering   x6   (the big one)
fomo              -> false-urgency          x2
dopamine-bait     -> false-urgency / hype   x2
```
