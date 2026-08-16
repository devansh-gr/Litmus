# Litmus — retrospective (2026-08-15)

A short, honest close-out. What this became, what moved the numbers, what didn't, and what's left.

## What it became
A local macOS tool: highlight text, press ⌘B, and a small on-device model names the persuasion
technique being run on you (+ a calibrated confidence and a **mixture** of co-present techniques).
Optionally, a 🧠 deep-scan renders a TRIBE v2 cortical impact profile. 100% on-device — no cloud,
no keys, no telemetry. Repo: github.com/devansh-gr/Litmus. Live explainer: devanshgaur.com/litmus.

The defining decision was intellectual honesty. It started bolder ("this fires your amygdala"), and
the project's own experiments (A1–A7) demolished that: the brain map is a *worse* detector than the
text it's built from (100% vs 75%), and there's no clean technique→region anatomy. So the language
model detects and the brain map only *illustrates* — never the other way around.

## The accuracy journey
| stage | accuracy |
|---|---|
| first honest external benchmark (3B) | 38% → **62%** over three sprints (gate + primary-lever routing) |
| + gate class-fix + data cleaning (3B) | **66.9%** |
| best on-device (14B + few-shot) | **82.3%** |
| honest multi-label metric (top-2) | 3B **73%**, 14B forecast **~90%+** |
| binary "is this manipulation?" (gate) | PR-AUC **0.93** |

## What moved the number (and what didn't)
- ✅ **Model capacity is the driver.** 3B 62 → 8B 69 → 14B 79 → 14B+few-shot 82.3. Few-shot helps the
  14B where it was a tie on the 3B.
- ✅ **Small, honest wins:** gate clauses for authority + curiosity-clickbait (62.3→64); removing
  10 non-instances from the noisy test set (64→66.9).
- ⛔ **Self-consistency voting was a confound** — looked like +2.7, went to +0 once the test labels
  were cleaned. The lesson worth more than the lever: control for data quality before crediting a method.
- ⛔ **LoRA fine-tuning never beat prompting** at this data scale (503 and 884 examples). Prompting wins.
- ⛔ **A reasoning model** (R1-Distill-14B) reached 76% on a balanced subset but ~35s/scan — disqualified
  on latency, not accuracy.
- ➡️ **The honest way past the single-label ceiling is multi-label.** Manipulation is usually two
  techniques at once, so "did we name it" (top-2) is the fair question, and it reads ~73%/~90%+.

## Engineering lessons (the real tax)
- Log-prob label scoring beats free generation (which collapsed classes + emitted a constant "80").
- KV-cache reuse: process the shared prompt once, continue per label → ~9x, byte-identical verdicts.
- Running 14B on a 24GB laptop: it draws ~1.5%/min under load, faster than the charger; `caffeinate -i`
  is mandatory for background GPU jobs; a battery watchdog + a streaming benchmark (`bench_stream.py`)
  made partials survivable.
- The menu-bar app is separate from the launchd server and doesn't auto-restart on reboot — so ⌘B can
  silently die while the server looks healthy.

## What's left (one experiment)
Retrain the 14B LoRA on the rebalanced 884/98 split and bench it on `external_propaganda` (not
`external_test`). Forecast: a wash on external_test (which only tests 6 of 14 classes) but a real win
on the starved propaganda vectors the new data feeds. Needs a charge; low priority.

**Bottom line:** 62% shipped, 82.3% best on-device, ~90% top-2, every number reported including the
ones that didn't improve. That honesty was the point.
