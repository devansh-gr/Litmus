# Litmus

**A cognitive-manipulation x-ray for your Mac.** Highlight any text — a headline, a
DM, an ad, an AI answer — press **⌘B**, and a local AI names the persuasion technique
being used on you, with an honest confidence. On demand, it shows *where in your
cortex* the words land.

Everything runs **on-device**. No cloud, no API keys, no telemetry — the text you
analyze never leaves the machine.

> `⌘B` on *"Act now or lose everything forever."* → **False urgency · 87%**
> `⌘B` on *"Hello! How can I help you today?"* → **No manipulation ✓**

---

## The one idea

Two models, each doing only what it's genuinely good at:

| | **Detection** | **Interpretation** |
|---|---|---|
| model | local Llama-3.2-3B (4-bit MLX, ~2 GB) | Meta's **TRIBE v2** fMRI model |
| job | *names* the technique + confidence | shows *where in cortex* it lands |
| speed | ~0.6 s | ~15 s (opt-in) |
| role | the verdict | a visceral picture — **never** the detector |

Why the split? Because the project's own experiments proved the brain map is a
*worse* detector than the text it's built from (100% vs 75% — information theory
guarantees it). So detection is the language model's job; the brain map exists to make
manipulation *felt*, not to raise accuracy. It's also **cortex-only**, so we say
"engages your value / language cortex," never "fires your fear center."

### Detection is two-stage

A single 14-way classifier forced *every* input toward its nearest label — so a
friendly greeting came back "Hype 84%." The fix is a **gate**:

```
text ──▶ [ Stage 1: is this manipulation at all? ] ──no──▶ "No manipulation ✓"
                         │ yes
                         ▼
              [ Stage 2: which of 14 techniques? ] ──▶ verdict + calibrated confidence
```

The gate draws the line by *intent* — "the sale ends Friday" is ordinary; "hurry,
don't miss out!" is manipulation. The 14 techniques span two families: **broadcast
persuasion** (fear, urgency, hype, FOMO, authority, social proof, …) and
**interpersonal manipulation** (guilt-tripping, gaslighting, love-bombing,
blame-shifting/DARVO).

---

## Does it actually work? (honest benchmarking)

Most detectors quote an accuracy number computed on data the authors wrote themselves.
That number is almost always inflated. So we measured on **300 independently-labeled
examples from public datasets** (dark-patterns + logical-fallacy + clickbait corpora)
that we neither wrote nor tuned on:

| | self-authored (tuned on) | **external (never seen)** |
|---|---|---|
| accuracy (shipped 3B, single-label) | ~80% | **~62%** (was 38% three sprints ago) |
| accuracy (best on-device, 14B + few-shot) | — | **82.3%** |
| top-2 accuracy (honest multi-label metric) | — | **73%** (3B) · ~90%+ forecast (14B) |
| macro-F1 | — | **0.37–0.49** (config-dependent) |
| gate PR-AUC (ranks manipulation vs benign) | — | **0.93** |
| gate recall @ shipped threshold | — | **0.78** |
| confidence calibration (ECE, lower=better) | — | **~0.19** uncalibrated (0.10 with calibration on) |

The honest read: the detector **tells manipulation from benign well** (gate PR-AUC 0.93,
recall 0.78 — up from 0.56 after we taught the gate e-commerce dark patterns it was reading
as plain facts) **and now names the technique far better** (accuracy 44→61% after teaching
stage-2 to route by the *primary lever* — supply=FOMO, clock=urgency, crowd=social-proof,
reward/curiosity=dopamine, credential=authority). We then reported every lever, including the
failures. On the 3B, few-shot and a LoRA fine-tune both tied at ~62% (its ceiling). The real
driver was **model size**: a bigger on-device model climbs 3B 62% → 8B 69% → 14B 79%, and the
**14B with few-shot reaches 82.3%** (the fast 3B stays the instant default; the 14B is an opt-in
high-accuracy mode). Self-consistency voting *looked* like a win but was a confound (+0 once we
cleaned the test data), and the fine-tune never beat plain prompting. And because manipulation is
often two techniques at once, we also report a **top-2** number: ~73% on the 3B, ~90%+ forecast on
the 14B — the honest question is "did we name the manipulation," not "which single label." The shown confidence is **assertive by product choice** —
calibration is available (`CPD_CALIB_A/B`, ECE→0.10) but ships off so the % reads high. What's
left is mostly gate misses and cross-taxonomy label noise. We show these numbers on purpose —
the full methodology lives in `docs/BENCHMARKING.md` and `server/lora/README.md`.

📊 Full methodology and roadmap: [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) ·
results: [`server/tests/RESULTS.md`](server/tests/RESULTS.md)

---

## Quick start

**1. Server** (holds the models; runs on `127.0.0.1:8765`):
```sh
cd server
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e vendor/tribev2 fastapi uvicorn nilearn scipy mlx mlx-lm scikit-learn
.venv/bin/python apply_patches.py        # CUDA-free (Apple Silicon) + offline-TTS patches
.venv/bin/hf auth login                  # accept meta-llama/Llama-3.2-3B on HF first
.venv/bin/python build_baseline.py       # writes baseline.npz for /brainmap (one-time)
.venv/bin/python server.py               # or run 24/7 via scripts/launchd/ (see below)
```

**2. App:**
```sh
brew install xcodegen
./scripts/build.sh                       # xcodegen + xcodebuild + stable re-sign
open build/Debug/CorticalPersuasionDecoder.app
```
Grant **Accessibility** when prompted (so ⌘B can read your selection), then highlight
text anywhere and press **⌘B**. For the brain map: **🧠 menu → Deep-scan cortex**.

The server is meant to run **24/7** — install `scripts/launchd/ai.zeonsystems.cpd.server.plist`
to `~/Library/LaunchAgents/` and it starts at login and restarts on crash.

---

## Configuration (env)

| var | default | what |
|---|---|---|
| `CPD_GATE_NONE_THRESHOLD` | `0.65` | gate cutoff (swept on external-dev) — lower flags more |
| `CPD_CALIB_A` / `CPD_CALIB_B` | `1.0` / `0.0` | Platt calibration of the shown %; **off by default** (assertive). Set `0.56` / `-1.10` for the honest (lower, calibrated) % |
| `CPD_ABSTAIN_BELOW` | `0.45` | below this confidence, flag the technique as `uncertain` |
| `CPD_FEWSHOT` | `1` | few-shot exemplars on the technique prompt (gate stays zero-shot) |
| `CPD_MLX_ADAPTER` | *(empty)* | path to an optional LoRA adapter (`server/lora/`); empty = base model |
| `CPD_CONTEXTUAL_CALIB` | `0` | Calibrate-Before-Use gate debias (evaluated → no gain; off) |
| `CPD_LLM_BACKEND` | `mlx` | `mlx` (4-bit ~2 GB) or `transformers` (fp32 ~6.5 GB) |
| `CPD_BRAINMAP_MODE` | `text` | text-only (default) or `audio` TRIBE path |
| `CPD_TRIBE_WARM_SECS` | `0` | keep TRIBE warm N s between deep-scans (launchd sets `120`) |
| `CPD_HOTKEY` | `B` | the trigger key |

---

## Testing & benchmarking

```sh
cd server
.venv/bin/python -m pytest tests/ -q                                    # ~65 behavior/contract tests
.venv/bin/python tests/bench_full.py --data tests/data/external_test.jsonl  # honest metrics
.venv/bin/python tests/gate_diag.py --data tests/data/external_dev.jsonl    # gate recall diag + sweep
.venv/bin/python tests/interpersonal_report.py                         # interpersonal family
```
`bench_full.py` reports macro-F1 + Wilson/bootstrap CIs, per-class P/R, confusion, MCC,
**gate PR-AUC**, and **confidence calibration (ECE + reliability)** — the metrics the
methodology research says a subjective, imbalanced, two-stage, confidence-scored
classifier actually needs. `gate_diag.py` localizes what the gate misses (it's how the
dark-pattern recall gap was found and fixed). Regression suites `test_dark_patterns.py`
and `test_interpersonal.py` lock in the recall/precision wins. Swift self-tests (OCR,
secret-guard) live in `scripts/`.

---

## Repo layout

```
Sources/CorticalPersuasionDecoder/   # the SwiftUI/AppKit menu-bar app
  App/         menu bar, ⌘B handling, capture→classify→overlay wiring
  Capture/     PasteboardCapture (⌘B → silent copy), Region + OCR
  Classifier/  Classifier protocol · RemoteClassifier (HTTP) · MockClassifier
  Overlay/     floating verdict card + mini brain
  Taxonomy/    the 14 vectors (persuasion + interpersonal) + plain-English mechanisms
  Support/     config, permissions, sensitive-text guard
server/                              # the local inference server
  server.py            FastAPI: /classify (two-stage gate + LLM) · /brainmap (TRIBE v2)
  tribe_events.py      text-only vs audio event dispatch for TRIBE
  build_baseline.py    builds the neutral z-score baseline
  tests/               eval sets (dev + external) · bench_full.py · pytest suite
  experiments/         A1–A7: the validation behind the two-model split
docs/                                # architecture, field manual, benchmarking
```

## Further reading
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the whole system on one page.
- **[docs/BENCHMARKING.md](docs/BENCHMARKING.md)** — how to properly benchmark this, and why.
- Interactive **[docs/architecture.html](docs/architecture.html)** and
  **[docs/field-manual.html](docs/field-manual.html)** — the same, as self-contained pages (open locally).

## Requirements
macOS 14+, Apple Silicon (uses MPS), Xcode, Homebrew, `uv`, and a Hugging Face account
with the Llama-3.2 license accepted.

---

*Built to defend attention, not harvest it — which is why it runs entirely on your machine.*
