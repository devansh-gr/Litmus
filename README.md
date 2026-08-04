# Cortical Persuasion Decoder

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
| accuracy | ~80% | **~45%** |
| gate PR-AUC (ranks manipulation vs benign) | — | **0.92** |
| gate recall @ shipped threshold | — | **0.66** |
| confidence calibration (ECE, lower=better) | — | **0.10** |

The honest read: the detector **tells manipulation from benign decently** (gate PR-AUC
0.92) and its confidence now **means something** (when it says 70%, it's right ~70% of
the time — calibrated with Platt scaling). But the specific *technique* labels are shakier
(the clickbait→dopamine mapping is loose, urgency/FOMO overlap), and the ~35-point gap from
the self-authored number is the classic cost of grading your own homework. We show these
numbers on purpose — the full methodology and roadmap live in `docs/BENCHMARKING.md`.

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
| `CPD_GATE_NONE_THRESHOLD` | `0.70` | gate cutoff (tuned on external data by F0.5) |
| `CPD_CALIB_A` / `CPD_CALIB_B` | `0.56` / `-1.10` | Platt calibration of the shown % |
| `CPD_ABSTAIN_BELOW` | `0.45` | below this calibrated confidence, flag `uncertain` |
| `CPD_CONTEXTUAL_CALIB` | `0` | Calibrate-Before-Use gate debias (needs re-tune; off) |
| `CPD_LLM_BACKEND` | `mlx` | `mlx` (4-bit ~2 GB) or `transformers` (fp32 ~6.5 GB) |
| `CPD_BRAINMAP_MODE` | `text` | text-only (default) or `audio` TRIBE path |
| `CPD_TRIBE_WARM_SECS` | `120` | keep TRIBE warm between deep-scans |
| `CPD_HOTKEY` | `B` | the trigger key |

---

## Testing & benchmarking

```sh
cd server
.venv/bin/python -m pytest tests/ -q                                    # 30 behavior/contract tests
.venv/bin/python tests/bench_full.py --data tests/data/external_holdout.jsonl  # honest metrics
.venv/bin/python tests/run_eval.py                                      # quick accuracy
```
`bench_full.py` reports macro-F1 + Wilson/bootstrap CIs, per-class P/R, confusion, MCC,
**gate PR-AUC**, and **confidence calibration (ECE + reliability)** — the metrics the
methodology research says a subjective, imbalanced, two-stage, confidence-scored
classifier actually needs. Swift self-tests (OCR, secret-guard) live in `scripts/`.

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
- Interactive **[architecture](https://claude.ai/code/artifact/33401b3a-e07f-4934-88b8-03056054a5fb)**
  and **[field manual](https://claude.ai/code/artifact/d5fc32a8-e2dd-4f29-a041-e55580e8b2db)** pages.

## Requirements
macOS 14+, Apple Silicon (uses MPS), Xcode, Homebrew, `uv`, and a Hugging Face account
with the Llama-3.2 license accepted.

---

*Built to defend attention, not harvest it — which is why it runs entirely on your machine.*
