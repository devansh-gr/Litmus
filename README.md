# Cortical Persuasion Decoder

A local, real-time "cognitive-manipulation x-ray" for macOS. Select any text and
press **⌘B**, and it names the **persuasion technique** the content uses — then,
on demand, shows the **cortical impact profile**: which of your brain's cortical
systems the content recruits.

Everything runs **on your machine**. No cloud, no API keys, no telemetry — the
text you analyze never leaves the Mac. (The ⌘B hotkey needs a one-time
**Accessibility** grant so the app can read your selection.)

## Architecture — the two halves do what each is actually good at

- **Detection** (fast, ~3 s): a local **Llama-3.2-3B-Instruct** scores the
  log-probability of each persuasion-vector label → the technique + a *calibrated*
  confidence. This is the x-ray verdict.
- **Interpretation** (opt-in, ~2 min): **Meta's TRIBE v2** (a foundation model of
  fMRI brain responses, trained on 700+ subjects) predicts cortical activation for
  the text, z-scored vs a neutral baseline and grouped into named systems
  (language / value-evaluation / executive) with an engagement level.

The two are decoupled behind a `Classifier` protocol (`MockClassifier` for
offline dev, `RemoteClassifier` → the local server).

### Honest scope (see the vault write-ups for the evidence)
- The brain map is **not a better detector** — a plain text model classifies at
  100% vs the brain map's 75%; the brain map is a deterministic function of the
  text, so it cannot add detection signal. Its value is making the manipulation
  *visceral and grounded* ("this recruited your value cortex"), not raising
  accuracy.
- It is **cortex-only** — it cannot see the amygdala / nucleus accumbens, so we
  say "value-evaluation cortex," never "your fear center."

## Repo layout

```
Sources/CorticalPersuasionDecoder/   # the SwiftUI/AppKit app
  App/         entry point, menu bar, capture→classify→overlay wiring
  Capture/     PasteboardCapture (⌘B → copy), RegionSelector + RegionCapture + OCR
  Hotkey/      HotkeyMonitor (global ⌘B, non-consuming)
  Classifier/  Classifier protocol, MockClassifier, RemoteClassifier (HTTP)
  Overlay/     floating verdict card + cortical impact profile
  Taxonomy/    8-vector → brain-region → mechanism table
  Support/     Config (mock/remote, thresholds)
server/                              # the local inference server
  server.py            FastAPI: /classify (LLM) + /brainmap (TRIBE v2)
  build_baseline.py    builds baseline.npz from a neutral corpus
  apply_patches.py     idempotent CUDA-free / privacy patches for TRIBE v2
  experiments/         a1–a7: the validation scripts behind the claims above
```

## Running it

### 1. Server (once)
```sh
cd server
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e vendor/tribev2 fastapi uvicorn nilearn scipy
.venv/bin/python apply_patches.py         # CUDA-free (Apple Silicon) + offline-TTS patches
.venv/bin/hf auth login                   # gated: accept meta-llama/Llama-3.2-3B on HF first
.venv/bin/python build_baseline.py        # writes baseline.npz (~20 min, one-time)
.venv/bin/python server.py                # serves 127.0.0.1:8765
```
(`vendor/tribev2` = a checkout of `github.com/facebookresearch/tribev2`.)

### 2. App
```sh
brew install xcodegen        # once
xcodegen generate
xcodebuild -project CorticalPersuasionDecoder.xcodeproj -target CorticalPersuasionDecoder -configuration Debug build
open build/Debug/CorticalPersuasionDecoder.app
```
Then grant Accessibility when prompted, select text anywhere + **⌘B** → verdict
card appears (override the key with `CPD_HOTKEY=D` etc.). For the cortical
profile, click the **🧠 menu bar → Deep-scan cortex**.

### Config (env)
- `CPD_CLASSIFIER=mock|remote` (default remote)
- `CPD_ENDPOINT_URL` (default `http://127.0.0.1:8765`)
- `CPD_MIN_CONFIDENCE` (default 30 — suppress weak verdicts)
- `CPD_LLM` (default `meta-llama/Llama-3.2-3B-Instruct`)

## Requirements
macOS 14+, Apple Silicon (uses MPS), full Xcode, Homebrew, `uv`, a Hugging Face
account with the Llama-3.2 license accepted.
