# Cortical Persuasion Decoder — agent context

**What it is:** a local macOS tool. Select text, press **⌘B** → it names the persuasion
technique (+ calibrated confidence + runner-up "mixture"). Optionally (🧠 menu → Deep-scan)
it renders a **cortical impact profile** — which brain systems the content engages — via
Meta's TRIBE v2 fMRI model. Everything runs on-device: no cloud, no API keys, no telemetry.

## Architecture — each model does only what it's actually good at
- **Detection = local LLM** (Llama-3.2-3B-Instruct, 4-bit **MLX**). Scores log-prob of each of
  8 vector labels + `none` → argmax + calibrated confidence. This is the verdict.
- **Interpretation = TRIBE v2** (self-hosted, cortex-only fMRI predictor). Renders a cortical
  map. It is NOT a detector.
- Decoupled behind a Swift `Classifier` protocol → local FastAPI server (`server/server.py`):
  `POST /classify` (fast) + `POST /brainmap` (slow, opt-in).

## Hard truths — proven by experiments A1–A7 + I1. DO NOT relitigate.
- **The brain map is NOT a better detector.** A plain Llama text-embedding classifies emotion
  at 100% vs the brain map's 75% (A5); information theory guarantees it (brain = f(text)).
  Detection is the LLM's job; the brain map's only value is *visceral interpretation*, not accuracy.
- **TRIBE is cortex-only** — it cannot see the amygdala / nucleus accumbens. Never say "fires
  your fear center." Say "value / language-evaluation cortex."
- **Fear/outrage/reward hit the SAME fronto-orbital regions** (r≈0.85, A3). There is no
  defensible per-vector→region anatomy. (The old Taxonomy claiming fear→amygdala was deleted;
  `mechanism` strings are now technique-level, not neuroanatomy.)
- **A single sentence's RAW brain map is dominated by auditory + sensorimotor cortex**, not
  semantics. Only a CURATED semantic-ROI set (inferior-frontal / orbital / vmPFC) recovers
  content signal (A7: emotional-vs-neutral d=0.95). `/brainmap` reports exactly those, grouped
  into a profile: Value(orbitofrontal) / Language(inferior-frontal) / Executive(dlPFC).
- **The auditory dominance is model-intrinsic** (TRIBE trained on speech), not from the TTS
  audio — text-only input does NOT fix it (I1). Curated ROIs are mandatory either way.
- The z-score baseline (`baseline.npz`) **must be built from NEUTRAL text**; an emotional
  baseline cancels the very signal you want.

## Engineering gotchas — you WILL hit these
- **Build:** full Xcode + `xcodegen generate` (project.yml → .xcodeproj, git-ignored) →
  `xcodebuild -project … -target CorticalPersuasionDecoder -configuration Debug build`.
  App is a background agent (LSUIElement), **ad-hoc signed, unsandboxed**.
- **Ad-hoc signing ⇒ the Accessibility grant is tied to the code hash.** Every rebuild can
  silently kill the ⌘B hotkey (no error). Re-toggle the app in System Settings ▸ Privacy &
  Security ▸ Accessibility (or sign with an Apple Dev cert for a stable identity).
- **Detector = MLX 4-bit (~2GB) on purpose.** TRIBE hides its OWN 3B text-encoder (~7GB); a
  co-loaded fp32 detector + TRIBE ≈ 13GB → swap-thrash on the 24GB Mac. Server frees TRIBE
  after each `/brainmap`; keep the light MLX detector resident.
- **MLX GPU streams are thread-local** → ALL detector work must run on one dedicated thread
  (`_detector` executor) or FastAPI's threadpool throws "no Stream(gpu,N) in current thread".
- **Llama GQA breaks MPS** (24 Q vs 8 KV heads → mps_matmul "failed to infer result type").
  transformers backend needs `attn_implementation="eager"`; MLX handles it natively.
- **All TRIBE patches live in `server/apply_patches.py`** (idempotent, run after any reinstall):
  WhisperX float16→int8 (Apple CPU), DataLoader `num_workers`→0 (else a silent 4-hour deadlock),
  device routing (audio encoder on MPS = 4m→3s), and **gTTS→offline `say`+ffmpeg** (gTTS
  uploaded the highlighted text to Google — a privacy leak).
- Run Python probes with **`python -u`** — `nohup` buffers stdout and it looks hung at 0% CPU.
- 24GB RAM: watch `sysctl vm.swapusage`. Heavy model churn maxes swap; only a reboot clears it.
  `/brainmap` self-skips when swap >90% and returns an error the card shows.

## Interaction & flow
- ⌘B → synthesize ⌘C to grab the selection → guards (pause / secret-skip / <3 chars / busy) →
  `/classify` → verdict card (vector · confidence · "also:" mixture). Busy-flag serialises
  classify vs deep-scan so two Metal workloads never overlap.
- 🧠 menu bar: Deep-scan cortex (opt-in) · Capture Region (OCR) · Pause capture · Quit.

## Run
```sh
cd server && .venv/bin/python server.py            # wait for "MLX detector ready"
open build/Debug/CorticalPersuasionDecoder.app     # (xcodebuild first if not built)
```
Env: `CPD_CLASSIFIER=mock|remote`, `CPD_LLM_BACKEND=mlx|transformers`, `CPD_HOTKEY=B`,
`CPD_MIN_CONFIDENCE=30`, `CPD_ENDPOINT_URL`.

## Layout & detail
`Sources/` Swift app (App / Capture / Classifier / Overlay / Taxonomy / Support / Hotkey).
`server/` FastAPI server, `build_baseline.py`, `apply_patches.py`, `vendor/tribev2`.
`server/experiments/` the A1–A7 + text_only validation scripts (README table = results).
Full narrative + literature checks: Obsidian vault `03 Projects/Cortical_Persuasion_Decoder/`.
Repo: github.com/devansh-gr/Media_Emotion_Detector (private).

## Open options (undecided — ask before doing)
Switch `/brainmap` to text-only (faster; needs a neutral text-only baseline rebuild) ·
subcortical training / Track B (weeks, interpretation-only, parked) · stable code-signing
(needs Apple ID) · confidence-calibration study · browser extension · vision-LLM for images.
