# Cortical Persuasion Decoder — agent context

**What it is:** a local macOS tool. Select text, press **⌘B** → it names the persuasion
technique (+ calibrated confidence + runner-up "mixture"). Optionally (🧠 menu → Deep-scan)
it renders a **cortical impact profile** — which brain systems the content engages — via
Meta's TRIBE v2 fMRI model. Everything runs on-device: no cloud, no API keys, no telemetry.

## Architecture — each model does only what it's actually good at
- **Detection = local LLM** (Llama-3.2-3B-Instruct, 4-bit **MLX**), **two-stage**: (1) a yes/no
  **manipulation gate** (`GATE_SYSTEM`) — benign text (greetings, requests, facts, reviews) → `none`;
  (2) only if it passes, score log-prob of each vector label → argmax + calibrated confidence.
  The gate exists because a single 12-way classifier forced greetings into "hype 84%". The shown
  confidence is **Platt-calibrated** (`CPD_CALIB_A/B`); below `CPD_ABSTAIN_BELOW` (0.45) the card
  flags `uncertain`. Gate uses `GATE_TEMP=1.0` (unsharpened) + `CPD_GATE_NONE_THRESHOLD` (0.70,
  swept on external-dev). `CPD_CONTEXTUAL_CALIB` (Calibrate-Before-Use) available. **Honest external
  benchmark (300 examples we didn't author): ~41% accuracy, gate PR-AUC 0.92 / recall 0.66, ECE
  0.10** — see `tests/RESULTS.md` + `docs/BENCHMARKING.md`. This is the verdict.
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
- **`/brainmap` now ships the TEXT-ONLY path by default** (`CPD_BRAINMAP_MODE=text`): synthetic
  word events, NO TTS / NO WhisperX. Re-validated end-to-end against a text-only neutral
  baseline — emotional vs neutral still separates at held-out **d=0.77** (large), carried by the
  Language/inferior-frontal system (`verify_text_only_baseline.py`). The baseline MUST match the
  mode (server refuses a mismatch); the old audio baseline is kept as `baseline_audio.npz`.
- The z-score baseline (`baseline.npz`) **must be built from NEUTRAL text**; an emotional
  baseline cancels the very signal you want.

## Engineering gotchas — you WILL hit these
- **Build:** `scripts/build.sh` (xcodegen generate → xcodebuild Debug → re-sign). App is a
  background agent (LSUIElement), unsandboxed. Do NOT rely on Xcode's automatic ad-hoc signing.
- **Stable signing is SOLVED (self-signed, no Apple ID).** Ad-hoc signing tied the Accessibility
  (TCC) grant to the code hash, so every rebuild silently killed ⌘B. Fix: a persistent
  self-signed identity **"CPD Local Signing"** (`scripts/make_signing_identity.sh`, run once —
  creates + trusts a local code-signing cert). `scripts/sign_app.sh` re-signs the build with it,
  giving a STABLE designated requirement (`identifier … and certificate leaf = H"b33a…"`) that
  survives rebuilds → the grant sticks. `build.sh` chains all three. One-time cost: after the
  FIRST stable-signed build the grant must be re-added once (the requirement changed from the old
  ad-hoc hash); after that, rebuilds keep it. Falls back to ad-hoc with a warning if the identity
  is absent (fresh checkout still runs).
- **Detector = MLX 4-bit (~2GB) on purpose.** TRIBE hides its OWN 3B text-encoder (~7GB); a
  co-loaded fp32 detector + TRIBE ≈ 13GB → swap-thrash on the 24GB Mac. Server frees TRIBE
  after each `/brainmap`; keep the light MLX detector resident.
- **MLX GPU streams are thread-local** → ALL detector work must run on one dedicated thread
  (`_detector` executor) or FastAPI's threadpool throws "no Stream(gpu,N) in current thread".
- **Llama GQA breaks MPS** (24 Q vs 8 KV heads → mps_matmul "failed to infer result type").
  transformers backend needs `attn_implementation="eager"`; MLX handles it natively.
- **`num_workers` MUST be forced to 0 AFTER `from_pretrained`.** The checkpoint's own config
  resets `data.num_workers` to `N_CPUS` (~20) at load time, silently overriding the source
  default — those forked DataLoader workers DEADLOCK on macOS (froze the baseline build at 2/30,
  0% CPU, and spawn ~14 stuck children). `tribe_events.harden_tribe(model)` sets it back to 0 and
  MUST run after every load (both `server.get_tribe()` and `build_baseline.py` call it). The
  `apply_patches.py` source patch is necessary but NOT sufficient — the checkpoint wins.
- **Other TRIBE patches live in `server/apply_patches.py`** (idempotent, run after any reinstall):
  WhisperX float16→int8 (Apple CPU), device routing (audio encoder on MPS = 4m→3s), and
  **gTTS→offline `say`+ffmpeg** (gTTS uploaded the highlighted text to Google — a privacy leak).
  Note the default text-only path uses none of the audio patches; they matter only in `audio` mode.
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
./scripts/build.sh                                 # xcodegen + xcodebuild + stable re-sign
open build/Debug/CorticalPersuasionDecoder.app
```
**The server runs 24/7 via a launchd agent** (`scripts/launchd/ai.zeonsystems.cpd.server.plist`,
installed at `~/Library/LaunchAgents/`): `RunAtLoad` + `KeepAlive` → starts at login and respawns
on crash; logs to `~/Library/Logs/cpd_server.log`; port 8765. So you normally do NOT hand-run
`server.py` — that would double-bind the port. Manage it with launchctl (gui/$(id -u) domain):
`bootstrap`/`bootout` to install/remove, `kickstart -k …/ai.zeonsystems.cpd.server` to restart.
It's a LaunchAgent, so it runs while the user is logged in (not at the login screen). The **🧠
menu-bar icon is the APP**, a separate process; the server is headless with no icon.
Env: `CPD_CLASSIFIER=mock|remote`, `CPD_LLM_BACKEND=mlx|transformers`, `CPD_HOTKEY=B`,
`CPD_ENDPOINT_URL`, `CPD_BRAINMAP_MODE=text|audio` (default `text`;
rebuild `baseline.npz` with the matching mode via `build_baseline.py`),
`CPD_TRIBE_WARM_SECS` (keep TRIBE resident N s after a `/brainmap` so back-to-back deep
scans skip the ~7GB reload; launchd default 120, code default 0 = free immediately).

## Deep-scan performance
`/brainmap` is ~15s when RAM is healthy — most of it is the ~7GB TRIBE reload (freed after
each call). Two levers, both safe: (1) an **exact-text result cache** (`_brainmap_cache`) — a
repeated scan returns instantly with NO model load, checked *before* the swap guard so it
works even under memory pressure; (2) **`CPD_TRIBE_WARM_SECS`** keeps TRIBE resident for a
window so consecutive scans skip the reload — self-protecting because a scan can't start while
swap > 90%. `baseline.npz` is also cached in memory (`_get_baseline`; rebuilding it needs a
server restart). NOTE: none of this runs while swap is maxed — `/brainmap` returns a "reboot to
reclaim swap" error; measure warm-vs-cold only after a reboot.

## Layout & detail
`Sources/` Swift app (App / Capture / Classifier / Overlay / Taxonomy / Support / Hotkey).
`server/` FastAPI server, `tribe_events.py` (mode dispatch + `harden_tribe`), `build_baseline.py`,
`apply_patches.py`, `vendor/tribev2`.
`server/experiments/` the A1–A7 + text-only validation scripts (README table = results).
`scripts/` `make_signing_identity.sh` · `sign_app.sh` · `build.sh`.
Full narrative + literature checks: Obsidian vault `03 Projects/Cortical_Persuasion_Decoder/`.
Repo: github.com/devansh-gr/CorticalPersuasionDecoder (private; renamed from Media_Emotion_Detector, old URL redirects).

## Demo (how to film it)
Lead with the magic: highlight manipulative text → ⌘B → verdict card. Then range (3 tactics),
a neutral control ("No manipulation detected ✓"), then the Deep-scan wow, then the on-device
close. Use a keystroke visualizer (KeyCastr) + Focus/DND + Screen Studio. Capture Region now
works too (needs Screen Recording granted). Warm the deep-scan with a throwaway scan first (or
reboot if swap is high).
Verified example texts: `Act now or lose everything forever.` → false-urgency ~99%; `This will
completely change your life and the entire world forever.` → hype ~93%; `Everyone is switching,
don't get left behind.` → fomo ~78%; a plain factual sentence → neutral. Full playbook +
shot list in the vault: `04 Skills/(C) Demo Filming Guide.md`. Overview docs: `docs/`.

## Testing & benchmark (server/tests/)
- **HONEST benchmark:** `python tests/bench_full.py --data tests/data/external_holdout.jsonl` —
  300 externally-labeled examples (dark-patterns + LOGIC + clickbait) we did NOT author or tune on.
  **~41% accuracy, macro-F1 0.19, gate PR-AUC 0.92 / recall 0.66, ECE 0.10.** Self-authored sets
  (`run_eval.py`, `realistic_set`) read ~80% but that's overfit — grading your own homework.
  `bench_full.py` gives macro-F1 + Wilson/bootstrap CIs, per-class, confusion, MCC, gate PR-AUC,
  and calibration (ECE). Data hygiene + caveats: `tests/data/README.md`. Full roadmap: `docs/BENCHMARKING.md`.
- **pytest** (`tests/`, ~35 green): contract (JSON shape the Swift decoder needs), behavior,
  taxonomy-sync (server VECTORS == Swift enum), robustness, benign/realistic FP guards,
  **fomo↔false-urgency separation**, **calibration + abstain**. Hit the live server, skip if down.
  `pip install -r tests/requirements-test.txt` (pytest + sklearn/scipy for bench_full).
- **Benchmarking follow-ups** (see `docs/BENCHMARKING.md` "Status"): external coverage for fear /
  outrage / tribal / critical-thinking / hype / manufactured-awe (no clean dataset); the clickbait→
  dopamine mapping is loose; CheckList robustness; Tier-3 multi-annotator ground truth. Deep-scan
  latency still needs a **reboot** to re-measure (swap keeps maxing).
- **Latency bench:** `python bench/latency_bench.py` (cold/warm/cached). Classify p50 ~450ms
  after the KV-cache fix (was ~4s).
- **Swift self-tests** (`scripts/`, no permission/server needed): `ocr_selftest.swift` (OCR round-trip)
  and `sensitive_selftest.swift` (secret guard) — compile with `swiftc` against the real sources.

## Known issues (TODO)
- **Capture Region now WORKS** (2026-07-29). It was blocked by (a) Screen Recording not granted
  and (b) the `.transient` overlay bug (the card was hidden because this LSUIElement app is never
  "active"). Both fixed; the pipeline logs a successful drag-select → screenshot → OCR → verdict.
  Clean-shot fixes (exclude own windows + 120ms settle) landed too. If it regresses, re-grant
  Screen Recording and REOPEN the app. Everything works: ⌘B classify (instant "Analyzing…"),
  Deep-scan, Capture Region, Pause, Quit, + menu QoL (last verdict / recent history / copy /
  analyze-clipboard).

## Open options (undecided — ask before doing)
Subcortical training / Track B (weeks, interpretation-only, parked) · confidence-calibration
study · browser extension · vision-LLM for images.
(DONE, no longer options: text-only `/brainmap` is the default; stable self-signed code-signing.)
