# Server test & benchmark suite

Validates the detector — the LLM `/classify` path. Tests hit the **live** server
(`:8765`) so they exercise the real deployed path, and **skip** (not fail) if it's down.

## Run

```sh
cd server
.venv/bin/pip install -r tests/requirements-test.txt   # once (pytest)
.venv/bin/python -m pytest tests/ -q                   # unit / contract / behavior
.venv/bin/python tests/run_eval.py                     # accuracy benchmark (72 examples)
.venv/bin/python bench/latency_bench.py                # latency (cold/warm/cached)
```

## What's here

| File | What it checks |
|---|---|
| `data/eval_set.jsonl` | Labeled benchmark — 6 examples per vector (+ neutral). |
| `run_eval.py` | Overall accuracy, per-vector recall/precision, confusions, confidence calibration, p50/p95. Exit ≠0 if accuracy < 70% (CI gate). |
| `RESULTS.md` | Latest recorded run + interpretation (strengths / weaknesses). |
| `test_classify_contract.py` | The JSON shape the Swift `RemoteClassifier` decodes. |
| `test_classify_behavior.py` | Must-hold semantics: neutral→none, clear cases hit + confident, determinism. |
| `test_taxonomy_sync.py` | Server `VECTORS`/`DEFINITIONS` == Swift `PersuasionVector` (drift guard). |
| `test_robustness.py` | Long / emoji / other-script / whitespace input never crashes; `/health`. |
| `../bench/latency_bench.py` | Cold vs warm vs cached classify latency. |

Swift-side self-tests live in `../../scripts/`: `ocr_selftest.swift`, `sensitive_selftest.swift`.

## Current status
15 pytest tests green · benchmark 86.1% accuracy · classify p50 ~450ms (cached ~1ms).
See `RESULTS.md` for the per-vector breakdown and known weak seams.
