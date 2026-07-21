# Validation experiments

These are the scripts behind the project's claims — run once during the research
phase, kept for provenance/reproducibility. Not part of the running product.

| Script | Question it answered | Result |
|---|---|---|
| `a1_smoketest.py` | Does TRIBE v2 run on Apple Silicon at all? | Yes — `(5, 20484)` fsaverage5 predictions on MPS. |
| `a2_signal_test.py` | Does a single sentence produce *any* systematic brain signal? | Yes — 34/75 regions separate fear vs neutral (p<0.05 vs ~3.8 by chance). |
| `a2_timing.py` | True cold per-sentence latency? | ~76 s (16 s events + 60 s predict), warm ~7 s. |
| `a3_emotion_test.py` | Do fear/outrage/reward hit *distinct* regions? | No — same fronto-orbital regions (r=0.83–0.90); a learned decoder still gets 75% 4-way. |
| `a4_text_baseline.py` | Does bag-of-words beat the brain map? | No (43% vs 75%) — but TF-IDF was a strawman; uninformative. |
| `a5_llama_baseline.py` | Does the *Llama embedding* beat the brain map? | Yes, 100% vs 75% → the brain projection adds **no** detection power. |
| `a6_offline_tts_check.py` | Is the gTTS→Google privacy leak gone? | Yes — offline macOS `say`, `(4, 20484)` still produced. |
| `text_only_probe.py` | Does feeding TRIBE word-events *without* audio remove the TTS/auditory confound? | No — auditory is model-intrinsic (still d=1.72); but text-only is feasible, faster, drops all audio deps, and preserves the semantic signal (d=0.99). |
| `a7_curated_roi.py` | Can *curated* semantic ROIs separate emotional per-sentence? | Yes — d=0.95, p=0.02 → the impact profile is honest. |

Full narrative + literature verification live in the Obsidian vault
(`Cortical_Persuasion_Decoder/01 In Progress/`).
