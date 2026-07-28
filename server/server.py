"""Local inference server for Cortical Persuasion Decoder.

Two endpoints, split by what each model is ACTUALLY good at (see A5 in the vault):

  POST /classify   -> Llama-3.2-3B-Instruct names the persuasion vector.
                      FAST (~1-3s). This is the DETECTOR. A5 proved the text
                      model beats the brain map at detection (100% vs 75%), and
                      information theory says it must (brain = f(text)).

  POST /brainmap   -> TRIBE v2 predicts cortical activation for the text and we
                      reduce 20484 fsaverage5 vertices to named Destrieux regions.
                      SLOW (~22s). This is the INTERPRETER, not a detector. It is
                      the only component that can honestly say WHERE in cortex the
                      content lands, grounded in a model trained on 700+ subjects.

Everything runs locally. No text ever leaves the machine (offline TTS patched in).
"""

import json
import os
import logging
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from tribe_events import BRAINMAP_MODE, build_events, harden_tribe

HERE = Path(__file__).parent
CACHE = HERE / "cache"
# 3B detector (accurate: 5/5 in testing; the 1B mislabels neutral text). The 24GB
# memory pressure came from co-loading TRIBE v2 -- which itself contains a 3B Llama
# text-encoder (~13GB of models total). Fix is architectural (free TRIBE after each
# /brainmap, see below), not a smaller detector. Override with CPD_LLM if desired.
LLM_NAME = os.environ.get("CPD_LLM", "meta-llama/Llama-3.2-3B-Instruct")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cpd")

VECTORS = [
    "fear-mongering",
    "critical-thinking-suppression",
    "tribal-in-group-bias",
    "dopamine-bait",
    "outrage",
    "authority-appeal",
    "false-urgency",
    "social-proof-conformity",
    "hype-hope-mongering",
    "fomo",
    "manufactured-awe",
    "none",
]

# Explicit definitions: the 3B model collapsed dopamine-bait and social-proof
# into "false-urgency" without them.
DEFINITIONS = {
    "fear-mongering": "makes you afraid of harm, danger, disease or catastrophe",
    "critical-thinking-suppression": "tells you not to question, think or investigate",
    "tribal-in-group-bias": "pits 'us' against 'them', an in-group versus an enemy",
    "dopamine-bait": "dangles a reward, prize, win, jackpot, freebie or exclusive unlock",
    "outrage": "invites moral anger and disgust at someone's behaviour",
    "authority-appeal": "leans on experts, officials, science or authority to settle it",
    "false-urgency": "imposes a deadline or time pressure: act now, expires, last chance",
    "social-proof-conformity": "everyone else is doing it, don't be left out, join the crowd",
    "hype-hope-mongering": "sells an exciting or utopian upside — this will change your life or the world — to make you want in",
    "fomo": "plays on fear of being left behind or missing the future if you don't get on board now",
    "manufactured-awe": "inflates something as revolutionary, unprecedented or world-changing to overwhelm skepticism",
    "none": "neutral, factual or informational; no manipulation",
}

SYSTEM = (
    "You are a persuasion analyst. Identify the PRIMARY manipulation technique in "
    "the text. The techniques are:\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    + "\nAnswer with the technique name only."
)

@asynccontextmanager
async def _lifespan(app):
    # Load the detector at startup (in the background) so the first ⌘B doesn't eat
    # a cold load in the request path. TRIBE v2 stays lazy (per-brainmap).
    threading.Thread(target=warm_detector, daemon=True).start()
    yield


app = FastAPI(title="Cortical Persuasion Decoder", lifespan=_lifespan)

# Detector backend: "mlx" (4-bit, ~2GB, Apple-native — default) or "transformers"
# (fp32, ~6.5GB). MLX is far lighter, so it coexists with TRIBE without thrashing.
LLM_BACKEND = os.environ.get("CPD_LLM_BACKEND", "mlx")
MLX_MODEL = os.environ.get("CPD_MLX_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit")

# Confidence sharpening. With 12 candidate labels a raw softmax dilutes the winner
# (a clear false-urgency landed at ~20%), which reads as unsure. Dividing the
# scores by a temperature < 1 sharpens the distribution so the top vector reports
# an ASSERTIVE confidence. Lower = more assertive. Override with CPD_CONFIDENCE_TEMP.
CONFIDENCE_TEMP = float(os.environ.get("CPD_CONFIDENCE_TEMP", "0.25"))

# Keep TRIBE resident for this many seconds after a /brainmap so back-to-back deep
# scans skip the ~7GB reload (the reload is most of the ~15s a warm scan takes).
# 0 = free immediately (safest on a low-RAM machine). The warm window is self-
# protecting: a scan can't start while swap > 90%, so TRIBE only lingers when RAM
# is actually healthy.
TRIBE_WARM_SECS = float(os.environ.get("CPD_TRIBE_WARM_SECS", "0"))

_lock = threading.Lock()
_llm = None
_tok = None
_mlx = None
_tribe = None
_atlas = None
_baseline = None                        # cached baseline.npz arrays (mean/std/n/mode)
_free_timer = None                      # idle timer that frees TRIBE after the warm window
_classify_cache: dict[str, dict] = {}   # memoize verdicts by exact text
_brainmap_cache: dict[str, dict] = {}   # memoize cortical profiles by exact text

# MLX's GPU stream is thread-local, so ALL detector work (load + inference) must
# run on one dedicated thread — otherwise FastAPI's threadpool hops workers and
# MLX raises "no Stream(gpu, N) in current thread". Also serialises requests.
_detector = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")


class TextIn(BaseModel):
    text: str


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_llm():
    global _llm, _tok
    with _lock:
        if _llm is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("loading %s ...", LLM_NAME)
            _tok = AutoTokenizer.from_pretrained(LLM_NAME)
            # attn_implementation="eager" is REQUIRED on MPS. Llama 3.2 uses
            # grouped-query attention (24 Q heads, 8 KV heads); the SDPA path
            # broadcasts 8->24 inside mps_matmul, which Metal cannot infer and
            # which hard-kills the process ("Failed to infer result type").
            # Eager calls repeat_kv() to materialise 24 KV heads before the
            # matmul, avoiding the broadcast entirely.
            _llm = (
                AutoModelForCausalLM.from_pretrained(
                    LLM_NAME, dtype=torch.float32, attn_implementation="eager"
                )
                .to(_device())
                .eval()
            )
            log.info("LLM ready on %s (eager attn)", _device())
    return _llm, _tok


def get_mlx():
    global _mlx
    with _lock:
        if _mlx is None:
            from mlx_lm import load
            log.info("loading MLX %s ...", MLX_MODEL)
            _mlx = load(MLX_MODEL)
            log.info("MLX detector ready")
    return _mlx


def warm_detector():
    """Load the active detector backend on the dedicated detector thread."""
    _detector.submit(get_mlx if LLM_BACKEND == "mlx" else get_llm)


def _chat_prompt(tok, text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True,
    )


def _label_scores(text: str) -> list[float]:
    """Length-normalised log P(label | prompt) for each vector, via the active
    backend. Same scoring for both so accuracy/calibration are comparable."""
    if LLM_BACKEND == "mlx":
        import mlx.core as mx
        model, tok = get_mlx()

        def enc(s, special=True):
            try:
                return tok.encode(s, add_special_tokens=special)
            except TypeError:
                return tok._tokenizer.encode(s, add_special_tokens=special)

        prompt_ids = enc(_chat_prompt(tok, text))
        scores = []
        for label in VECTORS:
            lab_ids = enc(label, special=False)
            ids = prompt_ids + lab_ids
            logits = model(mx.array([ids]))[0]
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            total = sum(lp[len(prompt_ids) + j - 1, tokid].item()
                        for j, tokid in enumerate(lab_ids))
            scores.append(total / len(lab_ids))
        return scores

    # transformers backend
    model, tok = get_llm()
    prompt_ids = tok(_chat_prompt(tok, text), return_tensors="pt").input_ids.to(_device())
    scores = []
    with torch.no_grad():
        for label in VECTORS:
            lab_ids = tok(label, add_special_tokens=False, return_tensors="pt").input_ids.to(_device())
            ids = torch.cat([prompt_ids, lab_ids], dim=1)
            logits = model(ids).logits[0]
            logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
            tgt = ids[0, 1:]
            n_lab = lab_ids.shape[1]
            lp = logprobs[-n_lab:].gather(1, tgt[-n_lab:].unsqueeze(1)).sum()
            scores.append((lp / n_lab).item())
    return scores


def get_tribe():
    global _tribe
    with _lock:
        if _tribe is None:
            from tribev2.demo_utils import TribeModel

            log.info("loading TRIBE v2 ...")
            _tribe = TribeModel.from_pretrained(
                "facebook/tribev2", cache_folder=str(CACHE), device=_device()
            )
            harden_tribe(_tribe)  # num_workers=0: checkpoint resets it to ~20 -> deadlock
            log.info("TRIBE v2 ready (num_workers=%s)", _tribe.data.num_workers)
    return _tribe


def _free_tribe():
    """Release TRIBE v2 (and its 3B text-encoder) so it doesn't co-reside with the
    detector between brain-map calls."""
    global _tribe
    import gc

    with _lock:
        _tribe = None
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _schedule_free_tribe():
    """Free TRIBE now, or after an idle window if CPD_TRIBE_WARM_SECS > 0 so
    consecutive deep scans skip the reload. The timer is cancelled at the start of
    every /brainmap (see _cancel_free_timer), so it can never fire mid-scan."""
    global _free_timer
    if TRIBE_WARM_SECS <= 0:
        _free_tribe()
        return
    with _lock:
        if _free_timer is not None:
            _free_timer.cancel()
        _free_timer = threading.Timer(TRIBE_WARM_SECS, _free_tribe)
        _free_timer.daemon = True
        _free_timer.start()


def _cancel_free_timer():
    global _free_timer
    with _lock:
        if _free_timer is not None:
            _free_timer.cancel()
            _free_timer = None


def _get_baseline():
    """Load baseline.npz once and cache its arrays (avoids re-reading ~160KB of
    per-vertex mean/std on every scan). Rebuilding the baseline needs a restart."""
    global _baseline
    with _lock:
        if _baseline is None:
            p = HERE / "baseline.npz"
            if not p.exists():
                return None
            z = np.load(p)
            _baseline = {k: z[k] for k in z.files}
    return _baseline


def _swap_pressure() -> float:
    """Fraction of swap in use (0..1), 0 if unknown. Loading TRIBE (~7GB) when swap
    is already near-full sends the machine into disk thrash, so /brainmap bails."""
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True, timeout=2).stdout
        total = float(re.search(r"total = ([\d.]+)M", out).group(1))
        used = float(re.search(r"used = ([\d.]+)M", out).group(1))
        # Require an absolute floor so a small-but-full swap on a RAM-rich machine
        # doesn't false-trip.
        return (used / total) if (total and used > 10_000) else 0.0
    except Exception:
        return 0.0


def get_atlas():
    """Destrieux parcellation on fsaverage5 -> matches TRIBE's 20484 vertices."""
    global _atlas
    with _lock:
        if _atlas is None:
            from nilearn import datasets

            a = datasets.fetch_atlas_surf_destrieux()
            labels = [l.decode() if isinstance(l, bytes) else l for l in a["labels"]]
            annot = np.concatenate(
                [np.asarray(a["map_left"]), np.asarray(a["map_right"])]
            )
            _atlas = (labels, annot)
    return _atlas


@app.get("/health")
def health():
    return {"ok": True, "device": _device()}


@app.post("/classify")
def classify(inp: TextIn):
    """DETECTION. Fast (~2-3s). The LLM is the right tool here (A5).

    We do NOT let the model *write* a label and a confidence number. Free-form
    generation collapsed dopamine-bait and social-proof into "false-urgency", and
    emitted a constant "80" for every verdict -- a tic, not a probability.

    Instead we score the log-likelihood of each candidate label under the model
    and softmax over them. That gives the argmax (better accuracy) AND a genuinely
    calibrated confidence, which is what the project's "confidence is mandatory"
    rule actually demands.
    """
    key = inp.text.strip()
    if key in _classify_cache:                 # memoize exact-text repeats
        return {**_classify_cache[key], "cached": True}

    import math
    scores = _detector.submit(_label_scores, inp.text).result()  # pinned thread (MLX)
    m = max(scores)
    exps = [math.exp((s - m) / CONFIDENCE_TEMP) for s in scores]  # T<1 => assertive
    z = sum(exps)
    probs = [e / z for e in exps]

    best = max(range(len(VECTORS)), key=lambda i: probs[i])
    ranked = sorted(
        [{"vector": v, "p": round(p, 4)} for v, p in zip(VECTORS, probs)],
        key=lambda d: -d["p"],
    )
    result = {
        "vector": VECTORS[best],
        "confidence": int(round(probs[best] * 100)),
        "rationale": DEFINITIONS[VECTORS[best]],
        "alternatives": ranked[1:4],
        "source": f"llama-3.2-3b-instruct ({LLM_BACKEND}, label log-prob scoring)",
    }
    if len(_classify_cache) < 512:
        _classify_cache[key] = result
    return result


# Curated VALUE + LANGUAGE-EVALUATION ROIs. Ranking ALL regions of a single
# sentence's map surfaces low-level acoustic cortex (auditory + sensorimotor
# articulation) driven by the TTS step, not the content -- experiment A7 showed
# sensorimotor separates emotional/neutral at d~0.94, LARGER than the semantic
# signal. But A7 also showed these curated semantic ROIs separate emotional from
# neutral at d=0.95, p=0.02, matching A2 and published fMRI on emotional-word
# reading (OFC + inferior frontal). So we report THESE, honestly, rather than the
# acoustic-dominated raw ranking.
SEMANTIC_ROIS = [
    "G_front_inf-Orbital", "G_front_inf-Triangul", "G_front_inf-Opercular",
    "S_orbital-H_Shaped", "S_orbital_med-olfact", "S_orbital_lateral",
    "G_rectus", "G_subcallosal", "G_front_middle", "G_front_sup", "G_orbital",
]

# The impact profile groups the curated ROIs into interpretable cortical SYSTEMS.
# This is the visceral story the brain map exists to tell: how strongly the
# content recruits each system above neutral text. (Detection is still the LLM's
# job -- this is the "where it lands" visualisation.)
CORTICAL_SYSTEMS = {
    "Value / evaluation (orbitofrontal)": [
        "G_front_inf-Orbital", "S_orbital-H_Shaped", "S_orbital_med-olfact",
        "S_orbital_lateral", "G_orbital", "G_rectus", "G_subcallosal",
    ],
    "Language processing (inferior frontal)": [
        "G_front_inf-Triangul", "G_front_inf-Opercular",
    ],
    "Executive / cognitive-load (dlPFC)": [
        "G_front_middle", "G_front_sup",
    ],
}


def _level(z: float) -> str:
    if z >= 0.5:
        return "high"
    if z >= 0.15:
        return "elevated"
    if z <= -0.15:
        return "below baseline"
    return "low"


@app.post("/brainmap")
def brainmap(inp: TextIn):
    """INTERPRETATION. Slow. Returns MEASURED cortical regions.

    Two corrections that make this honest:

    1. BASELINE. Raw activation is dominated by "there is speech" (auditory
       cortex). We z-score each vertex against a reference corpus (baseline.npz)
       so only CONTENT-driven deviation survives. Without this the answer is the
       same for every sentence.

    2. NO ASSERTED MAPPING. We do not claim vector->region. A3 showed
       fear/outrage/reward all load on the SAME fronto-orbital regions
       (r=0.83-0.90), so "fear targets the amygdala" would be fiction -- and this
       model cannot see the amygdala at all (cortex-only).
    """
    # Exact-text memoization: a repeated scan (e.g. the same demo sentence) returns
    # instantly with NO model load — so it works even under memory pressure.
    key = inp.text.strip()
    if key in _brainmap_cache:
        return {**_brainmap_cache[key], "cached": True}

    pressure = _swap_pressure()
    if pressure > 0.90:
        return {"error": f"Low memory (swap {int(pressure * 100)}% full) — brain map "
                         "skipped to avoid disk thrashing. Reboot to reclaim swap."}

    _cancel_free_timer()   # a warm-window timer must not free TRIBE mid-scan
    tribe = get_tribe()
    CACHE.mkdir(exist_ok=True)
    txt = CACHE / "live_input.txt"
    txt.write_text(inp.text)

    try:
        # Default "text" mode injects synthetic word events (no TTS/WhisperX); the
        # baseline is built the same way. See tribe_events.py / experiment I1.
        events = build_events(tribe, inp.text, str(txt))
        preds, _ = tribe.predict(events=events)
        vertex_mean = np.asarray(preds).mean(axis=0)   # (20484,)
    finally:
        # TRIBE v2 carries a 3B text-encoder + audio encoders (~7GB). Free it after
        # each call (or after the warm window) so the resident footprint stays just
        # the 3B detector, keeping the interactive /classify fast on a 24GB machine.
        _schedule_free_tribe()

    b = _get_baseline()
    if b is None:
        return {"error": "baseline.npz missing -- run build_baseline.py first"}
    # The baseline is only valid for the mode it was built in (text vs audio give
    # different absolute activations). Refuse a mismatch rather than z-score garbage.
    base_mode = str(b["mode"]) if "mode" in b else "audio"
    if base_mode != BRAINMAP_MODE:
        return {"error": f"baseline.npz was built in '{base_mode}' mode but the server "
                         f"is running in '{BRAINMAP_MODE}' mode. Rebuild with "
                         f"CPD_BRAINMAP_MODE={BRAINMAP_MODE} python build_baseline.py."}
    zvert = (vertex_mean - b["mean"]) / b["std"]   # deviation from neutral baseline

    labels, annot = get_atlas()
    idx = {name: i for i, name in enumerate(labels)}

    def region_z(name: str):
        i = idx.get(name)
        if i is None:
            return None
        mask = annot == i
        return float(zvert[mask].mean()) if mask.sum() else None

    regions = []
    for name in SEMANTIC_ROIS:
        z = region_z(name)
        if z is not None:
            regions.append({"region": name, "activation_z": round(z, 3)})
    regions.sort(key=lambda r: -r["activation_z"])

    # The impact profile: mean engagement per cortical system, with a level.
    profile = []
    for system, rois in CORTICAL_SYSTEMS.items():
        zs = [region_z(n) for n in rois]
        zs = [z for z in zs if z is not None]
        if not zs:
            continue
        z = float(np.mean(zs))
        profile.append({"system": system, "z": round(z, 3), "level": _level(z)})
    profile.sort(key=lambda p: -p["z"])

    engagement = float(np.mean([r["activation_z"] for r in regions])) if regions else 0.0

    result = {
        "impact_profile": profile,
        "value_cortex_engagement_z": round(engagement, 3),
        "headline_regions": regions[:4],
        "baseline_n": int(b["n"]),
        "n_vertices": int(vertex_mean.shape[0]),
        "mode": BRAINMAP_MODE,
        "source": f"TRIBE v2 (local, {BRAINMAP_MODE} path) -> cortical impact profile, "
                  "z-scored vs neutral baseline",
        "interpretation": "how strongly the content recruits each cortical system "
                          "above neutral text. Validated d=0.95 emotional-vs-neutral "
                          "(A7), consistent with published fMRI on emotional-word reading.",
        "caveat": "predicted cortical BOLD, NOT a detector (the LLM is strictly "
                  "better). NOT subcortical: cannot see amygdala/accumbens. We report "
                  "only semantic systems (emotional text also drives TTS-acoustic "
                  "sensorimotor cortex, which we exclude).",
    }
    if len(_brainmap_cache) < 256:
        _brainmap_cache[key] = result
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
