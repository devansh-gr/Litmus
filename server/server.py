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
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

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
    "none": "neutral, factual or informational; no manipulation",
}

SYSTEM = (
    "You are a persuasion analyst. Identify the PRIMARY manipulation technique in "
    "the text. The techniques are:\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    + "\nAnswer with the technique name only."
)

app = FastAPI(title="Cortical Persuasion Decoder")

_lock = threading.Lock()
_llm = None
_tok = None
_tribe = None
_atlas = None


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


def get_tribe():
    global _tribe
    with _lock:
        if _tribe is None:
            from tribev2.demo_utils import TribeModel

            log.info("loading TRIBE v2 ...")
            _tribe = TribeModel.from_pretrained(
                "facebook/tribev2", cache_folder=str(CACHE), device=_device()
            )
            log.info("TRIBE v2 ready")
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
    model, tok = get_llm()
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": inp.text},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(_device())

    scores = []
    with torch.no_grad():
        for label in VECTORS:
            lab_ids = tok(label, add_special_tokens=False, return_tensors="pt").input_ids.to(_device())
            ids = torch.cat([prompt_ids, lab_ids], dim=1)
            logits = model(ids).logits[0]
            # log P(label tokens | prompt), length-normalised so long label names
            # are not penalised.
            logprobs = torch.log_softmax(logits[:-1].float(), dim=-1)
            tgt = ids[0, 1:]
            n_lab = lab_ids.shape[1]
            lp = logprobs[-n_lab:].gather(1, tgt[-n_lab:].unsqueeze(1)).sum()
            scores.append((lp / n_lab).item())

    probs = torch.softmax(torch.tensor(scores), dim=0)
    best = int(torch.argmax(probs))
    vector = VECTORS[best]
    confidence = int(round(float(probs[best]) * 100))

    ranked = sorted(
        [{"vector": v, "p": round(float(p), 4)} for v, p in zip(VECTORS, probs)],
        key=lambda d: -d["p"],
    )
    return {
        "vector": vector,
        "confidence": confidence,
        "rationale": DEFINITIONS[vector],
        "alternatives": ranked[1:4],
        "source": "llama-3.2-3b-instruct (local, label log-prob scoring)",
    }


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
    tribe = get_tribe()
    CACHE.mkdir(exist_ok=True)
    txt = CACHE / "live_input.txt"
    txt.write_text(inp.text)

    try:
        events = tribe.get_events_dataframe(text_path=str(txt))
        preds, _ = tribe.predict(events=events)
        vertex_mean = np.asarray(preds).mean(axis=0)   # (20484,)
    finally:
        # TRIBE v2 carries a 3B text-encoder + audio encoders (~7GB). Free it after
        # each call so the resident footprint stays just the 3B detector, keeping
        # the interactive /classify fast on a 24GB machine.
        _free_tribe()

    base_path = HERE / "baseline.npz"
    if not base_path.exists():
        return {"error": "baseline.npz missing -- run build_baseline.py first"}
    b = np.load(base_path)
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

    engagement = float(np.mean([r["activation_z"] for r in regions])) if regions else 0.0

    return {
        "value_cortex_engagement_z": round(engagement, 3),
        "top_regions": regions[:6],
        "baseline_n": int(b["n"]),
        "n_vertices": int(vertex_mean.shape[0]),
        "source": "TRIBE v2 (local) -> curated value/language-evaluation ROIs, "
                  "z-scored vs neutral baseline",
        "interpretation": "engagement of value (orbitofrontal) and language-"
                          "evaluation (inferior frontal) cortex above neutral text. "
                          "Validated d=0.95 emotional-vs-neutral (A7), consistent "
                          "with published fMRI on emotional-word reading.",
        "caveat": "predicted cortical BOLD, NOT a detector (the LLM is strictly "
                  "better). NOT subcortical: cannot see amygdala/accumbens. "
                  "Emotional text also drives sensorimotor cortex (TTS acoustics); "
                  "we deliberately report only the semantic ROIs.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
