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
LLM_NAME = "meta-llama/Llama-3.2-3B-Instruct"

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


# Low-level AUDITORY cortex. TRIBE v2's text path synthesises speech, so these
# regions light up for ANY sentence simply because sound exists -- an artifact of
# our TTS step, not of the content. The user READS the text, and the modality
# literature says low-level sensory representations do NOT transfer between
# listening and reading (semantic regions like IFG/MTG do). Reporting these as
# findings would be a lie, so they are separated out.
AUDITORY_ARTIFACT = {
    "S_temporal_transverse",
    "G_temp_sup-G_T_transv",
    "G_temp_sup-Plan_tempo",
    "G_temp_sup-Plan_polar",
    "G_temp_sup-Lateral",
    "Lat_Fis-post",
}


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

    events = tribe.get_events_dataframe(text_path=str(txt))
    preds, _ = tribe.predict(events=events)
    vertex_mean = np.asarray(preds).mean(axis=0)   # (20484,)

    base_path = HERE / "baseline.npz"
    if not base_path.exists():
        return {"error": "baseline.npz missing -- run build_baseline.py first"}
    b = np.load(base_path)
    zvert = (vertex_mean - b["mean"]) / b["std"]   # content-driven deviation

    labels, annot = get_atlas()
    content, artifacts = [], []
    for idx, name in enumerate(labels):
        if name == "Unknown":
            continue
        mask = annot == idx
        if not mask.sum():
            continue
        entry = {"region": name, "activation_z": round(float(zvert[mask].mean()), 3)}
        (artifacts if name in AUDITORY_ARTIFACT else content).append(entry)

    content.sort(key=lambda r: -abs(r["activation_z"]))
    artifacts.sort(key=lambda r: -abs(r["activation_z"]))

    return {
        "top_regions": content[:6],
        "excluded_auditory_artifacts": artifacts[:3],
        "baseline_n": int(b["n"]),
        "n_vertices": int(vertex_mean.shape[0]),
        "source": "TRIBE v2 (local) -> Destrieux/fsaverage5, z-scored vs baseline corpus",
        "caveat": "predicted cortical BOLD. NOT a detector (the text model is "
                  "strictly better at that). NOT subcortical: this model cannot "
                  "see the amygdala or nucleus accumbens.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
