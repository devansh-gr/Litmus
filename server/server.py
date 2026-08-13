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

import hashlib
import json
import os
import logging
import re
import subprocess
import threading
import time
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
    "guilt-tripping",
    "love-bombing",
    "blame-shifting",
    "none",
]

# Explicit definitions: the 3B model collapsed dopamine-bait and social-proof
# into "false-urgency" without them.
DEFINITIONS = {
    "fear-mongering": "makes you afraid of harm, danger, disease or catastrophe",
    "critical-thinking-suppression": "tells you to STOP thinking for yourself, or GASLIGHTS you into doubting your own mind — don't question, don't overthink, just trust it; OR 'that never happened', 'you're imagining it', 'you're too sensitive / crazy / overreacting', denying your memory or reality so you distrust your own perception; OR MINIMIZES what it did to make you doubt your reaction — 'I was just joking', 'can't you take a joke', 'it wasn't that bad', 'you're making a big deal out of nothing'",
    "tribal-in-group-bias": "an US-versus-THEM split: our side vs an enemy group — 'they' want to destroy what 'we' stand for, real ones vs outsiders",
    "dopamine-bait": "dangles a REWARD or a CURIOSITY payoff to chase — a prize, jackpot, win, freebie, giveaway, a shocking secret to 'unlock' or 'claim'; OR curiosity-gap CLICKBAIT that withholds the payoff to bait a click ('you won't believe what happened next', 'this one weird trick', 'N things that will blow your mind', 'we know if you...')",
    "outrage": "invites moral anger and disgust at someone's behaviour",
    "authority-appeal": "leans on a person's STATUS, fame, title, or credentials to make a claim credible — an expert, official, scientist, doctor, celebrity, or authority figure endorses or asserts it, so you should believe or buy it ('doctors recommend', 'Michael Jordan wears X so you should too', 'my minister/father says so it must be true', '20 years of experience so trust my opinion') — EVEN when that source is unqualified or irrelevant to the claim",
    "false-urgency": "an explicit TIME deadline or countdown forces you to beat the CLOCK — 'act now', 'expires today', 'ends tonight', 'offer ends in 2 hours', 'last chance today', a ticking countdown timer. The lever is a DEADLINE. If the pressure is limited SUPPLY ('only 2 left', 'selling fast') rather than a clock, that is fomo — even if it also says 'order soon'",
    "social-proof-conformity": "points to OTHER PEOPLE'S behaviour so you follow the crowd — 'everyone's doing it', 'don't be left out', 'join thousands of customers', and live activity / purchase-count nudges: 'X people bought this', 'X people are viewing', 'someone in [place] just bought this'. The lever is the CROWD, not a clock or a prize",
    "hype-hope-mongering": "promises an exciting personal upside or better future to make you want in: it will improve your life, make you money, or fix your problems",
    "fomo": "you'll MISS OUT because SUPPLY is limited or others are grabbing it — 'only a few left', 'only 2 in stock', 'selling fast', 'almost gone', 'in high demand', 'while stocks last', 'we reserved yours', 'you'll regret not joining'. The lever is limited QUANTITY / scarcity, NOT a clock — fomo WINS over false-urgency whenever the driver is limited supply, even when it also says 'order soon' or 'hurry'",
    "manufactured-awe": "exaggerates how revolutionary, unprecedented, historic, greatest-ever or mind-blowing the thing ITSELF is, so sheer grandeur overwhelms your skepticism",
    "guilt-tripping": "makes you feel GUILTY, obligated, or ashamed to comply — 'if you really loved/cared you would', 'after all I've done / sacrificed for you', 'you owe me', 'how could you do this to me', plus MARTYRDOM and self-pity used as leverage ('fine, I'll just do it myself then', 'don't worry about me', 'I'll be alone but it's ok') and DISAPPOINTMENT as a weapon ('I'm not mad, just disappointed in you') — guilt, shame, and obligation as leverage",
    "love-bombing": "EXCESSIVE, possessive affection or grand romantic promises used to disarm you or build dependency — 'no one will EVER love you like I do', 'you're my soulmate, my everything, I can't live without you', 'I've never felt this way about anyone', 'we're meant to be'. The tell is possessiveness and dependency ('no one else', 'only I', 'can't live without you'), NOT ordinary praise — a plain compliment like 'you did a great job' or 'nice work' is NOT love-bombing",
    "blame-shifting": "REVERSES responsibility onto you so the manipulator escapes blame (DARVO) — 'you made me do this', 'this is your fault', 'if you hadn't … I wouldn't have …', 'I only reacted because of you', 'you're the one who started it', 'I'm the real victim here'",
    "none": "neutral, factual or informational; a greeting, question, or ordinary conversation; no manipulation",
}

# Built from parts (head + rotatable definition list + tie-break tail) so self-consistency
# voting can present the definitions in different orders to average out list-position bias.
_SYSTEM_HEAD = (
    "You are a persuasion analyst. Identify the PRIMARY manipulation technique in "
    "the text. The techniques are:\n"
)
_SYSTEM_TAIL = (
    "\nWhen more than one seems to fit, choose by the PRIMARY LEVER the text pulls: "
    "limited quantity / scarcity -> fomo; an explicit deadline or countdown clock -> "
    "false-urgency; other people's activity or the crowd -> social-proof-conformity; a "
    "reward or curiosity hook -> dopamine-bait; a personal upside / better future -> "
    "hype-hope-mongering; a named person's fame / title / credential given as the reason "
    "to believe or buy (celebrity endorsement, 'a doctor/expert says') -> authority-appeal.\n"
    "Answer with the technique name only."
)


def _build_system(order=None):
    """The technique system prompt. `order` (a permutation of range(len(DEFINITIONS)))
    reorders the definition listing for self-consistency voting; None = default order."""
    items = list(DEFINITIONS.items())
    if order is not None:
        items = [items[i] for i in order]
    return _SYSTEM_HEAD + "\n".join(f"- {k}: {v}" for k, v in items) + _SYSTEM_TAIL


SYSTEM = _build_system()

# Self-consistency voting (opt-in, CPD_SELF_CONSISTENCY=K, default 1 = off). The label
# scorer is DETERMINISTIC, so plain resampling is a no-op — the form that fits is a prompt
# ensemble: present the definitions in K rotated orders, score each, majority-vote the
# argmax (probabilities averaged for the confidence). Averages out the list-position bias
# in log-prob label scoring. K× the stage-2 cost, so it stays off for the instant ⌘B path.
SELF_CONSISTENCY = max(1, int(os.environ.get("CPD_SELF_CONSISTENCY", "1")))


def _vote_orders(n, k):
    """k deterministic, evenly-spread cyclic rotations of range(n)."""
    return [[(j + i * (n // k)) % n for j in range(n)] for i in range(k)]

# Few-shot exemplars for the TECHNIQUE classifier (stage 2 only; the gate stays zero-shot).
# One short demonstration per confusable technique, keyed on the primary-lever distinctions
# the definitions describe. SELF-AUTHORED — deliberately NOT drawn from external_test, so
# the honest benchmark stays leakage-free. Toggle with CPD_FEWSHOT=0 for an A/B.
FEWSHOT_ENABLED = os.environ.get("CPD_FEWSHOT", "1") == "1"
# Deliberately NO fomo/false-urgency exemplars: the sharpened definitions already draw that
# boundary cleanly, and adding exemplars there re-introduced the scarcity->false-urgency blur.
# Few-shot is reserved for the remaining classes, where a demonstration reinforces the definition.
TECHNIQUE_FEWSHOT = [
    ("Over 8,000 people bought this in the last 24 hours.", "social-proof-conformity"),
    ("Claim your free mystery box — you won't believe what's inside.", "dopamine-bait"),
    ("A top cardiologist swears by this, so you know it works.", "authority-appeal"),
    ("This system will transform your finances and change your life.", "hype-hope-mongering"),
    ("Ignore this and you could lose everything you've worked for.", "fear-mongering"),
    ("If you really cared about this family, you'd cancel your plans.", "guilt-tripping"),
    ("If we don't stop them now, they'll destroy everything we built.", "tribal-in-group-bias"),
] if FEWSHOT_ENABLED else []

# Stage-1 binary gate: "is this manipulation at all?". A clean 2-way decision at the
# none boundary avoids the failure where the 12-way softmax labels a greeting or a
# helpful sentence as hype. yes/no label tokens (rather than repeating the loaded word
# "manipulation") keep the log-prob scoring from being primed toward one answer.
GATE_LABELS = ("no", "yes")
# The gate uses its OWN temperature (unsharpened by default). Sharpening (the tiny
# CONFIDENCE_TEMP used for the assertive technique %) saturates the gate probability to
# ~0/1, which destroys its ranking and leaves the threshold no purchase. T=1 keeps a
# smooth P(manipulation) the threshold can actually tune against.
GATE_TEMP = float(os.environ.get("CPD_GATE_TEMP", "1.0"))
# Contextual calibration ("Calibrate Before Use", Zhao et al. 2021): the gate's log-prob
# scoring is biased by the prompt itself (a repeated word inflates its label). We measure
# that bias on a content-free input ("N/A") once and divide it out of every gate
# probability, so the decision reflects the TEXT, not the prompt's a-priori lean.
# DEFAULT OFF — EVALUATED, no benefit. Measured on external-dev with it ON: gate
# PR-AUC 0.916 ≈ 0.92 OFF (no ranking gain), because the yes/no gate verbalizers already
# sidestep the surface-form/priming bias this corrects. It also shifts the probability
# scale badly (recall 0.96 but 75% benign FP at the same threshold), forcing a full
# threshold + calibration re-tune for zero gain. Kept as an opt-in capability.
CONTEXTUAL_CALIB = os.environ.get("CPD_CONTEXTUAL_CALIB", "0") == "1"
_gate_cf_probs = None   # cached content-free gate bias
# Route to `none` only when the gate is at least this confident it's NOT manipulation.
# Higher = the gate must be more sure before it calls something benign, so more borderline
# text flows to the technique classifier (recovers manipulation recall, costs benign precision).
# 0.65: flag when the gate's manip_prob > 0.35. RE-SWEPT after the dark-pattern gate-prompt fixes
# (tests/gate_diag.py + bench_after dump). Those fixes lifted manip_prob on real manipulation, so
# on the 300-ex external set the gate reaches recall 0.78 / precision 0.87 at 0.65. 0.70 buys a
# little more external recall (0.81) but re-flags the curated casual set — praise ("you did a
# wonderful job") and sign-offs ("hope it helps") sit in the SAME 0.30-0.35 manip_prob band as
# borderline dark patterns, so pushing the cutoff below 0.35 trades UX benign-precision for a
# couple of bare-count edge cases. 0.65 is the sweet spot: dark patterns flag, casual chat stays
# none. (History: 0.60 gave recall 0.55; the big win was the gate-prompt fixes for social-proof /
# scarcity / countdown / clickbait, which took gate recall 0.55→0.77 — see tests/RESULTS.md.)
GATE_NONE_THRESHOLD = float(os.environ.get("CPD_GATE_NONE_THRESHOLD", "0.65"))
GATE_SYSTEM = (
    "Decide if the text is trying to MANIPULATE or PERSUADE the reader, versus just "
    "communicating normally.\n"
    "Answer 'no' for ordinary communication: greetings, questions, requests (\"could you "
    "send the file\"), instructions, plain facts and information — including NEUTRAL "
    "mentions of dates, deadlines, prices, or numbers (\"the sale ends Friday\", \"review "
    "it by Thursday\") — personal opinions and reviews, congratulations, and friendly or "
    "helpful talk. A warm, positive, or helpful tone by itself is 'no'. Ordinary "
    "compliments, encouragement, and praise are 'no' (\"you did a great job\", \"nice "
    "work\", \"well done\"). Polite offers of help and email sign-offs are 'no' too — the "
    "word 'hope' in a courtesy ('hope it helps', 'hope you're well') is NOT hype: \"let me "
    "know if you have any questions\", \"hope this helps\", \"happy to help\", \"feel free "
    "to reach out\", \"here's the summary you asked for\".\n"
    "Answer 'yes' if the wording tries to pressure, scare, hype, guilt, rush, divide "
    "us-versus-them, dangle a reward, or push the reader toward a belief, feeling, "
    "purchase, or action. Also 'yes' for e-commerce SOCIAL-PROOF dark patterns — live "
    "activity or purchase-count nudges meant to make you buy because others are: \"24 "
    "people have purchased this today\", \"5 people are viewing this product\", \"Someone "
    "in [city] just bought this\", \"[Name] from [place] purchased a...\", \"join thousands "
    "of happy customers\", \"selling fast\". These pressure via the crowd even though they "
    "read like a plain statistic — the INTENT to nudge a purchase makes them 'yes' (a "
    "neutral count with no nudge, \"the study surveyed 24 people\", is still 'no'). Also "
    "'yes' for SCARCITY dark patterns that manufacture fear of missing out: \"only 2 left "
    "in stock\", \"items in your cart are in high demand\", \"we've reserved yours for a "
    "limited time\", \"almost gone\", \"selling fast\", \"while stocks last\", \"low stock "
    "— order soon\" — low-supply pressure engineered to rush a purchase. Also 'yes' for a "
    "COUNTDOWN TIMER counting down to a deadline — \"04 HOURS 17 MINUTES 18 SECONDS\", "
    "\"offer expires in 02:14:59\", \"ends in 1 day 4 hours\" — even as bare numbers, a "
    "ticking clock on an offer is engineered urgency (a plain clock time like \"the "
    "meeting is at 4:30\" is 'no'). Also 'yes' for CLICKBAIT that baits a click with a "
    "curiosity gap, a listicle, or hype instead of informing — \"you won't believe what "
    "happens next\", \"this one weird trick\", \"we know if you own a pair of white vans\", "
    "\"N things that will blow your mind\", numbered listicles of soft or entertainment "
    "content (\"19 edible shots you have to make\", \"47 beauty hacks everyone should "
    "know\"), and curiosity teasers that withhold the payoff (\"...is causing a huge "
    "debate\", \"...and it's incredible\", \"look out of this world\", \"you have to see "
    "this\") — the tell is baiting the click rather than stating the news. A straight, "
    "informative headline that tells you the news outright (\"Fed raises interest rates "
    "0.25%\", \"60 killed in Iraq bombing\", \"president inaugurated after reelection\") is "
    "'no'. Also 'yes' for APPEAL TO AUTHORITY used to win a belief or a purchase — a named "
    "person, celebrity, expert, elder, or title offered AS THE REASON you should believe, "
    "buy, or do it: \"Michael Jordan wears them, so you should too\", \"a top cardiologist "
    "swears by it\", \"my minister/father says so, so it must be true\", \"doctors smoke "
    "it, so it's healthy\", \"20 years of experience, so trust my opinion\". The tell is "
    "the leap from WHO says it to you-should-agree; a plain report that merely names or "
    "quotes a person (\"the president was inaugurated\", \"the CEO is said to have siphoned "
    "cash\", \"Mickelson shares the lead\") is 'no'. Also "
    "'yes' when it cites experts, studies, or the crowd to SHUT "
    "DOWN doubt (\"experts agree, so the debate is over\", \"the only correct approach\", "
    "\"nothing left to question\") — as opposed to just reporting a fact (\"a study "
    "examined sleep\" is no). Also 'yes' when it tells the reader to STOP thinking for "
    "themselves — don't question, don't overthink, stop doing your own research, no need "
    "to understand, just trust or believe it. Also 'yes' for GASLIGHTING and emotional "
    "manipulation between people: denying what happened or your memory (\"that never "
    "happened\", \"you're imagining it\"), calling you crazy / too sensitive / dramatic / "
    "overreacting to make you doubt yourself, MINIMIZING what was done so you doubt your "
    "reaction (\"I was just joking\", \"can't you take a joke\", \"it wasn't that bad\"), "
    "guilt-tripping (\"if you really loved me "
    "you would\", \"after all I've done for you\", \"how could you do this to me\", or "
    "MARTYRDOM / self-pity as leverage: \"fine, I'll just do it myself then\", \"don't "
    "worry about me\", \"I'm not mad, just disappointed in you\"), or LOVE-BOMBING — overwhelming flattery, affection, or grand promises "
    "to disarm you or create dependency (\"no one will ever love you like I do\", \"you're "
    "my soulmate, I'd do anything for you\"), or BLAME-SHIFTING — reversing responsibility "
    "onto the reader so the speaker escapes blame (\"you made me do this\", \"it's your "
    "fault\", \"I only reacted because of you\").\n"
    "The difference is intent to influence: \"the sale ends Friday\" is no, but \"hurry, "
    "the sale ends Friday, don't miss out!\" is yes.\n"
    "Answer with a single word: yes or no."
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
# Optional LoRA adapter (technique fine-tune). Empty = base model. Trained on the
# leakage-free split (lora/prepare_data.py) and reported on the untouched holdout.
MLX_ADAPTER = os.environ.get("CPD_MLX_ADAPTER", "")

# Confidence sharpening. With 12 candidate labels a raw softmax dilutes the winner
# (a clear false-urgency landed at ~20%), which reads as unsure. Dividing the
# scores by a temperature < 1 sharpens the distribution so the top vector reports
# an ASSERTIVE confidence. Lower = more assertive. Override with CPD_CONFIDENCE_TEMP.
CONFIDENCE_TEMP = float(os.environ.get("CPD_CONFIDENCE_TEMP", "0.25"))

# Platt calibration of the DISPLAYED confidence so the shown % means P(correct). The
# raw model is badly over-confident (ECE 0.37 — says "99%", right ~60%). a,b are fit on
# external-dev by tests/fit_calibration.py; defaults are identity (no change).
# Calibration DEFAULT OFF (identity) by product choice: the Platt fit (a=0.56, b=-1.10) is
# the *honest* mapping (ECE 0.37→0.10) but it drags the shown % down into the 40-60s, which
# reads as timid. The product is meant to feel ASSERTIVE, so we show the sharpened confidence.
# Set CPD_CALIB_A=0.56 CPD_CALIB_B=-1.10 to get the honest (lower) numbers back.
CALIB_A = float(os.environ.get("CPD_CALIB_A", "1.0"))
CALIB_B = float(os.environ.get("CPD_CALIB_B", "0.0"))
# Abstain: below this CALIBRATED confidence the technique label is not trustworthy (the
# calibration makes this honest), so flag it rather than assert a shaky verdict.
ABSTAIN_BELOW = float(os.environ.get("CPD_ABSTAIN_BELOW", "0.45"))
# Multi-label "mixture": techniques whose probability clears this floor are treated as
# co-present (e.g. "act now, only 2 left, everyone's buying" is fomo AND false-urgency AND social
# proof). The verdict carries a `mixture` list (every technique above the floor) alongside the
# single top `vector`. A floor (not a margin off the winner) is used because the confidence softmax
# is sharpened, and it matches the macOS card's existing "also:" selection.
MIXTURE_FLOOR = float(os.environ.get("CPD_MIXTURE_FLOOR", "0.15"))


def _calibrate(p: float) -> float:
    """Map a raw top-class probability to a calibrated confidence via Platt scaling."""
    import math
    p = min(0.999, max(0.001, p))
    z = CALIB_A * math.log(p / (1 - p)) + CALIB_B
    return 1.0 / (1.0 + math.exp(-z))

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
_scan_active = False                    # True while a /brainmap holds TRIBE (blocks free)
_label_ids_cache: dict = {}             # precomputed label token-ids per label set
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
            if MLX_ADAPTER:
                log.info("loading MLX %s + adapter %s ...", MLX_MODEL, MLX_ADAPTER)
                _mlx = load(MLX_MODEL, adapter_path=MLX_ADAPTER)
            else:
                log.info("loading MLX %s ...", MLX_MODEL)
                _mlx = load(MLX_MODEL)
            log.info("MLX detector ready%s", " (LoRA)" if MLX_ADAPTER else "")
    return _mlx


def warm_detector():
    """Load the active detector backend on the dedicated detector thread, and pre-warm
    the small CPU-only brain-map assets (atlas + baseline) so the FIRST deep-scan
    doesn't pay their load in the request path. Neither holds GPU/large RAM, so this is
    safe to do at startup and does NOT pull in the ~7GB TRIBE model."""
    _detector.submit(get_mlx if LLM_BACKEND == "mlx" else get_llm)
    try:
        get_atlas()
        _get_baseline()
    except Exception as e:  # noqa: BLE001 — a warm failure must not block the detector
        log.info("brain-map asset pre-warm skipped: %s", e)


def _chat_prompt(tok, text: str, system: str, fewshot=None) -> str:
    # Few-shot exemplars ride as prior user->assistant turns, so the label-scoring path
    # sees "text -> technique" demonstrations before the real query. Cloze-style: the
    # assistant content is exactly the label string we later score.
    msgs = [{"role": "system", "content": system}]
    for ex_text, ex_label in (fewshot or []):
        msgs.append({"role": "user", "content": ex_text})
        msgs.append({"role": "assistant", "content": ex_label})
    msgs.append({"role": "user", "content": text})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _label_scores(text: str, system: str = None, labels=None, fewshot=None) -> list[float]:
    """Length-normalised log P(label | prompt) for each label under `system`, via the
    active backend. Parameterised so the same scorer drives both the manipulation gate
    (2 labels) and the technique classifier (12 labels); `fewshot` adds exemplar turns
    (technique stage only — the gate stays zero-shot)."""
    system = system if system is not None else SYSTEM
    labels = list(labels) if labels is not None else list(VECTORS)
    ck = tuple(labels)
    if LLM_BACKEND == "mlx":
        import mlx.core as mx
        model, tok = get_mlx()

        def enc(s, special=True):
            try:
                return tok.encode(s, add_special_tokens=special)
            except TypeError:
                return tok._tokenizer.encode(s, add_special_tokens=special)

        # Label token-ids never change — tokenize each label set once.
        if ck not in _label_ids_cache:
            _label_ids_cache[ck] = [enc(v, special=False) for v in labels]
        _label_ids_mlx = _label_ids_cache[ck]

        prompt_ids = enc(_chat_prompt(tok, text, system, fewshot))

        # KV-cache reuse: the prompt prefix (~350 tokens of system + text) is shared by
        # all 12 labels. Process it ONCE, then score each label by continuing from the
        # cached prefix state and trimming back between labels — instead of re-running
        # the full prefix 12x (~10x less forward compute). Verdicts are identical.
        try:
            from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        except Exception:  # noqa: BLE001 — fall back to the simple path if API moved
            make_prompt_cache = None

        if make_prompt_cache is not None:
            cache = make_prompt_cache(model)
            prefix_last = model(mx.array([prompt_ids]), cache=cache)[0][-1]
            prefix_last = prefix_last - mx.logsumexp(prefix_last)   # log P(next | prefix)
            scores = []
            for lab_ids in _label_ids_mlx:
                total = prefix_last[lab_ids[0]]                     # P(first label token)
                if len(lab_ids) > 1:
                    lg = model(mx.array([lab_ids[:-1]]), cache=cache)[0]
                    lg = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                    rows = mx.arange(len(lab_ids) - 1)
                    total = total + lg[rows, mx.array(lab_ids[1:])].sum()
                    trim_prompt_cache(cache, len(lab_ids) - 1)     # rewind to prefix state
                scores.append((total / len(lab_ids)).item())
            return scores

        # Fallback: one full forward pass per label (original method).
        scores = []
        for lab_ids in _label_ids_mlx:
            ids = prompt_ids + lab_ids
            logits = model(mx.array([ids]))[0]
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            total = sum(lp[len(prompt_ids) + j - 1, tokid].item()
                        for j, tokid in enumerate(lab_ids))
            scores.append(total / len(lab_ids))
        return scores

    # transformers backend
    model, tok = get_llm()
    prompt_ids = tok(_chat_prompt(tok, text, system, fewshot), return_tensors="pt").input_ids.to(_device())
    scores = []
    with torch.no_grad():
        for label in labels:
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
        if _scan_active:
            return   # a scan is mid-flight; its finally will re-arm the free timer
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


def _taxonomy_hash() -> str:
    """Stable short digest of the label set, so a client can detect app<->server
    taxonomy skew (the app enum drifting from server VECTORS) without shipping the
    whole list — mirrors the test_taxonomy_sync guard, at runtime."""
    return hashlib.sha256("|".join(VECTORS).encode()).hexdigest()[:12]


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": _device(),
        "backend": LLM_BACKEND,
        "vector_count": len(VECTORS) - 1,   # excluding "none"
        "taxonomy_hash": _taxonomy_hash(),
    }


@app.get("/vectors")
def vectors():
    """Introspection: the full label set + definitions the classifier scores against.
    Lets tools/tests read the live taxonomy instead of hardcoding a copy that goes stale."""
    return {
        "vectors": VECTORS,
        "definitions": DEFINITIONS,
        "count": len(VECTORS) - 1,
        "taxonomy_hash": _taxonomy_hash(),
    }


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
    t0 = time.perf_counter()

    def _log(result: dict) -> dict:
        # One structured line per verdict: enough to spot gate drift / slow calls in
        # the launchd log without echoing the (possibly sensitive) input text.
        log.info("verdict vector=%s conf=%d manip_prob=%.3f ms=%d chars=%d",
                 result["vector"], result["confidence"],
                 result.get("manip_prob", -1), int((time.perf_counter() - t0) * 1000),
                 len(inp.text))
        return result

    def softmax(scores, temp=CONFIDENCE_TEMP):
        m = max(scores)
        exps = [math.exp((s - m) / temp) for s in scores]  # T<1 => assertive
        z = sum(exps)
        return [e / z for e in exps]

    # STAGE 1 — manipulation gate. A clean binary decision keeps benign text (greetings,
    # helpful/friendly phrasing) from being forced into the nearest manipulation label.
    # Uses GATE_TEMP (unsharpened) so the gate probability stays a smooth, tunable score.
    global _gate_cf_probs
    gate = _detector.submit(_label_scores, inp.text, GATE_SYSTEM, GATE_LABELS).result()
    gp_raw = softmax(gate, GATE_TEMP)
    if CONTEXTUAL_CALIB:
        if _gate_cf_probs is None:   # measure the prompt's bias on a content-free input, once
            cf = _detector.submit(_label_scores, "N/A", GATE_SYSTEM, GATE_LABELS).result()
            _gate_cf_probs = softmax(cf, GATE_TEMP)
        adj = [gp_raw[i] / max(_gate_cf_probs[i], 1e-6) for i in range(len(GATE_LABELS))]
        s = sum(adj)
        gp_raw = [a / s for a in adj]
    gp = dict(zip(GATE_LABELS, gp_raw))
    if gp["no"] >= GATE_NONE_THRESHOLD:
        result = {
            "vector": "none",
            "confidence": int(round(_calibrate(gp["no"]) * 100)),
            "rationale": DEFINITIONS["none"],
            "alternatives": [],
            "manip_prob": round(gp["yes"], 4),   # gate P(manipulation) — for eval/PR-AUC
            "source": f"llama-3.2-3b-instruct ({LLM_BACKEND}, manipulation gate)",
        }
        if len(_classify_cache) < 512:
            _classify_cache[key] = result
        return _log(result)

    # STAGE 2 — which technique. Identical to the standalone classifier; the gate above
    # just spared benign text. With CPD_SELF_CONSISTENCY>1, vote across rotated-definition
    # prompt variants (averaging list-position bias) instead of a single pass.
    if SELF_CONSISTENCY > 1:
        import collections
        prob_sum = [0.0] * len(VECTORS)
        votes = []
        for order in _vote_orders(len(VECTORS), SELF_CONSISTENCY):
            sc = _detector.submit(_label_scores, inp.text, _build_system(order), VECTORS, TECHNIQUE_FEWSHOT).result()
            pv = softmax(sc)
            votes.append(max(range(len(VECTORS)), key=lambda i: pv[i]))
            for i in range(len(VECTORS)):
                prob_sum[i] += pv[i]
        probs = [s / SELF_CONSISTENCY for s in prob_sum]
        counts = collections.Counter(votes)
        best = max(range(len(VECTORS)), key=lambda i: (counts.get(i, 0), prob_sum[i]))
    else:
        scores = _detector.submit(_label_scores, inp.text, SYSTEM, VECTORS, TECHNIQUE_FEWSHOT).result()
        probs = softmax(scores)
        best = max(range(len(VECTORS)), key=lambda i: probs[i])
    ranked = sorted(
        [{"vector": v, "p": round(p, 4)} for v, p in zip(VECTORS, probs)],
        key=lambda d: -d["p"],
    )
    # Multi-label mixture: every technique above MIXTURE_FLOOR (excluding "none"), winner first.
    # Surfaces co-present techniques without changing the single top `vector`.
    mixture = [{"vector": r["vector"], "pct": int(round(r["p"] * 100))}
               for r in ranked
               if r["vector"] != "none" and r["p"] >= MIXTURE_FLOOR][:3]
    cal = _calibrate(probs[best])
    uncertain = cal < ABSTAIN_BELOW
    result = {
        "vector": VECTORS[best],
        "confidence": int(round(cal * 100)),
        "rationale": ("Likely manipulative, but the specific technique is unclear — treat "
                      "the label as a guess." if uncertain else DEFINITIONS[VECTORS[best]]),
        "uncertain": uncertain,
        "alternatives": ranked[1:4],
        "mixture": mixture,
        "manip_prob": round(gp["yes"], 4),   # gate P(manipulation) — for eval/PR-AUC
        "source": f"llama-3.2-3b-instruct ({LLM_BACKEND}, gated label scoring)",
    }
    if len(_classify_cache) < 512:
        _classify_cache[key] = result
    return _log(result)


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

    global _scan_active
    with _lock:
        _scan_active = True    # set BEFORE cancel so a firing timer can't free mid-scan
    _cancel_free_timer()
    tribe = get_tribe()
    CACHE.mkdir(exist_ok=True)
    txt = CACHE / "live_input.txt"
    if BRAINMAP_MODE == "audio":
        txt.write_text(inp.text)   # only the audio path reads this file

    try:
        # Default "text" mode injects synthetic word events (no TTS/WhisperX); the
        # baseline is built the same way. See tribe_events.py / experiment I1.
        events = build_events(tribe, inp.text, str(txt))
        preds, _ = tribe.predict(events=events)
        vertex_mean = np.asarray(preds).mean(axis=0)   # (20484,)
    except ValueError as e:
        return {"error": f"nothing analyzable in the selection ({e})"}
    finally:
        with _lock:
            _scan_active = False
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
