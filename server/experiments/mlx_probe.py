"""Validate the MLX 4-bit detector: log-prob label scoring, accuracy + latency.

If this matches the fp32 transformers detector's accuracy at a fraction of the
memory/latency, we swap it into the server.
"""

import time
import mlx.core as mx
from mlx_lm import load

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

VECTORS = [
    "fear-mongering", "critical-thinking-suppression", "tribal-in-group-bias",
    "dopamine-bait", "outrage", "authority-appeal", "false-urgency",
    "social-proof-conformity", "none",
]
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
SYSTEM = ("You are a persuasion analyst. Identify the PRIMARY manipulation technique in "
          "the text. The techniques are:\n"
          + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
          + "\nAnswer with the technique name only.")

TESTS = [
    ("You just unlocked an exclusive free jackpot prize, claim it instantly!", "dopamine-bait"),
    ("The council approved the new drainage plan on Tuesday.", "none"),
    ("Experts and top scientists all agree, so there is no need to question this.", "authority-appeal"),
    ("Act now, only hours left before this deadly offer expires forever.", "false-urgency"),
    ("Everyone in our community already switched, do not be the last one left.", "social-proof-conformity"),
    ("These corrupt officials stole your money and laughed about it.", "outrage"),
]


def main():
    t0 = time.time()
    model, tok = load(MODEL)
    print(f"model loaded in {time.time()-t0:.1f}s\n", flush=True)

    def enc(s, special=True):
        try:
            return tok.encode(s, add_special_tokens=special)
        except TypeError:
            return tok._tokenizer.encode(s, add_special_tokens=special)

    def label_scores(text):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = enc(prompt)
        scores = []
        for label in VECTORS:
            lab_ids = enc(label, special=False)
            ids = prompt_ids + lab_ids
            logits = model(mx.array([ids]))[0]                 # (seq, vocab)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            total = 0.0
            for j, tokid in enumerate(lab_ids):
                pos = len(prompt_ids) + j
                total += lp[pos - 1, tokid].item()
            scores.append(total / len(lab_ids))
        return scores

    # warm
    _ = label_scores("warmup")
    correct = 0
    for text, want in TESTS:
        t = time.time()
        s = label_scores(text)
        probs = mx.softmax(mx.array(s)).tolist()
        best = max(range(len(VECTORS)), key=lambda i: probs[i])
        got = VECTORS[best]
        ok = got == want
        correct += ok
        print(f"[{time.time()-t:4.1f}s] {'ok ' if ok else 'MISS'} {got:<26} {int(probs[best]*100)}%  "
              f"(want {want})", flush=True)
    print(f"\n{correct}/{len(TESTS)} correct")


if __name__ == "__main__":
    main()
