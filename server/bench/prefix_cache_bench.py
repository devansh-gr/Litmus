#!/usr/bin/env python3
"""Technique-stage latency under three KV strategies + decode throughput.

  naive    : one full forward per label (no reuse)
  deployed : prefill the whole prompt once per input, score labels, trim back (server.py)
  static   : the input-independent prompt prefix is prefilled ONCE across requests;
             per input only the suffix (input text + template tail) and label tokens run
  batched  : static, but all labels run as one padded batch over a tiled copy of the cache
             (one forward for the suffix + one for every label, instead of 1 + K)

Also checks the three give the same scores, and times greedy decode (ms/token).
    .venv/bin/python bench/prefix_cache_bench.py --model mlx-community/Qwen2.5-14B-Instruct-4bit
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import server as s
from bench.latency_bench import SAMPLES


def lse(x, axis=-1):
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


def score_from(model, cache, last_logits, labels):
    """Score every label continuing from the cache state; trim back after each."""
    base = lse(last_logits)
    out = []
    for ids in labels:
        tot = base[ids[0]]
        if len(ids) > 1:
            lg = lse(model(mx.array([ids[:-1]]), cache=cache)[0])
            tot = tot + lg[mx.arange(len(ids) - 1), mx.array(ids[1:])].sum()
            trim_prompt_cache(cache, len(ids) - 1)
        out.append(tot / len(ids))
    out = mx.stack(out); mx.eval(out)
    return out


def naive(model, prompt, labels):
    out = []
    for ids in labels:
        lp = lse(model(mx.array([prompt + ids]))[0])
        n = len(prompt)
        out.append(sum(lp[n + j - 1, t] for j, t in enumerate(ids)) / len(ids))
    out = mx.stack(out); mx.eval(out)
    return out


def deployed(model, prompt, labels):
    cache = make_prompt_cache(model)
    last = model(mx.array([prompt]), cache=cache)[0][-1]
    return score_from(model, cache, last, labels)


def static_step(model, cache, static_len, prompt, labels):
    suffix = prompt[static_len:]
    last = model(mx.array([suffix]), cache=cache)[0][-1]
    sc = score_from(model, cache, last, labels)
    trim_prompt_cache(cache, len(suffix))          # back to the shared prefix
    return sc


def build_batched(model, cache, K, room=64):
    """K-row copy of the static cache, built once and kept resident, with `room` free slots."""
    bc = make_prompt_cache(model)
    for c, b in zip(cache, bc):
        k, v = c.state
        pad = lambda x: mx.zeros((K, x.shape[1], room, x.shape[3]), x.dtype)
        b.keys = mx.concatenate([mx.repeat(k, K, 0), pad(k)], axis=2)
        b.values = mx.concatenate([mx.repeat(v, K, 0), pad(v)], axis=2)
        b.offset = k.shape[2]
    mx.eval([x for b in bc for x in (b.keys, b.values)])
    return bc


def batched_step(model, cache, bc, static_len, prompt, labels):
    suffix = prompt[static_len:]
    n, T = len(suffix), max(map(len, labels)) - 1
    base = lse(model(mx.array([suffix]), cache=cache)[0][-1])
    for c, b in zip(cache, bc):                       # copy only the n suffix rows into every batch row
        b.keys[..., static_len:static_len + n, :] = c.keys[..., static_len:static_len + n, :]
        b.values[..., static_len:static_len + n, :] = c.values[..., static_len:static_len + n, :]
        b.offset = static_len + n
    if T == 0:                                        # single-token labels: the suffix pass is the whole job
        trim_prompt_cache(cache, n)
        out = mx.stack([base[l[0]] for l in labels]); mx.eval(out)
        return out
    lg = lse(model(mx.array([l[:-1] + [0] * (T - len(l) + 1) for l in labels]), cache=bc))
    out = mx.stack([base[l[0]] + sum((lg[i, j, t] for j, t in enumerate(l[1:])), mx.array(0.0))
                    for i, l in enumerate(labels)]) / mx.array([len(l) for l in labels])
    mx.eval(out)
    trim_prompt_cache(cache, n); trim_prompt_cache(bc, n + T)
    return out


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t0)
    return r, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    ap.add_argument("--naive", type=int, default=3, help="inputs to run the slow naive path on")
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--stage", choices=("technique", "gate"), default="technique")
    a = ap.parse_args()

    model, tok = load(a.model)
    enc = lambda t, sp=True: tok.encode(t, add_special_tokens=sp)
    if a.stage == "gate":
        labels = [enc(v, False) for v in s.GATE_LABELS]
        prompts = [enc(s._chat_prompt(tok, t, s.GATE_SYSTEM)) for t in SAMPLES]
    else:
        labels = [enc(v, False) for v in s.VECTORS]
        prompts = [enc(s._chat_prompt(tok, t, s.SYSTEM, s.TECHNIQUE_FEWSHOT)) for t in SAMPLES]
    static_len = min(next(i for i, (x, y) in enumerate(zip(prompts[0], p)) if x != y) for p in prompts[1:])

    deployed(model, prompts[0], labels)                         # warm-up
    t_build = time.perf_counter()
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompts[0][:static_len]]), cache=cache))
    t_build = time.perf_counter() - t_build
    bc = build_batched(model, cache, len(labels))

    res = {"model": a.model, "stage": a.stage, "prompt_tokens": len(prompts[0]), "static_tokens": static_len,
           "labels": len(labels), "label_tokens": sum(map(len, labels)), "static_build_s": t_build,
           "deployed_s": [], "static_s": [], "batched_s": [], "naive_s": [], "max_abs_diff": 0.0, "argmax_agree": 0}
    # each strategy is timed in its own block: interleaving them leaves the freed full-prompt cache of
    # the deployed path competing with the next call and made the 14B static timings bimodal
    sc = {}
    for k, fn in [("deployed", lambda p: deployed(model, p, labels)),
                  ("static", lambda p: static_step(model, cache, static_len, p, labels)),
                  ("batched", lambda p: batched_step(model, cache, bc, static_len, p, labels))] + \
                 ([("naive", lambda p: naive(model, p, labels))] if a.naive else []):
        sc[k] = []
        for p in prompts[:a.naive] if k == "naive" else prompts:
            r, t = timed(lambda: fn(p), 1); sc[k].append(r); res[k + "_s"] += t
    for i, d in enumerate(sc["deployed"]):
        others = [sc[k][i] for k in sc if k != "deployed" and i < len(sc[k])]
        res["max_abs_diff"] = max(res["max_abs_diff"], *(mx.abs(d - o).max().item() for o in others))
        res["argmax_agree"] += int(all(mx.argmax(d).item() == mx.argmax(o).item() for o in others))

    # greedy decode throughput after a short prompt
    c2 = make_prompt_cache(model)
    y = mx.argmax(model(mx.array([prompts[0][-32:]]), cache=c2)[0][-1]); mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(a.decode):
        y = mx.argmax(model(y.reshape(1, 1), cache=c2)[0][-1]); mx.eval(y)
    res["decode_ms_per_tok"] = 1000 * (time.perf_counter() - t0) / a.decode

    med = lambda xs: statistics.median(xs) if xs else None
    res["median"] = {k: med(res[k + "_s"]) for k in ("naive", "deployed", "static", "batched")}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
