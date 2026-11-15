"""Packed label pass: every label's continuation tokens go through ONE forward over the cached prompt,
under a tree-shaped mask (each label sees the prompt and its own earlier tokens, never another label).

Tokens are ordered by depth j (all labels' j=0 tokens, then j=1, ...). Every token at depth j sits at
position offset+j, so RoPE is applied per depth group instead of to consecutive positions."""
import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache

_PLAN = None  # list of (start, length, depth) while a packed pass runs


class _GroupRoPE:
    def __init__(self, rope):
        self.rope = rope

    def __call__(self, x, offset=0):
        if _PLAN is None:
            return self.rope(x, offset=offset)
        B, H, _, D = x.shape
        parts = []
        for s, g, j in _PLAN:                         # same position for the whole group: fold it into heads
            xs = x[:, :, s:s + g, :].reshape(B, H * g, 1, D)
            parts.append(self.rope(xs, offset=offset + j).reshape(B, H, g, D))
        return mx.concatenate(parts, axis=2)


def install(model):
    for layer in model.layers:
        if not isinstance(layer.self_attn.rope, _GroupRoPE):
            layer.self_attn.rope = _GroupRoPE(layer.self_attn.rope)


def _lse(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def packed_scores(model, cache, last_logits, labels):
    """Score all labels from the cache state in one pass; cache is trimmed back afterwards."""
    global _PLAN
    base = _lse(last_logits)
    order, plan, depth = [], [], 0                     # order[k] = (label i, depth j)
    while True:
        grp = [(i, depth) for i, l in enumerate(labels) if len(l) - 1 > depth]
        if not grp:
            break
        plan.append((len(order), len(grp), depth)); order += grp; depth += 1
    if not order:
        out = mx.stack([base[l[0]] for l in labels]); mx.eval(out); return out
    n, off = len(order), cache[0].offset
    idx = {o: k for k, o in enumerate(order)}
    tree = [[order[q][0] == order[k][0] and order[k][1] <= order[q][1] for k in range(n)] for q in range(n)]
    mask = mx.concatenate([mx.ones((n, off), dtype=mx.bool_), mx.array(tree)], axis=1)
    saved = [c.make_mask for c in cache]
    for c in cache:
        c.make_mask = lambda *a, **k: mask
    _PLAN = plan
    try:
        lg = _lse(model(mx.array([[labels[i][j] for i, j in order]]), cache=cache)[0])
    finally:
        _PLAN = None
        for c, m in zip(cache, saved):
            c.make_mask = m
    out = mx.stack([base[l[0]] + sum((lg[idx[(i, j)], t] for j, t in enumerate(l[1:])), mx.array(0.0))
                    for i, l in enumerate(labels)]) / mx.array([len(l) for l in labels])
    mx.eval(out)
    trim_prompt_cache(cache, n)
    return out


def packed_step(model, cache, static_len, prompt, labels):
    suffix = prompt[static_len:]
    last = model(mx.array([suffix]), cache=cache)[0][-1]
    out = packed_scores(model, cache, last, labels)
    trim_prompt_cache(cache, len(suffix))
    return out
