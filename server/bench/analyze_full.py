#!/usr/bin/env python3
"""Summarizes full_bench.py output: latency distributions, numerical agreement, pipeline accuracy.
    python bench/analyze_full.py bench/results/full_3b_2026-11-14.jsonl [...]"""
import json, math, sys
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
import server as s

def pct(xs, q):
    xs = sorted(xs); k = (len(xs) - 1) * q; f = math.floor(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)

def softmax(v, t):
    m = max(v); e = [math.exp((x - m) / t) for x in v]; z = sum(e); return [x / z for x in e]

def predict(rec):
    if softmax(rec["gate"], s.GATE_TEMP)[0] >= s.GATE_NONE_THRESHOLD:    # GATE_LABELS = ("no", "yes")
        return "none"
    return s.VECTORS[max(range(len(rec["tech"])), key=lambda i: rec["tech"][i])]

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)

def summarize(path):
    meta, by = None, defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        if r.get("meta"): meta = r; continue
        by[r["strategy"]][r["i"]] = r
    out = {"model": meta["model"], "load_s": round(meta["load_s"], 2), "build_s": {k: round(v, 2) for k, v in meta["build_s"].items()},
           "static_len": meta["static_len"], "n": len(by["deployed"])}
    lat = {}
    for st, recs in by.items():
        for k in ("gate_s", "tech_s", "total"):
            xs = [r.get(k, r.get("gate_s", 0) + r["tech_s"]) if k == "total" else r.get(k) for r in recs.values()]
            xs = [x for x in xs if x is not None]
            if xs and not (st == "naive" and k != "tech_s"):
                lat[f"{st}.{k}"] = {q: round(pct(xs, p), 3) for q, p in [("p50", .5), ("p95", .95), ("p99", .99), ("max", 1)]}
    out["latency"] = lat
    d = by["deployed"]; agree = {}
    for st in ("static", "packed", "naive"):
        pairs = [(d[i], r) for i, r in by[st].items() if i in d]
        if not pairs: continue
        diffs = [abs(x - y) for a, b in pairs for x, y in zip(a["tech"], b["tech"])]
        flips = sum(max(range(15), key=lambda j: a["tech"][j]) != max(range(15), key=lambda j: b["tech"][j]) for a, b in pairs)
        conf = [abs(softmax(a["tech"], s.CONFIDENCE_TEMP)[j] - softmax(b["tech"], s.CONFIDENCE_TEMP)[j]) for a, b in pairs for j in range(15)]
        g = {}
        if st != "naive":
            g["gate_maxdiff"] = round(max(abs(x - y) for a, b in pairs for x, y in zip(a["gate"], b["gate"])), 4)
            g["pipeline_flips"] = sum(predict(a) != predict(b) for a, b in pairs)
        margins = []
        for a, b in pairs:
            top = sorted(a["tech"], reverse=True); margins.append(top[0] - top[1])
        agree[st] = {"n": len(pairs), "tech_argmax_flips": flips, "max_abs_diff": round(max(diffs), 4),
                     "p99_abs_diff": round(pct(diffs, .99), 4), "max_conf_change": round(max(conf), 4),
                     "near_ties_lt_0.05": sum(m < 0.05 for m in margins), **g}
    out["agreement"] = agree
    acc = {}
    for st in ("deployed", "static", "packed"):
        recs = by[st].values()
        for split, keep in [("all", lambda r: True), ("holdout", lambda r: r["holdout"])]:
            rs = [r for r in recs if keep(r)]; k = sum(predict(r) == r["label"] for r in rs)
            acc[f"{st}.{split}"] = [k, len(rs)] + [round(x, 1) for x in wilson(k, len(rs))]
    out["accuracy"] = acc
    out["preds_holdout"] = {i: predict(r) == r["label"] for i, r in by["static"].items() if r["holdout"]}
    if by["naive"]:
        out["naive_vs_deployed"] = round(sum(r["tech_s"] for r in by["naive"].values()) /
                                         sum(d[i]["tech_s"] for i in by["naive"]), 2)
    return out

res = [summarize(p) for p in sys.argv[1:]]
for r in res:
    print(json.dumps({k: v for k, v in r.items() if k != "preds_holdout"}, indent=1))
if len(res) == 2:   # exact McNemar, 3B vs 14B on the holdout
    a, b = res[0]["preds_holdout"], res[1]["preds_holdout"]
    n01 = sum(a[i] and not b[i] for i in a); n10 = sum(b[i] and not a[i] for i in a); n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    print(json.dumps({"mcnemar_holdout": {"only_first_right": n01, "only_second_right": n10, "p_exact": p}}))
