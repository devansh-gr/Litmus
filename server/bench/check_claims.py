#!/usr/bin/env python3
"""Recomputes every number stated in the paper (tag paper-v20) from the raw files in bench/results/
and prints PASS or FAIL against the value as printed. No model is run; label lengths come from the
cached tokenizer if it is available offline (otherwise those two checks are skipped).
    .venv/bin/python bench/check_claims.py"""
import json, math, os, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import server as s
from bench.analyze_full import pct, predict, summarize, wilson

R = HERE / "results"
fails = 0


def check(claim, paper, got, tol, src):
    global fails
    ok = abs(paper - got) <= tol
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {claim:<58} paper {paper:<9g} data {got:<11.4g} {src}")


def records(path):
    by, metas = defaultdict(dict), []
    for line in open(path):
        r = json.loads(line)
        if r.get("meta"): metas.append(r)
        else: by[r["strategy"]][r["i"]] = r
    return by, metas


# Table I, Sec. IV and VI: full-path latency, agreement, accuracy (full_bench.py, 300 inputs per model)
F = {"3B": R / "full_3b_2026-11-14.jsonl", "14B": R / "full_14b_2026-11-14.jsonl"}
TABLE = {"3B": {"deployed": (1.56, 1.95, 3.53, 5.80), "static": (0.046, 0.260, 0.306, 0.382), "packed": (0.044, 0.127, 0.170, 0.241)},
         "14B": {"deployed": (6.73, 8.49, 15.21, 17.53), "static": (0.172, 0.966, 1.140, 1.506), "packed": (0.171, 0.512, 0.683, 0.995)}}
SPEEDUP = {"3B": 20.4, "14B": 22.0}
ACC = {"3B": (66.0, 58.1, 73.1), "14B": (86.0, 79.5, 90.7)}
summ = {}
for m, path in F.items():
    by, metas = records(path)
    src = path.name
    summ[m] = summarize(path)
    for st, vals in TABLE[m].items():
        recs = list(by[st].values())
        assert len(recs) == 300, (m, st, len(recs))
        tot = [r["gate_s"] + r["tech_s"] for r in recs]
        got = (pct([r["gate_s"] for r in recs], .5), pct([r["tech_s"] for r in recs], .5), pct(tot, .5), pct(tot, .95))
        for name, pv, gv in zip(("gate p50", "technique p50", "both p50", "both p95"), vals, got):
            dec = len(str(pv).split(".")[1])
            check(f"Table I {m} {st} {name} (s)", pv, gv, 0.5 * 10 ** -dec + 1e-9, src)
    d, p = by["deployed"], by["packed"]
    sp = statistics.median((d[i]["gate_s"] + d[i]["tech_s"]) / (p[i]["gate_s"] + p[i]["tech_s"]) for i in d)
    check(f"{m} median paired speedup, deployed/packed (x)", SPEEDUP[m], sp, 0.05, src)
    flips = sum(predict(d[i]) != predict(by["static"][i]) for i in d)
    pack_flips = sum(predict(by["static"][i]) != predict(p[i]) for i in d)
    check(f"{m} decisions changed by caching (of 300)", {"3B": 3, "14B": 0}[m], flips, 0, src)
    check(f"{m} decisions changed by packing (of 300)", 0, pack_flips, 0, src)
    k = sum(predict(r) == r["label"] for r in p.values() if r["holdout"])
    acc, lo, hi = wilson(k, 150)
    for name, pv, gv in zip(("accuracy", "CI low", "CI high"), ACC[m], (acc, lo, hi)):
        check(f"{m} holdout top-1 {name} (%), packed", pv, gv, 0.05, src)
    if m == "3B":
        diff = max(abs(x - y) for i in d for key in ("gate", "tech") for x, y in zip(d[i][key], p[i][key]))
        check("3B largest base vs packed score difference (nats)", 0.031, diff, 0.0005, src)
        check("3B no-reuse / deployed technique time, 10 inputs (x)", 15.4, summ[m]["naive_vs_deployed"], 0.05, src)
        b = metas[0]["build_s"]
        check("3B static prefix build, both stages (s)", 2.7, b["gate"] + b["tech"], 0.05, src)
        blk = {r["stage"]: r for r in map(json.loads, open(R / "prefix_cache_blocked_2026-11-14.jsonl")) if "Llama" in r["model"]}
        pl = metas[0]["prompt_len"]
        for stage, key, pv in (("technique", "tech", 98.7), ("gate", "gate", 98.5)):
            b = blk[stage]
            check(f"3B {stage} prompt fraction that is static (%)", pv, 100 * b["static_tokens"] / b["prompt_tokens"], 0.05,
                  "prefix_cache_blocked (bench/latency_bench.py sample)")
            print(f"      note: over the 300 benchmark inputs the median is {100 * metas[0]['static_len'][key] / statistics.median(pl[key]):.1f}%")
    else:
        tot = [r["gate_s"] + r["tech_s"] for r in p.values()]
        check("14B packed full path p99 (s)", 1.30, pct(tot, .99), 0.005, src)
        builds = [mt["build_s"]["gate"] + mt["build_s"]["tech"] for mt in metas]
        check("14B static prefix build (s); file holds 3 runs, max shown", 13.8, max(builds), 0.05, src)
        print(f"      note: the three 14B build times in {src} are {', '.join(f'{x:.2f}' for x in builds)} s")
n01, n10 = (sum(summ[a]["preds_holdout"][i] and not summ[b]["preds_holdout"][i] for i in summ[a]["preds_holdout"])
            for a, b in (("3B", "14B"), ("14B", "3B")))
check("holdout discordant inputs won by the 14B", 34, n10, 0, "McNemar")
check("holdout discordant inputs, total", 38, n01 + n10, 0, "McNemar")
pv = 2 * sum(math.comb(n01 + n10, k) for k in range(min(n01, n10) + 1)) / 2 ** (n01 + n10)
check("exact McNemar p < 1e-6 (1 = true)", 1, float(pv < 1e-6), 0, f"p = {pv:.2e}")

# Sec. IV, Eq. (2): the 14B no-reuse baseline comes from an earlier harness (prefix_cache_bench.py at 70595f6)
old = [json.loads(l) for l in open(R / "prefix_cache_2026-11-14.jsonl")]
q14 = next(r for r in old if r.get("model", "").endswith("14B-Instruct-4bit"))
nv, dp = q14["naive_vs_deployed_same_input"]
check("14B no-reuse / deployed technique time (x)", 15.8, nv / dp, 0.05, "prefix_cache_2026-11-14 (one input pair)")
print(f"      note: median over that file's inputs gives {q14['median']['naive'] / q14['median']['deployed']:.2f}x; "
      "per-input no-reuse times were not kept, so n = 3 is not recorded in the file")
P, K, LBL = 1446, 15, 62
check("Eq. (2) predicted reuse factor (x)", 14.6, K * (P + LBL / K - 1) / (P + LBL - K), 0.05, "arithmetic")

# Sec. IV and V: arithmetic on the stated configuration (W, B, layers, KV heads)
B = 273e9
kv = lambda layers, toks: layers * 2 * 8 * 128 * 2 * toks / 1e9
check("3B resident prefix caches, both stages (GB)", 0.31, kv(28, 1446 + 1290), 0.005, "28 layers, 8x128 KV, fp16")
check("14B resident prefix caches, both stages (GB)", 0.53, kv(48, 1427 + 1281), 0.005, "48 layers, 8x128 KV, fp16")
check("Eq. (3) bound, 14B (ms)", 30.4, 8.31e9 / B * 1e3, 0.05, "W / B")
check("Eq. (3) bound, 3B (ms)", 6.6, 1.81e9 / B * 1e3, 0.05, "W / B")

# Sec. V and Fig. 1(b): pass cost (pass_cost_bench.py; medians of 9, spread from a 30-rep repeat)
pc = {json.loads(l)["model"].split("/")[1][:5]: {int(k): v for k, v in json.loads(l)["ms"].items()} for l in open(R / "pass_cost_2026-11-14.jsonl")}
t3, t14 = pc["Llama"], pc["Qwen2"]
src = "pass_cost_2026-11-14"
check("14B single-token pass (ms)", 38.6, t14[1], 0.05, src)
check("3B single-token pass (ms)", 10.5, t3[1], 0.05, src)
check("14B bound / measured single pass (%)", 79, 100 * 8.31e9 / B * 1e3 / t14[1], 0.5, src)
check("3B bound / measured single pass (%)", 63, 100 * 1.81e9 / B * 1e3 / t3[1], 0.5, src)
check("14B t(8) / t(1) (x)", 2.8, t14[8] / t14[1], 0.05, src)
for name, t in (("3B", t3), ("14B", t14)):
    print(f"      note: {name} t(32) / t(12) = {t[32] / t[12]:.3f} (paper: a 32-token pass costs no more than a 12-token one)")
rep = {json.loads(l)["model"].split("/")[1][:5]: json.loads(l) for l in open(R / "pass_cost_2026-11-16.jsonl")}
worst_med = max(abs(r["ms"][k] / pc[n][int(k)] - 1) for n, r in rep.items() for k in r["ms"])
worst_spread = 0.0
for r in rep.values():
    for k, xs in r["raw_ms"].items():
        xs = sorted(xs); med = statistics.median(xs)
        worst_spread = max(worst_spread, abs(pct(xs, .05) / med - 1), abs(pct(xs, .95) / med - 1))
check("30-rep medians within 5% of the 9-rep medians (1 = true)", 1, float(worst_med <= 0.05), 0, f"worst {100 * worst_med:.1f}%")
check("30-rep p5 and p95 within 5% of median (1 = true)", 1, float(worst_spread <= 0.05), 0, f"worst {100 * worst_spread:.1f}%")
try:   # Sec. V predictions: suffix pass (19 tokens, between the measured 16 and 24) plus each label pass
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-3B-Instruct-4bit")
    lens = [len(tok.encode(v, add_special_tokens=False)) for v in s.VECTORS]
    assert sum(lens) == LBL, lens
    for name, t, seq, packed in (("3B", t3, 239, 116), ("14B", t14, 878, 442)):
        t = dict(t); t[5] = (t[4] + t[6]) / 2; suffix = (t[16] + t[24]) / 2     # m = 5 and 19 were not measured
        check(f"{name} predicted sequential technique stage (ms)", seq, suffix + sum(t[n - 1] for n in lens if n > 1), 0.5, src + ", interpolated t(5), t(19)")
        check(f"{name} predicted packed technique stage, t(19) + t(48) (ms)", packed, suffix + t[48], 0.5, src + ", interpolated t(19)")
except Exception as e:
    print(f"SKIP  label-length predictions: tokenizer unavailable offline ({type(e).__name__})")

# Fig. 2: quantized matmul staircase (qmm_bench.py)
q = json.load(open(R / "qmm_2026-11-15.json"))
q4, ch = ({int(k): v for k, v in q[key].items()} for key in ("q4_us", "q4_chain_us"))
print(f"      note: 4-bit step ratios t(33)/t(32) = {q4[33] / q4[32]:.2f}, t(65)/t(64) = {q4[65] / q4[64]:.2f}, "
      f"t(32)/t(16) = {q4[32] / q4[16]:.2f}")
worst = max(abs(ch[m] / q4[m] - 1) for m in ch if m in q4)
check("dependent chain within 7% of independent calls (1 = true)", 1, float(worst <= 0.07), 0, f"worst {100 * worst:.1f}%")

# Sec. V: synthetic packed-token sweep (packed_sweep.py, 50 inputs)
sw = [json.loads(l) for l in open(R / "packed_sweep_50_2026-11-15.jsonl")]
gain = lambda r: statistics.median(a / b for a, b in zip(r["seq_s"], r["packed_s"]))
g = {(r["model"].split("/")[1][:5], r["packed_tokens"]): gain(r) for r in sw}
src = "packed_sweep_50_2026-11-15"
check("sweep gain at 8 tokens, 3B (x)", 1.0, g["Llama", 8], 0.1, src)
check("sweep gain at 96 tokens, 3B (x)", 3.5, g["Llama", 96], 0.05, src)
check("sweep gain at 96 tokens, 14B (x)", 2.8, g["Qwen2", 96], 0.05, src)
check("sweep max packed vs sequential difference (nats)", 0.067, max(r["max_diff"] for r in sw), 0.0005, src)
ctl = [r[k] for r in sw for k in ("causal_diff", "rope_diff")]
check("broken-control smallest difference (nats)", 0.6, min(ctl), 0.05, src)
check("broken-control largest difference (nats)", 13, max(ctl), 0.5, src)

print(f"\n{fails} FAIL" if fails else "\nall checks PASS")
sys.exit(1 if fails else 0)
