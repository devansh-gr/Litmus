#!/usr/bin/env python3
"""All 300 external_test inputs, both stages, three strategies timed in separate blocks:
  deployed : per-request prefill, labels reuse the prefix (server.py today)
  static   : input-independent prefix cached across requests, labels scored one pass each
  packed   : static, all labels in one tree-masked pass (bench/packed.py)
plus naive (no reuse) on the first --naive inputs, technique stage only.
Streams one JSON line per (strategy, input) with both score vectors and stage times.
    .venv/bin/python bench/full_bench.py --model mlx-community/Qwen2.5-14B-Instruct-4bit --naive 10"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
t_load = time.perf_counter()
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
import server as s
from bench.prefix_cache_bench import deployed, naive, static_step
from bench.packed import install, packed_step

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
ap.add_argument("--naive", type=int, default=10)
ap.add_argument("--out", required=True)
ap.add_argument("--strategies", default="deployed,static,packed")
ap.add_argument("--deployed-limit", type=int, default=None, help="run the slow deployed path on the first N inputs only")
ap.add_argument("--deployed-start", type=int, default=0, help="skip deployed inputs already measured")
a = ap.parse_args()
DATA = Path(__file__).resolve().parents[1] / "tests/data"
rows = [json.loads(l) for l in open(DATA / "external_test.jsonl")]
hold = {json.loads(l)["text"] for l in open(DATA / "external_holdout.jsonl")}

model, tok = load(a.model); install(model)
t_load = time.perf_counter() - t_load
enc = lambda t, sp=True: tok.encode(t, add_special_tokens=sp)
ST = {"gate": ([enc(v, False) for v in s.GATE_LABELS], lambda t: enc(s._chat_prompt(tok, t, s.GATE_SYSTEM))),
      "tech": ([enc(v, False) for v in s.VECTORS], lambda t: enc(s._chat_prompt(tok, t, s.SYSTEM, s.TECHNIQUE_FEWSHOT)))}
prompts = {k: [f(r["text"]) for r in rows] for k, (_, f) in ST.items()}
static_len, caches, build = {}, {}, {}
for k in ST:
    ps = prompts[k]
    static_len[k] = min(next(i for i, (x, y) in enumerate(zip(ps[0], p)) if x != y) for p in ps[1:])
    t0 = time.perf_counter()
    caches[k] = make_prompt_cache(model); mx.eval(model(mx.array([ps[0][:static_len[k]]]), cache=caches[k]))
    build[k] = time.perf_counter() - t0
out = open(a.out, "a")
out.write(json.dumps({"meta": True, "model": a.model, "load_s": t_load, "static_len": static_len, "build_s": build,
                      "prompt_len": {k: [len(p) for p in v] for k, v in prompts.items()}}) + "\n")

STRATS = {"deployed": lambda k, i: deployed(model, prompts[k][i], ST[k][0]),
          "static": lambda k, i: static_step(model, caches[k], static_len[k], prompts[k][i], ST[k][0]),
          "packed": lambda k, i: packed_step(model, caches[k], static_len[k], prompts[k][i], ST[k][0])}
for name in filter(None, a.strategies.split(",")):           # --strategies "" runs only the no-reuse path
    fn = STRATS[name]
    fn("gate", 0); fn("tech", 0)                                   # block warm-up, discarded
    idx = range(a.deployed_start, a.deployed_limit or len(rows)) if name == "deployed" else range(len(rows))
    for i in idx:
        r = rows[i]
        rec = {"strategy": name, "i": i, "label": r["label"], "holdout": r["text"] in hold}
        for k in ("gate", "tech"):
            t0 = time.perf_counter(); sc = fn(k, i); rec[k + "_s"] = time.perf_counter() - t0
            rec[k] = sc.tolist()
        out.write(json.dumps(rec) + "\n"); out.flush()
for i in range(a.naive):
    t0 = time.perf_counter(); sc = naive(model, prompts["tech"][i], ST["tech"][0])
    out.write(json.dumps({"strategy": "naive", "i": i, "tech_s": time.perf_counter() - t0, "tech": sc.tolist()}) + "\n")
    out.flush()
print("done", a.model)
