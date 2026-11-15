#!/usr/bin/env python3
"""Isolates one matmul from the model: time of x(M x K) @ W^T for a 14B MLP weight (13824 x 5120),
4-bit quantized (group 64, the checkpoints' format) vs dense fp16, as M grows. If the staircase in
pass_cost_bench.py comes from the quantized matmul kernels, it appears here and not in the fp16 run."""
import json, statistics, time
import mlx.core as mx

N, K, REPS, INNER = 13824, 5120, 15, 20
w = mx.random.normal((N, K)).astype(mx.float16)
wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
mx.eval(w, wq, sc, bi)
res = {"shape": [N, K], "mlx": mx.__version__, "q4_us": {}, "fp16_us": {}}
for M in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 32, 33, 40, 48, 64, 65, 96):
    x = mx.random.normal((M, K)).astype(mx.float16); mx.eval(x)
    for key, f in [("q4_us", lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=4)),
                   ("fp16_us", lambda: x @ w.T)]:
        mx.eval(f())
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter(); mx.eval([f() for _ in range(INNER)]); ts.append((time.perf_counter() - t0) / INNER)
        res[key][M] = 1e6 * statistics.median(ts)
# dependent chain, as in the model: up (N x K) then down (K x N), each op consuming the previous output,
# one eval per chain, so ops cannot overlap the way independently queued calls can
w2 = mx.random.normal((K, N)).astype(mx.float16)
wq2, sc2, bi2 = mx.quantize(w2, group_size=64, bits=4); mx.eval(w2, wq2, sc2, bi2)
res["q4_chain_us"], res["fp16_chain_us"] = {}, {}
for M in (1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 33, 40, 48, 64, 65, 96):
    x0 = mx.random.normal((M, K)).astype(mx.float16); mx.eval(x0)
    for key, up, dn in [("q4_chain_us", lambda x: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=4),
                         lambda y: mx.quantized_matmul(y, wq2, sc2, bi2, transpose=True, group_size=64, bits=4)),
                        ("fp16_chain_us", lambda x: x @ w.T, lambda y: y @ w2.T)]:
        def chain():
            x = x0
            for _ in range(INNER):
                x = dn(up(x)) * 1e-3
            return x
        mx.eval(chain()); ts = []
        for _ in range(REPS):
            t0 = time.perf_counter(); mx.eval(chain()); ts.append((time.perf_counter() - t0) / (2 * INNER))
        res[key][M] = 1e6 * statistics.median(ts)
print(json.dumps(res))
