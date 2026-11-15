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
print(json.dumps(res))
