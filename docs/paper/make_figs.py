"""Figures for the Litmus one-page note. All inputs are measured values from
server/tests/RESULTS.md or model config.json; derived curves are labelled as models."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*p, 100*(p-(c-h)), 100*((c+h)-p)

fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.75), constrained_layout=True)

# (a) accuracy vs latency
pts = [("3B", 188, 300, 0.6, None, "o"), ("8B", 207, 300, 7.4, None, "o"),
       ("14B", 238, 300, 12.7, None, "o"), ("14B+FS", 247, 300, 10.0, (3, 3), "s"),
       ("R1-14B*", 32, 42, 35.0, None, "^")]
for name, k, n, t, xr, m in pts:
    p, lo, hi = wilson(k, n)
    ax[0].errorbar(t, p, yerr=[[lo], [hi]], xerr=None if xr is None else [[xr[0]], [xr[1]]],
                   fmt=m, ms=3.5, color="k", mfc="w" if name.startswith("R1") else "k",
                   capsize=1.5, elinewidth=0.6)
    ax[0].annotate(name, (t, p), xytext={"14B+FS": (-30, 4), "14B": (4, -2), "R1-14B*": (5, -3)}.get(name, (4, 3)), textcoords="offset points", fontsize=6, ha="left")
ax[0].set_xscale("log"); ax[0].set_xlim(0.3, 150); ax[0].set_ylim(55, 92)
ax[0].set_xlabel("Latency per scan (s)"); ax[0].set_ylabel("Top-1 accuracy (%)")
ax[0].set_title("(a) Accuracy vs. latency", fontsize=7)

# (b) KV-cache footprint, fp16, from config.json
cfg = {"3B (L=28,H$_{kv}$=8)": (28, 8, 128), "8B (L=32,H$_{kv}$=8)": (32, 8, 128),
       "14B (L=48,H$_{kv}$=8)": (48, 8, 128), "14B MHA (H$_{kv}$=40)": (48, 40, 128)}
n = np.linspace(0, 8192, 200)
for (lab, (L, H, d)), ls in zip(cfg.items(), ["-", "--", "-", ":"]):
    ax[1].plot(n, 2*L*H*d*2*n/2**30, ls, color="0.55" if "3B" in lab or "8B" in lab else "k", label=lab)
ax[1].axvline(1446, color="0.7", lw=0.6); ax[1].text(1600, 3.4, "prompt\n1446 tok", fontsize=5.5)
ax[1].set_xlabel("Context length (tokens)"); ax[1].set_ylabel("KV cache (GiB)")
ax[1].set_ylim(0, 7.8); ax[1].legend(fontsize=5, loc="upper left", handlelength=1.6)
ax[1].set_title("(b) KV-cache memory", fontsize=7)

# (c) prefix-reuse speedup S(K) = K(P+l)/(P+K(l-1))
K = np.arange(1, 17); l = 62/15
for P, ls in [(350, "--"), (1446, "-")]:
    ax[2].plot(K, K*(P+l)/(P+K*(l-1)), ls, color="k", label=f"model, P={P}")
ax[2].plot(12, 4014/446, "o", ms=3.5, color="k", mfc="w", label="measured p50 (K=12)")
ax[2].set_xlabel("Candidate labels K"); ax[2].set_ylabel("Forward-token reduction (×)")
ax[2].legend(fontsize=5, loc="upper left"); ax[2].set_title("(c) Prefix KV reuse", fontsize=7)

# (d) decode is bandwidth bound: t >= n W / B
B = 273e9; t_gen = np.linspace(0, 1536, 100)
for lab, W, ls in [("14B decode", 7.7e9, "-"), ("3B decode", 1.7e9, "--")]:
    ax[3].plot(t_gen, t_gen*W/B, ls, color="k", label=lab)
ax[3].axhline(35, color="0.6", lw=0.6); ax[3].text(1150, 36.5, "R1-14B measured, 35 s", fontsize=5.5, ha="right")
ax[3].plot(0, 10, "s", ms=3.5, color="k", label="14B scoring")
ax[3].set_xlabel("Generated tokens n"); ax[3].set_ylabel("Latency (s)"); ax[3].set_ylim(0, 42)
ax[3].legend(fontsize=5, loc="upper left", bbox_to_anchor=(0, 0.8), handlelength=1.5); ax[3].set_title("(d) Decode roofline", fontsize=7)

for a in ax: a.tick_params(labelsize=6)
fig.savefig("fig_main.pdf"); print("ok", 2*48*8*128*2, 2*28*8*128*2, 1446*2*48*8*128*2/2**20)
