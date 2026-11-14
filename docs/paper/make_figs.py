"""Figures for the Litmus note. Inputs are measured values from server/tests/RESULTS.md,
model config.json files and safetensors sizes; derived curves are labelled as models.
Writes fig_main.pdf (4-panel strip, LaTeX one-pager) and fig_*.png (column width, Word)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})

W14, W3, BW = 8.31e9, 1.81e9, 273e9        # 4-bit weight bytes, M4 Pro bandwidth (B/s)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*p, 100*(p-(c-h)), 100*((c+h)-p)


def acc(ax, fs=6):
    pts = [("3B", 188, 300, 0.6, None, "o"), ("8B", 207, 300, 7.4, None, "o"),
           ("14B", 238, 300, 12.7, None, "o"), ("14B+FS", 247, 300, 10.0, (3, 3), "s"),
           ("R1-14B*", 32, 42, 35.0, None, "^")]
    for name, k, n, t, xr, m in pts:
        p, lo, hi = wilson(k, n)
        ax.errorbar(t, p, yerr=[[lo], [hi]], xerr=None if xr is None else [[xr[0]], [xr[1]]],
                    fmt=m, ms=3.5, color="k", mfc="w" if name.startswith("R1") else "k",
                    capsize=1.5, elinewidth=0.6)
        ax.annotate(name, (t, p), textcoords="offset points", fontsize=fs, ha="left",
                    xytext={"14B+FS": (-30, 4), "14B": (4, -2), "R1-14B*": (5, -3)}.get(name, (4, 3)))
    ax.set_xscale("log"); ax.set_xlim(0.3, 150); ax.set_ylim(55, 92)
    ax.set_xlabel("Latency per scan (s)"); ax.set_ylabel("Top-1 accuracy (%)")


def kv(ax, fs=5):
    cfg = {"3B (L=28, H$_{kv}$=8)": (28, 8, 128), "8B (L=32, H$_{kv}$=8)": (32, 8, 128),
           "14B (L=48, H$_{kv}$=8)": (48, 8, 128), "14B MHA (H$_{kv}$=40)": (48, 40, 128)}
    n = np.linspace(0, 8192, 200)
    for (lab, (L, H, d)), ls in zip(cfg.items(), ["-", "--", "-", ":"]):
        ax.plot(n, 2*L*H*d*2*n/2**30, ls, color="0.55" if lab[:2] in ("3B", "8B") else "k", label=lab)
    ax.axvline(1446, color="0.7", lw=0.6); ax.text(1600, 3.4, "prompt\n1446 tok", fontsize=fs + 0.5)
    ax.set_xlabel("Context length (tokens)"); ax.set_ylabel("KV cache (GiB)")
    ax.set_ylim(0, 7.8); ax.legend(fontsize=fs, loc="upper left", handlelength=1.6)


def reuse(ax, fs=5):
    K = np.arange(1, 17); l = 62/15
    for P, ls in [(350, "--"), (1446, "-")]:
        ax.plot(K, K*(P+l)/(P+K*(l-1)), ls, color="k", label=f"Eq. (3), P={P}")
    ax.plot(12, 4014/446, "o", ms=3.5, color="k", mfc="w", label="measured p50 (K=12, P≈350)")
    ax.set_xlabel("Candidate labels K"); ax.set_ylabel("Forward-token reduction (×)")
    ax.legend(fontsize=fs, loc="upper left")


def decode(ax, fs=5):
    n = np.linspace(0, 1536, 100)
    for lab, W, ls in [("14B decode bound", W14, "-"), ("3B decode bound", W3, "--")]:
        ax.plot(n, n*W/BW, ls, color="k", label=lab)
    ax.axhline(35, color="0.6", lw=0.6); ax.text(1100, 36.5, "R1-14B measured, 35 s", fontsize=fs + 0.5, ha="right")
    ax.plot(0, 10, "s", ms=3.5, color="k", label="14B log-prob scoring")
    ax.set_xlabel("Generated tokens n"); ax.set_ylabel("Latency (s)"); ax.set_ylim(0, 48)
    ax.legend(fontsize=fs, loc="upper left", bbox_to_anchor=(0, 0.7), handlelength=1.5)


def pipeline(ax):
    ax.set_xlim(0, 100); ax.set_ylim(3, 43.5); ax.axis("off")
    def box(x, y, w, h, txt, fill="w"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fill, ec="k", lw=0.6))
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=6.2)
    def arr(x0, y0, x1, y1, t=None):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k"))
        if t: ax.text((x0+x1)/2 + (1.2 if x0 == x1 else 0), (y0+y1)/2 + (0 if x0 == x1 else 1.2), t, fontsize=5.5, ha="left" if x0 == x1 else "center", va="center" if x0 == x1 else "bottom")
    box(1, 26, 15, 9, "Input text")
    box(22, 26, 19, 9, "Gate, K=2\n{manipulation, none}")
    box(47, 26, 19, 9, "Prefill prefix once\nP tokens → KV cache", fill="0.92")
    box(72, 26, 26, 9, "Score label i: append ℓ$_i$\ntokens, read log p, trim to P", fill="0.92")
    box(72, 5, 26, 9, "argmax + mixture\n(labels with p ≥ 0.15)")
    box(22, 5, 19, 9, "Output: none")
    arr(16.5, 30.5, 21.5, 30.5); arr(41.5, 30.5, 46.5, 30.5, "pass")
    arr(66.5, 30.5, 71.5, 30.5); arr(85, 25.5, 85, 14.5, "i = 1..15")
    arr(31.5, 25.5, 31.5, 14.5, "none")
    ax.annotate("", (75, 35.6), (95, 35.6), arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.4",
                connectionstyle="arc3,rad=0.35"))
    ax.text(85, 41.8, "loop over K labels, cache reused", fontsize=5.3, ha="center", color="0.3")


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.75), constrained_layout=True)
    for a, f, t in zip(ax, [acc, kv, reuse, decode],
                       ["(a) Accuracy vs. latency", "(b) KV-cache memory", "(c) Prefix KV reuse", "(d) Decode roofline"]):
        f(a); a.set_title(t, fontsize=7); a.tick_params(labelsize=6)
    fig.savefig("fig_main.pdf"); plt.close(fig)

    for name, f, size in [("acc", acc, (3.4, 1.75)), ("kv", kv, (3.4, 1.75)), ("reuse", reuse, (3.4, 1.65)),
                          ("decode", decode, (3.4, 1.75)), ("pipeline", pipeline, (7.0, 1.55))]:
        fig, a = plt.subplots(figsize=size, constrained_layout=True)
        f(a)
        if name != "pipeline": a.tick_params(labelsize=6.5)
        fig.savefig(f"fig_{name}.png", dpi=600); plt.close(fig)
    print("ok")
