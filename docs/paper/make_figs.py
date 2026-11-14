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


def hero(ax):
    """Fig. 1(a): every configuration on one frontier; each paper section is an arrow."""
    ax.axvspan(0.2, 1.0, color="0.93", lw=0)
    ax.text(0.45, 90.5, "interactive\n(< 1 s)", ha="center", va="top", fontsize=5.8, color="0.35")
    pts = {"3B": (188, 300, 0.6), "8B": (207, 300, 7.4), "14B": (238, 300, 12.7), "14B few-shot": (247, 300, 10.0)}
    xy = {}
    for name, (k, n, t) in pts.items():
        p, lo, hi = wilson(k, n); xy[name] = (t, p)
        ax.errorbar(t, p, yerr=[[lo], [hi]], xerr=[[3], [3]] if "few" in name else None, fmt="s" if "few" in name else "o",
                    ms=3.6, color="k", capsize=1.4, elinewidth=0.5, zorder=3)
    p, lo, hi = wilson(32, 42)
    ax.errorbar(35, p, yerr=[[lo], [hi]], fmt="^", ms=3.8, color="k", mfc="w", capsize=1.4, elinewidth=0.5, zorder=3)
    # Pareto frontier
    fx = [0.6, 7.4, 7.4, 10.0, 10.0, 150]; fy = [62.7, 62.7, 69.0, 69.0, 82.3, 82.3]
    ax.plot(fx, fy, color="0.6", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.text(60, 83.3, "Pareto frontier", fontsize=5.6, color="0.4", ha="center")
    arrow = dict(arrowstyle="-|>", lw=0.7, color="k", shrinkA=4, shrinkB=4)
    ax.annotate("", xy["8B"], xy["3B"], arrowprops={**arrow, "connectionstyle": "arc3,rad=-0.15"})
    ax.annotate("", xy["14B"], xy["8B"], arrowprops={**arrow, "connectionstyle": "arc3,rad=-0.25"})
    ax.text(2.3, 64.8, "model scale\n+16.6 pts (VII)", ha="center", fontsize=5.6)
    ax.annotate("", xy["14B few-shot"], xy["14B"], arrowprops=arrow)
    ax.text(14.5, 85.6, "few-shot +3.0 pts", fontsize=5.6, va="center")
    ax.text(42, 72.3, "reasoning model\ndecode-bound (VI),\nslower and less accurate", fontsize=5.6, va="top")
    for name, off in {"3B": (-16, 5), "8B": (5, -7), "14B": (5, -9), "14B few-shot": (-48, 6)}.items():
        ax.annotate(name, xy[name], textcoords="offset points", xytext=off, fontsize=6)
    ax.annotate("R1-14B*", (35, 76.2), textcoords="offset points", xytext=(5, 3), fontsize=6)
    ax.set_xscale("log"); ax.set_xlim(0.2, 150); ax.set_ylim(54, 91)
    ax.set_xlabel("Latency per classification (s), Apple M4 Pro"); ax.set_ylabel("Top-1 accuracy (%)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


def topk(ax):
    """Fig. 1(b): 3B top-k accuracy (bench_3b_top2, N=300)."""
    vals = [(192, "top-1"), (219, "top-2"), (222, "top-3")]
    for i, (k, lab) in enumerate(vals):
        p, lo, hi = wilson(k, 300)
        ax.bar(i, p, width=0.55, color="0.25" if i == 0 else "0.65")
        ax.errorbar(i, p, yerr=[[lo], [hi]], color="k", capsize=1.6, elinewidth=0.5)
        ax.text(i, p + hi + 1, f"{p:.1f}", ha="center", fontsize=6)
    ax.set_xticks(range(3)); ax.set_xticklabels([v[1] for v in vals], fontsize=6.5)
    ax.set_ylim(50, 92); ax.set_ylabel("3B accuracy (%)"); ax.set_yticks(range(50, 81, 10))
    ax.text(1, 91.5, "top-2 adds 9.0 pts:\nmost errors name a\nco-present technique", ha="center", va="top", fontsize=5.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.75), constrained_layout=True)
    for a, f, t in zip(ax, [acc, kv, reuse, decode],
                       ["(a) Accuracy vs. latency", "(b) KV-cache memory", "(c) Prefix KV reuse", "(d) Decode roofline"]):
        f(a); a.set_title(t, fontsize=7); a.tick_params(labelsize=6)
    fig.savefig("fig_main.pdf"); plt.close(fig)

    for name, f, size in [("acc", acc, (3.4, 1.75)), ("kv", kv, (3.4, 1.75)), ("reuse", reuse, (3.4, 1.65)),
                          ("decode", decode, (3.4, 1.75))]:
        fig, a = plt.subplots(figsize=size, constrained_layout=True)
        f(a); a.tick_params(labelsize=6.5)
        fig.savefig(f"fig_{name}.png", dpi=600); plt.close(fig)
    fig, a1 = plt.subplots(figsize=(7.0, 1.95), constrained_layout=True)
    hero(a1); a1.tick_params(labelsize=6.5)
    fig.savefig("fig_hero.png", dpi=600); plt.close(fig)
    print("ok")
