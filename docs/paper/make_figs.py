"""Figures for the Litmus note. Inputs are measured values from server/tests/RESULTS.md,
model config.json files and safetensors sizes; derived curves are labelled as models.
Writes fig_main.pdf (4-panel strip, LaTeX one-pager) and fig_*.png (column width, Word)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib.patches import FancyBboxPatch

# figures4papers layout rules (spines off, frameless legends, edged bars), drawn in grayscale and in the
# paper's Times font so figure text matches the body
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "lines.linewidth": 1.0, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False})

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


# server/bench/results/full_*_2026-11-14.jsonl, technique-stage p50: 3B 300 inputs (no reuse: 10),
# 14B 300 (deployed: 30; no reuse: 3, prefix_cache_2026-11-14.jsonl)
LAD = {"3B": [27.0, 1.954, 0.260, 0.127], "14B": [122.8, 8.493, 0.966, 0.512]}
PASS_M = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
PASS = {"3B": [10.5, 11.4, 12.7, 15.2, 21.9, 27.1, 41.0, 41.7, 41.8, 41.5, 74.0, 73.6],
        "14B": [38.6, 41.5, 45.7, 55.9, 84.7, 109.1, 153.3, 154.8, 155.5, 154.6, 287.2, 294.1]}


# p5 and p95 of technique-stage latency over 300 inputs (same files); no-reuse rows have 10 and 3 inputs, no whisker
LAD_P = {"3B": [None, (1.746, 3.05), (0.248, 0.300), (0.121, 0.163)],
         "14B": [None, (7.893, 9.874), (0.888, 1.205), (0.491, 0.674)]}
# individual no-reuse runs (3B, 10 inputs); the 14B file keeps only the median of its 3
NAIVE_3B = [27.9, 27.1, 26.5, 28.1, 26.4, 26.1, 28.0, 26.5, 27.0, 29.2]
NAIVE_N = {"3B": 10, "14B": 3}


C3, C14, INK, MUTED, GRID = "#2a78d6", "#d4582a", "#0b0b0b", "#52514e", "#e4e3df"   # validated pair (dataviz)
COL = {"3B": C3, "14B": C14}
MK = {"3B": "o", "14B": "s"}
plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")


def _clean(ax, grid_axis):
    ax.grid(axis=grid_axis, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK, length=2.5)
    ax.minorticks_off()


def ladder(ax):
    """Fig. 1(a): technique-stage latency as each redundancy is removed (dot plot, p5 to p95 whiskers)."""
    steps = ["No reuse", "Label reuse", "+ Request cache", "+ Packed labels"]
    ys, labels = [], []
    for g, m in enumerate(LAD):
        for i, v in enumerate(LAD[m]):
            yy = g * 6.4 + i; ys.append(yy); labels.append(steps[i])
            c = COL[m]
            if LAD_P[m][i]:
                ax.plot(LAD_P[m][i], [yy, yy], color=c, lw=1.6, solid_capstyle="butt", zorder=2)
            if i == 0:
                ax.plot(v, yy, MK[m], ms=5.5, mfc="white", mec=c, mew=1.3, zorder=3)
                lab = f"{v:.0f} s (n = {NAIVE_N[m]})"
            else:
                ax.plot(v, yy, MK[m], ms=5.5, color=c, mec="white", mew=0.6, zorder=3)
                lab = f"{v:.2f} s" if v < 10 else f"{v:.1f} s"
            hi = max(v, (LAD_P[m][i] or (0, 0))[1])
            ax.text(hi * 1.3, yy, lab, va="center", fontsize=8, color=INK)
        ax.text(0.035, g * 6.4 - 1.2, f"{m} model", fontsize=8.5, fontweight="bold", color=COL[m],
                va="center", transform=ax.get_yaxis_transform(), ha="left")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(0.06, 1500); ax.set_ylim(10.0, -1.9)
    ax.xaxis.set_major_formatter(plain)
    _clean(ax, "x"); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Technique-stage latency (s, log scale)", fontsize=8.5)


def _tiles(ax, edges, top):
    for a, b in zip(edges[:-1], edges[1:]):
        if (edges.index(a)) % 2 == 1:
            ax.axvspan(a, b, color="#f3f2ee", lw=0, zorder=0)
    for e in edges[1:-1]:
        ax.axvline(e, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)


def passcost(ax):
    """Fig. 1(b): latency of one forward pass of m tokens over the cached prefix, linear m."""
    _tiles(ax, [0, 32.5, 64.5, 68], 0)
    ax.axvspan(0.5, 6.5, color="#e6eef8", lw=0, zorder=0)
    for m, ts in PASS.items():
        ax.plot(PASS_M, ts, color=COL[m], lw=0.8, alpha=0.45, zorder=2)
        ax.plot(PASS_M, ts, MK[m], ms=4.2, color=COL[m], mec="white", mew=0.5, zorder=3, label=f"{m} model")
        ax.text(66, ts[-1], m, color=COL[m], fontsize=8.5, fontweight="bold", va="center", ha="left")
    for W, m in [(W14, "14B"), (W3, "3B")]:
        y = 1e3 * W / BW
        ax.axhline(y, color=COL[m], lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.text(34, y * 0.88, f"{m} weight-read bound", fontsize=7.5, color=MUTED, va="top", ha="left")
    ax.axvline(19, color=MUTED, lw=0.8, ls=":", zorder=1)
    tx = ax.get_xaxis_transform()
    for x in (32.5, 64.5):
        ax.text(x, 1.02, f"{int(x - 0.5)} rows", transform=tx, fontsize=7.5, ha="center", va="bottom", color=MUTED)
    ax.text(20.0, 420, "Request suffix\n(19 tokens)", fontsize=7.5, ha="left", va="center", color=MUTED, linespacing=1.0)
    ax.text(0.9, 4.5, "Label passes", fontsize=7.5, ha="left", va="bottom", color=MUTED)
    ax.set_yscale("log"); ax.set_ylim(3.6, 700); ax.set_xlim(0, 72)
    ax.set_yticks([5, 10, 20, 50, 100, 200]); ax.yaxis.set_major_formatter(plain)
    ax.set_xticks([1, 8, 16, 24, 32, 40, 48, 56, 64])
    _clean(ax, "y")
    ax.set_xlabel("New tokens per forward pass, $m$", fontsize=8.5)
    ax.set_ylabel("Latency (ms, log)", fontsize=8.5)


def membudget(ax):
    """Fig. 2: unified-memory budget. Weights, resident static caches, 15-row batched cache."""
    kib = 1024 / 1e9  # KiB -> GB
    rows = [("3B, GQA", W3 / 1e9, (1446 + 1290) * 112 * kib, 15 * 1446 * 112 * kib),
            ("14B, GQA", W14 / 1e9, (1427 + 1281) * 192 * kib, 15 * 1427 * 192 * kib),
            ("14B, MHA*", W14 / 1e9, (1427 + 1281) * 960 * kib, 15 * 1427 * 960 * kib)]
    for i, (lab, w, st, bt) in enumerate(rows):
        ax.barh(i, w, color="0.75", edgecolor="k", lw=0.4, label="4-bit weights" if i == 0 else None)
        ax.barh(i, st, left=w, color="0.3", edgecolor="k", lw=0.4, label="static caches (both stages)" if i == 0 else None)
        ax.barh(i, bt, left=w + st, color="w", edgecolor="k", lw=0.4, hatch="////",
                label="15-row batched cache" if i == 0 else None)
        ax.text(w + st + bt + 0.4, i, f"{w + st + bt:.1f} GB", va="center", fontsize=5.8)
    ax.axvline(24, color="k", lw=0.7, ls="--"); ax.text(24.4, -0.5, "24 GB unified memory", fontsize=5.6, ha="left")
    ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.invert_yaxis(); ax.set_xlim(0, 39); ax.set_ylim(2.55, -0.8); ax.set_xlabel("Memory (GB)")
    ax.legend(fontsize=5.4, loc="center right", bbox_to_anchor=(1.0, 0.56), handlelength=1.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


def qmm(ax):
    """Fig. 2: one 13824x5120 matmul vs rows M; server/bench/results/qmm_2026-11-15.json."""
    import json
    r = json.load(open("../../server/bench/results/qmm_2026-11-15.json"))
    _tiles(ax, [0, 32.5, 64.5, 100], 0)
    for key, lab, mk, c, filled in [("q4_us", "4-bit", "s", INK, True), ("fp16_us", "fp16", "o", MUTED, False)]:
        M = [int(m) for m in r[key]]; t = [v / 1000 for v in r[key].values()]
        ax.plot(M, t, color=c, lw=0.8, alpha=0.5, zorder=2)
        ax.plot(M, t, mk, ms=3.8, color=c, mfc=c if filled else "white", mec=c, mew=0.8, zorder=3, label=lab)
    Mc = [int(m) for m in r["q4_chain_us"]]; tc = [v / 1000 for v in r["q4_chain_us"].values()]
    ax.plot(Mc, tc, "x", ms=3.6, color=INK, mew=0.7, zorder=3, label="4-bit chain")
    for x, lab in [(21, "tile 1"), (48.5, "tile 2"), (80, "tile 3")]:
        ax.text(x, 0.3, lab, fontsize=7.5, ha="center", va="center", color=MUTED)
    ax.set_xlim(0, 100); ax.set_ylim(0, 3.0); ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks([1, 16, 32, 48, 64, 80, 96])
    _clean(ax, "y")
    ax.set_xlabel("Activation rows, $M$", fontsize=8.5); ax.set_ylabel("Latency (ms)", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.1)


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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 1.75), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1]})
    ladder(a1); passcost(a2)
    a1.set_title("(a)", fontsize=9, loc="left", x=-0.3, fontweight="bold"); a2.set_title("(b)", fontsize=9, loc="left", x=-0.14, fontweight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"fig_hero.{ext}", dpi=600)
    plt.close(fig)
    for name, f, size in [("x_reuse", ladder, (3.6, 2.5)), ("x_staircase", passcost, (3.6, 2.5))]:
        fig, a = plt.subplots(figsize=size, constrained_layout=True)
        f(a); fig.savefig(f"fig_{name}.png", dpi=600); plt.close(fig)
    for name, f, size in [("mem", membudget, (3.4, 1.42)), ("qmm", qmm, (3.0, 1.15))]:
        fig, a = plt.subplots(figsize=size, constrained_layout=True)
        f(a)
        for ext in (("png", "pdf", "svg") if name == "qmm" else ("png",)):
            fig.savefig(f"fig_{name}.{ext}", dpi=600)
        plt.close(fig)
    print("ok")
