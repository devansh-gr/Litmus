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


# server/bench/results/*_2026-11-14.jsonl (technique stage, medians over 10 inputs; naive over 3)
LAD = {"3B": [24.05, 1.80, 0.245, 0.116], "14B": [122.8, 7.69, 0.981, 0.442]}
PASS_M = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
PASS = {"3B": [10.5, 11.4, 12.7, 15.2, 21.9, 27.1, 41.0, 41.7, 41.8, 41.5, 74.0, 73.6],
        "14B": [38.6, 41.5, 45.7, 55.9, 84.7, 109.1, 153.3, 154.8, 155.5, 154.6, 287.2, 294.1]}


def ladder(ax):
    """Fig. 1(a): technique-stage latency as each redundancy is removed."""
    steps = ["no reuse", "+ label prefix reuse", "+ cross-request prefix", "packed label pass"]
    y = {"3B": 0.0, "14B": 1.0}
    for m, vals in LAD.items():
        for i, v in enumerate(vals):
            yy = y[m] + (i - 1.5) * 0.2
            pred = i == 3
            ax.barh(yy, v, height=0.17, color="w" if pred else ["0.8", "0.55", "0.15"][i],
                    edgecolor="k", lw=0.5, hatch="/////" if pred else None, zorder=2)
            txt = f"{v:.2f} s" if v < 10 else f"{v:.0f} s"
            ax.text(v * 1.12, yy, txt + (" (predicted)" if pred else ""), va="center", fontsize=5.8)
        f = vals[0] / vals[2]
        ax.text(0.052, y[m] - 0.44, f"{m}: {f:.0f}x faster with both caches", ha="left", va="center", fontsize=6.4, fontweight="bold")
    ax.axvline(1.0, color="0.5", lw=0.6, ls=":", zorder=1)
    ax.text(1.05, -0.66, "1 s", fontsize=5.8, color="0.35")
    ax.set_xscale("log"); ax.set_xlim(0.05, 400); ax.set_ylim(-0.72, 1.4)
    ax.set_yticks([y[m] + (i - 1.5) * 0.2 for m in LAD for i in range(4)])
    ax.set_yticklabels(steps * 2, fontsize=5.8)
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Technique-stage latency (s), 15 labels, Apple M4 Pro")
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)


def passcost(ax):
    """Fig. 1(b): cost of one forward pass of m tokens over a cached 1.4k-token prefix."""
    for (m, ts), mk in zip(PASS.items(), ["o", "s"]):
        ax.step(PASS_M, ts, where="post", color="0.6", lw=0.6)
        ax.plot(PASS_M, ts, mk, ms=3, color="k", mfc="w" if m == "3B" else "k", label=f"{m} measured")
    for W, lab in [(W14, "14B"), (W3, "3B")]:
        ax.axhline(1e3 * W / BW, color="k", lw=0.6, ls="--")
        ax.text(1.05, 1e3 * W / BW * 0.86, f"{lab} bound $W/B$", fontsize=5.6, ha="left", va="top")
    ax.axvspan(1, 6, color="0.92", lw=0, zorder=0)
    ax.text(2.45, 420, "label\npasses", fontsize=5.6, ha="center", va="top", color="0.3")
    ax.axvline(19, color="0.7", lw=0.6)
    ax.text(18, 420, "input\nsuffix", fontsize=5.6, ha="right", va="top", color="0.3")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_ylim(4, 500); ax.set_xlim(0.85, 75)
    ax.set_xlabel("New tokens in the pass, $m$"); ax.set_ylabel("Pass latency (ms)")
    ax.legend(fontsize=5.8, loc="lower right")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)


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
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 1.85), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]})
    ladder(a1); passcost(a2)
    a1.set_title("(a) Removing redundant work", fontsize=7); a2.set_title("(b) Cost of one short forward pass", fontsize=7)
    for a in (a1, a2): a.tick_params(labelsize=6.2)
    fig.savefig("fig_hero.png", dpi=600); plt.close(fig)
    for name, f, size in [("mem", membudget, (3.4, 1.42))]:
        fig, a = plt.subplots(figsize=size, constrained_layout=True)
        f(a); a.tick_params(labelsize=6.2)
        fig.savefig(f"fig_{name}.png", dpi=600); plt.close(fig)
    print("ok")
