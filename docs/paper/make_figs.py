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


NAVY, ORNG, GRY = "#1f4e79", "#c55a11", "0.55"


def pipeline(fig):
    """Fig. 1: real run of the deployed 3B on one input (tokens from its tokenizer, p from /classify)."""
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.0, 1.1], wspace=0.5)
    a, k, c = (fig.add_subplot(g) for g in gs)

    # (a) gate
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
    a.add_patch(FancyBboxPatch((0.3, 5.6), 9.2, 3.6, boxstyle="round,pad=0.2,rounding_size=0.4", fc="0.96", ec=GRY, lw=0.5))
    a.text(4.9, 7.4, "\u201cOnly 2 left in stock! Thousands of\nshoppers already grabbed theirs,\norder now before the deal ends.\u201d",
           ha="center", va="center", fontsize=5.6, style="italic")
    a.annotate("", (4.9, 3.9), (4.9, 5.4), arrowprops=dict(arrowstyle="-|>", lw=0.6))
    a.text(5.3, 4.6, "gate, K = 2", fontsize=5.6, va="center")
    for y, lab, p_, col in [(2.6, "manipulation", 0.998, NAVY), (1.1, "none", 0.002, GRY)]:
        a.barh(y, 6.0 * p_, left=3.3, height=0.9, color=col)
        a.text(3.1, y, lab, ha="right", va="center", fontsize=5.6)
        a.text(3.4 + 6.0 * p_, y, f"{p_:.3f}", va="center", fontsize=5.4)
    a.set_title("(a) Gate", fontsize=7, pad=2)

    # (b) KV cache strip + label branches
    k.set_xlim(-2, 100); k.set_ylim(-0.8, 10.8); k.axis("off")
    P = 1472; segs = [("system prompt + definitions", 1249, NAVY), ("exemplars", 202, "#5b8db8"), ("input", 21, "#a9c6e0")]
    x0, sc = 0, 58 / P
    for lab, n, col in segs:
        k.add_patch(plt.Rectangle((x0, 7.6), n * sc, 1.6, fc=col, ec="w", lw=0.4))
        x0 += n * sc
    k.text(24.5, 8.4, "system prompt (1249)", ha="center", va="center", fontsize=5.4, color="w")
    k.text(55, 9.9, "exemplars (202)", ha="right", fontsize=5.2)
    k.annotate("input (21)", (57.7, 9.2), (60, 9.9), fontsize=5.2, arrowprops=dict(arrowstyle="-", lw=0.4))
    k.annotate("", (0, 7.1), (58, 7.1), arrowprops=dict(arrowstyle="<->", lw=0.5, color=GRY))
    k.text(29, 6.6, "shared prefix P = 1472 tokens, prefilled once", ha="center", va="top", fontsize=5.4, color="0.25")
    rows = [("fomo", 2), ("false-urgency", 4), ("social-proof-conformity", 5), ("dopamine-bait", 5), ("11 further labels", None)]
    for i, (lab, n) in enumerate(rows):
        y = 4.9 - i * 1.25
        k.plot([58, 64, 66], [8.4, y, y], color=GRY, lw=0.5)
        if n:
            for j in range(n):
                k.add_patch(plt.Rectangle((66.5 + j * 2.1, y - 0.42), 1.8, 0.84, fc=ORNG, ec="none"))
            k.text(66.5 + n * 2.1 + 0.8, y, lab, va="center", fontsize=5.3)
        else:
            k.text(66.5, y, "\u22ee  11 further labels", va="center", fontsize=5.3, color="0.3")
    k.text(64, 7.0, "per label: append its tokens, read log p,\nthen trim the cache back to P", ha="left", va="center", fontsize=5.3, color=ORNG)
    k.set_title("(b) Prefix KV reuse across K = 15 labels", fontsize=7, pad=2)

    # (c) confidences
    labs = ["fomo", "false-urgency", "social-proof", "dopamine-bait", "11 others"]
    ps = [0.69, 0.207, 0.087, 0.010, 0.005]
    ys = np.arange(len(labs))[::-1]
    c.barh(ys, ps, color=[NAVY if p_ >= 0.15 else "0.75" for p_ in ps], height=0.62)
    for y, p_ in zip(ys, ps): c.text(p_ + 0.02, y, f"{p_:.3f}" if p_ < 0.1 else f"{p_:.2f}", va="center", fontsize=5.4)
    c.axvline(0.15, color=ORNG, lw=0.7, ls="--"); c.text(0.16, 4.75, "mixture floor 0.15", fontsize=5.2, color=ORNG, va="bottom"); c.set_ylim(-0.6, 5.3)
    c.set_yticks(ys); c.set_yticklabels(labs, fontsize=5.6); c.set_xlim(0, 0.9)
    c.tick_params(axis="x", labelsize=5.5); c.set_xlabel("confidence $c_i$, Eq. (2)", fontsize=6)
    for sp in ("top", "right"): c.spines[sp].set_visible(False)
    c.set_title("(c) Output", fontsize=7, pad=2)


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
    fig = plt.figure(figsize=(7.0, 1.75)); pipeline(fig)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.2)
    fig.savefig("fig_pipeline.png", dpi=600); plt.close(fig)
    print("ok")
