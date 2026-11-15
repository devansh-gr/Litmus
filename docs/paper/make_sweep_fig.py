"""Packed-token sweep plot for REPRODUCE.md (not in the 2-page paper)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False})
rows = [json.loads(l) for l in open(Path(__file__).parents[2] / "server/bench/results/packed_sweep_50_2026-11-15.jsonl")]
fig, ax = plt.subplots(figsize=(3.4, 1.6))
for tag, mk, fc in (("3B", "o", "white"), ("14B", "s", "black")):
    r = [x for x in rows if (("3B" in x["model"]) if tag == "3B" else ("14B" in x["model"]))]
    T = [x["packed_tokens"] for x in r]
    for key, ls in (("seq_med", ":"), ("packed_med", "-")):
        ax.plot(T, [1e3 * x[key] for x in r], ls, color="black", lw=0.8, marker=mk, mfc=fc, ms=3.5)
    nudge = {"3B": (0.85, 1.0), "14B": (1.0, 1.18)}[tag]          # keep 14B packed and 3B sequential apart
    ax.text(99, 1e3 * r[-1]["seq_med"] * nudge[0], f"{tag} sequential", va="center", fontsize=7)
    ax.text(99, 1e3 * r[-1]["packed_med"] * nudge[1], f"{tag} packed", va="center", fontsize=7)
for b in (32, 64):
    ax.axvline(b + 0.5, color="0.6", lw=0.6, ls="--")
ax.set_yscale("log"); ax.set_xticks([8, 16, 32, 48, 64, 96]); ax.set_xlim(4, 135)
ax.set_xlabel("Packed label tokens"); ax.set_ylabel("Label scoring (ms)")
fig.savefig(Path(__file__).parent / "fig_sweep.png", dpi=300, bbox_inches="tight")
