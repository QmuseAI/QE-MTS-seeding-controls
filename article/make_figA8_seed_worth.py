"""Fig. A8 (Sec. 'What a quantum seed is worth') from article_numbers.txt, the output of make_article_figs.py.

usage: python make_figA8_seed_worth.py [article_numbers.txt] [outdir]   (defaults: this folder)
(a) best-of-n energy vs sampling cost: uniform sampling, DCQO shots, tabu walk
(b) equivalent uniform samples per DCQO shot, per N (interpolation in log n between A2, A2m, A3)
(c) seed energy vs Hamming distance to the nearest optimum
"""
import re, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "article_numbers.txt")
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
os.makedirs(OUT, exist_ok=True)
txt = open(SRC).read()

# ---------------------------------------------------------------- style (as make_article_figs.py)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
                     "lines.markersize": 3.6, "errorbar.capsize": 0, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b"})
INK, MUTED = "#0b0b0b", "#8a8984"
COL = {"A1": "#1baf7a", "A2": "#2a78d6", "A2m": "#eb6834", "A3": "#e34948", "A2k": "#eda100", "A8": "#6b6a66", "Q1": "#6250d6"}
MK = {"A1": "^", "A2": "o", "A2m": ">", "A3": "h", "A2k": "<", "A8": "X", "Q1": "*"}

# ---------------------------------------------------------------- numbers
ARM = {}   # arm -> (mean E/Eopt, mean Hamming) on N=27-33
for a, e, h in re.findall(r"^\s+(A1|A2k|A2|A2m|A3|A8|Q1)\s+seed E/Eopt ([\d.]+)\s+Hamming ([\d.]+)", txt, re.M):
    ARM[a] = (float(e), float(h))
PERN = {}  # N -> dict arm -> mean seed E/Eopt
for n, q, a2, a2m, a3 in re.findall(r"N=(\d+): Q1 ([\d.]+)\s+A2 ([\d.]+)\s+A2m ([\d.]+)\s+A3 ([\d.]+)", txt):
    PERN[int(n)] = {"Q1": float(q), "A2": float(a2), "A2m": float(a2m), "A3": float(a3)}
assert set(ARM) == set(COL) and sorted(PERN) == list(range(27, 34)), (ARM.keys(), PERN.keys())

NS = np.arange(27, 34)
COST = {"A1": 1.0, "A2k": 1e3, "A2": 1e5, "A2m": 1e6, "A3": 1e7, "Q1": 1e5, "A8": 1000.0 * np.mean(NS ** 2)}
UNI = ["A1", "A2k", "A2", "A2m", "A3"]

def n_equiv(e, pts):
    """n at which the best-of-n uniform energy equals e (piecewise linear in E vs log10 n, as Table A2)."""
    ln = np.log10([p[0] for p in pts]); le = np.array([p[1] for p in pts])
    o = np.argsort(le)
    return 10 ** np.interp(e, le[o], ln[o])

neq = np.array([n_equiv(PERN[n]["Q1"], [(1e5, PERN[n]["A2"]), (1e6, PERN[n]["A2m"]), (1e7, PERN[n]["A3"])]) for n in NS])
ratio = neq / 1e5
neq_mean = n_equiv(ARM["Q1"][0], [(COST[a], ARM[a][0]) for a in UNI])
print("n_eq per N:", " ".join(f"{n}:{v:.2g}" for n, v in zip(NS, neq)))
print(f"uniform samples per DCQO shot: {ratio.min():.0f}-{ratio.max():.0f}; from N-averaged energies: {neq_mean/1e5:.0f}")
print(f"tabu walk cost (mean 1000N^2, N=27-33): {COST['A8']:.3g} evaluations, E/Eopt {ARM['A8'][0]:.3f}")

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.75), gridspec_kw=dict(width_ratios=[1.2, 0.85, 1.05]))

ax = axs[0]
x = [COST[a] for a in UNI]; y = [ARM[a][0] for a in UNI]
ax.plot(x, y, color=MUTED, lw=0.9, zorder=1, label="uniform: best of $n$")
for a in UNI:
    ax.scatter(COST[a], ARM[a][0], s=22, color=COL[a], marker=MK[a], zorder=3, edgecolor="white", lw=0.5)
    ax.annotate(a, (COST[a], ARM[a][0]), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=INK)
ax.scatter(COST["Q1"], ARM["Q1"][0], s=75, color=COL["Q1"], marker="*", zorder=4, edgecolor="white", lw=0.5,
           label="DCQO: best of $10^5$ shots")
ax.annotate("", xy=(neq_mean, ARM["Q1"][0]), xytext=(COST["Q1"] * 1.25, ARM["Q1"][0]),
            arrowprops=dict(arrowstyle="->", color=COL["Q1"], lw=0.8))
ax.text(COST["Q1"] / 2, ARM["Q1"][0], f"×{ratio.min():.0f}–{ratio.max():.0f}\nper shot",
        ha="right", va="center", fontsize=6.3, color=COL["Q1"])
ax.scatter(COST["A8"], ARM["A8"][0], s=30, color=COL["A8"], marker="X", zorder=4, edgecolor="white", lw=0.5,
           label="tabu walk ($1000N^2$ evaluations)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
ax.set_yticks([1, 1.2, 1.5, 2, 3, 5, 8]); ax.set_yticklabels(["1", "1.2", "1.5", "2", "3", "5", "8"])
ax.set_xticks([1, 1e2, 1e4, 1e6, 1e8]); ax.set_xlim(0.4, 1e8); ax.set_ylim(1.05, 10)
ax.set_xlabel("samples or evaluations"); ax.set_ylabel(r"seed energy $E/E_{\rm opt}$ (mean, $N$=27–33)")
ax.set_title("(a) information per sample", loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=6.0, handletextpad=0.3)

ax = axs[1]
ax.bar(NS, ratio, width=0.62, color=COL["Q1"], alpha=0.85)
for n, r in zip(NS, ratio):
    ax.text(n, r + 1.2, f"{r:.0f}", ha="center", fontsize=6.3, color=INK)
ax.set_xticks(NS); ax.set_xlabel("$N$"); ax.set_ylim(0, 75)
ax.set_ylabel("uniform samples per DCQO shot")
ax.set_title("(b) equivalent samples", loc="left")

ax = axs[2]
ax.plot([ARM[a][0] for a in UNI], [ARM[a][1] for a in UNI], color=MUTED, lw=0.9, zorder=1)
for a in UNI + ["A8"]:
    ax.scatter(ARM[a][0], ARM[a][1], s=24, color=COL[a], marker=MK[a], zorder=3, edgecolor="white", lw=0.5)
    off = {"A2k": (-2, 5), "A2": (3, 4), "A8": (4, -9), "A3": (4, -3), "A1": (2, -10)}.get(a, (4, -3))
    ax.annotate(a, (ARM[a][0], ARM[a][1]), xytext=off, textcoords="offset points", fontsize=6.5, color=INK)
ax.scatter(*ARM["Q1"], s=75, color=COL["Q1"], marker="*", zorder=4, edgecolor="white", lw=0.5)
ax.annotate("DCQO (Q1)", ARM["Q1"], xytext=(-6, -10), textcoords="offset points", fontsize=6.5, color=INK, ha="right")
ax.set_xscale("log"); ax.invert_xaxis(); ax.minorticks_off()
ax.set_xticks([8, 5, 3, 2, 1.5, 1.2]); ax.set_xticklabels(["8", "5", "3", "2", "1.5", "1.2"])
ax.set_ylim(9.0, 11.8)
ax.set_xlabel(r"seed energy $E/E_{\rm opt}$"); ax.set_ylabel(r"distance to optimum $d_H$")
ax.set_title("(c) energy vs proximity", loc="left")

fig.tight_layout(w_pad=1.2)
fig.savefig(f"{OUT}/figA8_seed_worth.pdf"); fig.savefig(f"{OUT}/figA8_seed_worth.png", dpi=200)
print("done")
