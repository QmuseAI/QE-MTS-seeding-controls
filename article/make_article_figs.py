"""Figures and derived numbers for the article (all inputs are analysis outputs; no MTS data are re-analysed).

usage  : python make_article_figs.py [OUT]        (run from anywhere; paths are relative to the project root)
inputs : step2_cluster/comment/out/table_gap_perN.csv          (run_main arms: D, CI, A0 log10 Q50, published gap)
         step2_cluster/run_ext/analysis_ext/gaps_ext.csv        (extension arms, DCQO arms, seed-cost-included gaps)
         step2_cluster/run_ext/analysis_ext/{within_arm,optima}.csv
         step2_cluster/seeds/seed_stats_all.csv, step2_cluster/seeds/seed_stats_dcqo.csv
         step1_digitize/data/{fig5_logratio_vector,fig2_q50_digitized}.csv
outputs: figA1-A7 .pdf and article_numbers.txt in OUT (default: this folder)
"""
import csv, os, sys, json
import numpy as np
from scipy import optimize, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.dirname(HERE)                                   # project root
S2 = f"{B}/step2_cluster"; EXT = f"{B}/analysis/frozen"
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE
os.makedirs(OUT, exist_ok=True)
NUM = []
def note(s): print(s); NUM.append(s)

# ---------------------------------------------------------------- style
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
                     "lines.markersize": 3.6, "errorbar.capsize": 0, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b"})
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
COL = {"A1": "#1baf7a", "A2": "#2a78d6", "A2m": "#eb6834", "A3": "#e34948", "A2k": "#eda100", "A8": "#6b6a66",
       "Q1": "#6250d6", "A5": "#008300", "A6": "#e87ba4", "A7a": "#eda100", "Q5": "#008300", "Q5v": "#6250d6", "PQ1_d8": "#2a78d6",
       "M2m_k50": "#2a78d6", "M2m_k10": "#1baf7a", "M2m_k1": "#008300", "P2m_d8": "#6250d6", "P2m_d4": "#e87ba4",
       "M3_k1": "#008300", "P3_d8": "#6250d6", "I2": "#2a78d6", "I2m": "#eb6834", "I3": "#e34948"}
MK = {"A1": "^", "A2": "o", "A2m": ">", "A3": "h", "A2k": "<", "A8": "X", "Q1": "*", "A5": "D", "A6": "v", "A7a": "P",
      "Q5": "D", "Q5v": "D", "PQ1_d8": "o"}
def leg(ax, ncol=1, dy=-0.2, fs=6.4):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, frameon=False, fontsize=fs, handletextpad=0.4, columnspacing=1.0)
def shade(ax, lo=26.5, hi=37.5): ax.axvspan(lo, hi, color="#f1f0ec", zorder=0, lw=0)
def zero(ax): ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
def eb(ax, N, D, lo, hi, key, label, **kw):
    N, D, lo, hi = map(np.asarray, (N, D, lo, hi))
    ax.errorbar(N, D, yerr=[np.clip(D - lo, 0, None), np.clip(hi - D, 0, None)], color=COL.get(key, INK),
                marker=MK.get(key, "o"), ms=kw.pop("ms", 3.6 if MK.get(key, "o") != "*" else 6), lw=1.1,
                elinewidth=0.7, label=label, zorder=3, **kw)

# ---------------------------------------------------------------- data
T = {}
with open(f"{EXT}/table_gap_perN.csv") as f:
    for r in csv.DictReader(f): T[int(r["N"])] = r
def tcol(arm, Nmin=23, Nmax=48):
    out = []
    for N in sorted(T):
        if Nmin <= N <= Nmax and T[N].get(f"D_{arm}", ""):
            out.append((N, float(T[N][f"D_{arm}"]), float(T[N][f"D_{arm}_lo"]), float(T[N][f"D_{arm}_hi"])))
    return list(map(np.array, zip(*out))) if out else [np.array([])] * 4
LQ0 = {N: float(T[N]["log10_Q50_A0"]) for N in T}
G = {}
with open(f"{EXT}/gaps_ext.csv") as f:
    for r in csv.DictReader(f):
        G.setdefault((r["arm"], r["ref"], int(r["seed_cost_counted"])), []).append(
            (int(r["N"]), float(r["D"]), float(r["lo"]), float(r["hi"])))
def gcol(arm, ref="A0", cost=0, Nmin=0, Nmax=99):
    v = sorted(x for x in G.get((arm, ref, cost), []) if Nmin <= x[0] <= Nmax)
    return list(map(np.array, zip(*v))) if v else [np.array([])] * 4
PUB = {}
with open(f"{B}/step1_digitize/data/fig5_logratio_vector.csv") as f:
    for r in csv.DictReader(l for l in f if not l.startswith("#")): PUB[int(r["N"])] = r
PD = {N: float(r["median_D"]) for N, r in PUB.items()}
F2 = {}
with open(f"{B}/step1_digitize/data/fig2_q50_digitized.csv") as f:
    for r in csv.DictReader(l for l in f if not l.startswith("#")): F2[(int(r["N"]), r["method"])] = float(r["log10_Q50"])
SEED = {}
for fn in (f"{S2}/seeds/seed_stats_all.csv", f"{B}/step2_cluster/seeds/seed_stats_dcqo.csv"):
    with open(fn) as f:
        for r in csv.DictReader(f):
            SEED.setdefault((r["arm"], int(r["N"])), []).append((float(r["seedE_min"]), float(r["ham_bestseed_to_opt"])))
EOPT = {27: 37, 28: 50, 29: 62, 30: 59, 31: 67, 32: 64, 33: 64, 34: 65, 35: 73, 36: 82, 37: 86, 38: 87, 39: 99, 40: 108,
        41: 108, 42: 101, 43: 109, 44: 122, 45: 118, 46: 131, 47: 135, 48: 140}
def seedq(arm, lo, hi):
    e = [x[0] / EOPT[N] for N in range(lo, hi + 1) for x in SEED.get((arm, N), [])]
    h = [x[1] for N in range(lo, hi + 1) for x in SEED.get((arm, N), [])]
    return (np.mean(e), np.mean(h)) if e else (np.nan, np.nan)

def fit_delta(Ns, D, T):
    Ns, D, T = map(np.asarray, (Ns, D, T))
    f = lambda L: np.sum((D - np.log10(1 + 10 ** L / T)) ** 2)
    r = optimize.minimize_scalar(f, bounds=(0, 13), method="bounded")
    return 10 ** r.x, np.sqrt(r.fun / len(D))

# ---------------------------------------------------------------- numbers
note("== seed quality and overhead on the common window N=27-33 (T_ref = our A0 median; published: digitised Fig.2 MTS)")
OV = {}
for arm in ("A1", "A2k", "A2", "A2m", "A3", "A8"):
    N, D, lo, hi = tcol(arm, 27, 33)
    d, rms = fit_delta(N, D, [10 ** LQ0[n] for n in N]); e, h = seedq(arm, 27, 33); OV[arm] = (e, h, d)
    note(f"   {arm:4s} seed E/Eopt {e:.3f}  Hamming {h:.2f}  Delta {d:.3g} (RMS {rms:.3f})  mean D {D.mean():+.3f}")
N, D, lo, hi = gcol("Q1", Nmax=33); d, rms = fit_delta(N, D, [10 ** LQ0[n] for n in N]); e, h = seedq("Q1", 27, 33)
OV["Q1"] = (e, h, d)
note(f"   Q1   seed E/Eopt {e:.3f}  Hamming {h:.2f}  Delta {d:.3g} (RMS {rms:.3f})  mean D {D.mean():+.3f}")
Np = np.arange(27, 34); dp = np.array([PD[n] for n in Np]); tp = np.array([10 ** F2[(n, "MTS")] for n in Np])
dpub, rpub = fit_delta(Np, dp, tp); OV["pub"] = (np.nan, np.nan, dpub)
note(f"   published QE-MTS (27-33): Delta {dpub:.3g} (RMS {rpub:.3f}), mean D {dp.mean():+.3f}")
# where the published overhead sits on the classical curve -> implied seed energy (log-log interpolation)
xs = np.array([OV[a][0] for a in ("A2k", "A2", "A2m", "A3")]); ys = np.log10([OV[a][2] for a in ("A2k", "A2", "A2m", "A3")])
o = np.argsort(ys); implied = np.interp(np.log10(dpub), ys[o], xs[o])
note(f"   seed energy implied for the published QE-MTS by the classical overhead curve: {implied:.2f} Eopt")
dq = gcol("Q1", Nmax=33)[1]
note(f"   Q1 vs published gap on N=27-33: RMS {np.sqrt(np.mean((dq-dp)**2)):.3f} dex, r={np.corrcoef(dq, dp)[0,1]:+.2f}, "
     f"mean {dq.mean():+.3f} vs {dp.mean():+.3f}")
note("   per-N D_Q1: " + "  ".join(f"{n}:{v:+.3f}" for n, v in zip(Np, dq)))
for arm in ("Q5", "PQ1_d8"):
    N, D, lo, hi = gcol(arm, Nmax=33); note(f"   {arm}: mean D 27-33 {D.mean():+.3f}; per N " + " ".join(f"{v:+.3f}" for v in D))
# classical overhead fits on 27-37 for the table (A2m-variants etc. are from summary_ext)
note("== DCQO seed equivalence: mean seed energy per N (Q1 vs best-of-n uniform)")
for n in range(27, 34):
    q = seedq("Q1", n, n)[0]; vals = {a: seedq(a, n, n)[0] for a in ("A2", "A2m", "A3")}
    note(f"   N={n}: Q1 {q:.3f}  A2 {vals['A2']:.3f}  A2m {vals['A2m']:.3f}  A3 {vals['A3']:.3f}")

# ---------------------------------------------------------------- Fig 1: gap of replicated vs informative seeds
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.35))
ax = axs[0]; shade(ax); zero(ax)
Ns = sorted(PD); ax.plot(Ns, [PD[n] for n in Ns], "s-", color=INK, ms=3.6, lw=1.3, label="published QE-MTS (DCQO seed ×100, [1])", zorder=4)
for arm, lab in (("A1", "A1: random string ×100"), ("A2", "A2: best of $10^5$ uniform ×100"),
                 ("A2m", "A2m: best of $10^6$ uniform ×100")):
    eb(ax, *tcol(arm, 23, 48), arm, lab)
eb(ax, *gcol("Q1", Nmax=33), "Q1", "Q1: best of $10^5$ DCQO shots ×100 (our DCQO)")
ax.set_xlabel("$N$"); ax.set_ylabel(r"$D=\log_{10}(\mathrm{TTS}_{\rm seeded}/\mathrm{TTS}_{\rm MTS})$")
ax.set_title("(a) replicated single seeds", loc="left"); ax.set_xlim(22.5, 48.5); ax.set_ylim(-0.25, 0.85)
leg(ax, 1, -0.2)
ax = axs[1]; shade(ax); zero(ax)
ax.plot(Ns, [PD[n] for n in Ns], "s-", color=INK, ms=3.6, lw=1.3, label="published QE-MTS", zorder=4)
eb(ax, *tcol("A5", 23, 42), "A5", "A5: 100 best distinct of $10^5$")
eb(ax, *tcol("A6", 27, 40), "A6", "A6: 100 best distinct of $10^7$")
eb(ax, *tcol("A7a", 27, 40), "A7a", "A7a: optimum, $N/4$ flips ×100")
eb(ax, *gcol("Q5", Nmax=33), "Q5v", "Q5: 100 best distinct DCQO shots")
ax.set_xlabel("$N$"); ax.set_title("(b) diverse or near-optimal seeds", loc="left"); ax.set_xlim(22.5, 42.5); ax.set_xticks([25, 30, 35, 40])
ax.set_ylim(-0.7, 0.6); leg(ax, 1, -0.2)
fig.tight_layout(); fig.savefig(f"{OUT}/figA1_gap.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: overhead vs seed quality (+ DCQO)
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]; shade(ax); zero(ax)
for arm, lab in (("A2k", "best of $10^3$ ×100"), ("A2", "best of $10^5$ ×100"), ("A2m", "best of $10^6$ ×100"),
                 ("A3", "best of $10^7$ ×100")):
    eb(ax, *tcol(arm, 27, 42), arm, lab)
eb(ax, *gcol("Q1", Nmax=33), "Q1", "best of $10^5$ DCQO shots ×100")
ax.plot(Ns, [PD[n] for n in Ns], "s-", color=INK, ms=3.6, lw=1.3, label="published QE-MTS", zorder=4)
ax.set_xlabel("$N$"); ax.set_ylabel("$D$"); ax.set_title("(a) replicated seeds of increasing quality", loc="left")
ax.set_xlim(26.5, 42.5); ax.set_ylim(-0.25, 1.15); leg(ax, 2, -0.2)
ax = axs[1]
for arm in ("A1", "A2k", "A2", "A2m", "A3", "A8"):
    e, h, d = OV[arm]; ax.scatter(e, d, s=26, color=COL[arm], marker=MK[arm], zorder=3, edgecolor="white", lw=0.6)
    ax.annotate(arm, (e, d), xytext={"A8": (4, -3), "A3": (-3, -11)}.get(arm, (4, -3)), textcoords="offset points", fontsize=7, color=INK)
e, h, d = OV["Q1"]; ax.scatter(e, d, s=80, color=COL["Q1"], marker="*", zorder=4, edgecolor="white", lw=0.6)
ax.annotate("DCQO (Q1)", (e, d), xytext=(-8, 6), textcoords="offset points", fontsize=7, color=INK, ha="right")
ax.axhline(dpub, color=INK, ls="--", lw=0.9); ax.text(7.5, dpub * 1.12, "published QE-MTS", fontsize=7, color=INK)
xx = [OV[a][0] for a in ("A1", "A2k", "A2", "A2m", "A3")]; yy = [OV[a][2] for a in ("A1", "A2k", "A2", "A2m", "A3")]
ax.plot(xx, yy, color=MUTED, lw=0.8, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xticks([8, 5, 3, 2, 1.5, 1.2]); ax.set_xticklabels(["8", "5", "3", "2", "1.5", "1.2"]); ax.minorticks_off()
ax.set_xlabel(r"replicated seed energy $E/E_{\rm opt}$ (mean, $N$=27–33)"); ax.set_ylabel(r"overhead $\Delta$ (evaluations)")
ax.set_title("(b) overhead vs seed quality", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figA2_seed_quality.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: how to use a seed
fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.45))
ax = axs[0]; shade(ax); zero(ax)
eb(ax, *gcol("A2m", Nmax=42), "A2m", "100 copies (QE-MTS protocol)")
for arm, lab in (("M2m_k50", "50 copies + 50 random"), ("M2m_k10", "10 copies + 90 random"),
                 ("M2m_k1", "1 copy + 99 random"), ("P2m_d8", "100 copies, $N/8$ flips each")):
    eb(ax, *gcol(arm), arm, lab, ms=3)
ax.set_title("(a) A2m seed (best of $10^6$)", loc="left"); ax.set_xlabel("$N$"); ax.set_ylabel("$D$")
ax.set_xlim(26.5, 42.5); ax.set_ylim(-0.25, 0.7); leg(ax, 1, -0.2, 6.0)
ax = axs[1]; shade(ax); zero(ax)
eb(ax, *gcol("A3", Nmax=42), "A3", "100 copies")
eb(ax, *gcol("M3_k1"), "M3_k1", "1 copy + 99 random", ms=3); eb(ax, *gcol("P3_d8"), "P3_d8", "100 copies, $N/8$ flips", ms=3)
ax.set_title("(b) A3 seed (best of $10^7$)", loc="left"); ax.set_xlabel("$N$"); ax.set_xlim(26.5, 42.5)
ax.set_ylim(-0.25, 1.15); leg(ax, 1, -0.2, 6.0)
ax = axs[2]; shade(ax); zero(ax)
eb(ax, *gcol("Q1", Nmax=33), "Q1", "100 copies (QE-MTS protocol)")
eb(ax, *gcol("PQ1_d8", Nmax=33), "PQ1_d8", "100 copies, $N/8$ flips", ms=3)
eb(ax, *gcol("Q5", Nmax=33), "Q5", "100 best distinct shots", ms=3)
ax.set_title("(c) DCQO seed (best of $10^5$ shots)", loc="left"); ax.set_xlabel("$N$"); ax.set_xlim(26.5, 33.5)
ax.set_ylim(-0.25, 1.0); leg(ax, 1, -0.2, 6.0)
fig.tight_layout(); fig.savefig(f"{OUT}/figA3_seed_usage.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: mechanism controls
fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.1), gridspec_kw=dict(width_ratios=[1.05, 1, 1.05]))
ax = axs[0]
rows = [("default", "A2m", "A0"), ("$K=50$", "H2m_K50", "H0_K50"), ("$K=200$", "H2m_K200", "H0_K200"),
        ("$p_{\\rm mut}=2/N$", "H2m_pm2", "H0_pm2"), ("tabu ×2", "H2m_tb2", "H0_tb2")]
HYP = []
for i, (lab, arm, ref) in enumerate(rows):
    N, D, lo, hi = gcol(arm, ref, Nmin=27, Nmax=37)
    d, rms = fit_delta(N, D, [10 ** LQ0[n] for n in N]) if ref == "A0" else (np.nan, np.nan)
    HYP.append((lab, D.mean())); ax.barh(i, D.mean(), height=0.55, color=COL["A2m"] if i == 0 else "#f3b293")
    ax.text(D.mean() + 0.01, i, f"{D.mean():+.2f}", va="center", fontsize=6.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.axvline(0, color=MUTED, lw=0.7); ax.set_xlim(0, 0.45)
ax.set_xlabel(r"mean $D$, $N$=27–37 (A2m seed vs same MTS)"); ax.set_title("(a) MTS settings", loc="left")
ax = axs[1]; shade(ax); zero(ax)
for arm, lab in (("I2", "ITS from A2 seed"), ("I2m", "ITS from A2m seed"), ("I3", "ITS from A3 seed")):
    eb(ax, *gcol(arm, "I0"), arm, lab, ms=3)
eb(ax, *gcol("A2m", Nmax=37), "A8", "MTS from A2m seed ×100 (for scale)", ms=3, alpha=0.6)
ax.set_xlabel("$N$"); ax.set_ylabel("$D$ vs same solver, random start"); ax.set_xlim(26.5, 37.5)
ax.set_ylim(-0.3, 0.7); ax.set_title("(b) no population: iterated tabu", loc="left"); leg(ax, 1, -0.22, 6.0)
ax = axs[2]
W = {}
with open(f"{EXT}/within_arm.csv") as f:
    for r in csv.DictReader(f): W.setdefault((r["arm"], r["variable"]), []).append(float(r["rho"]))
arms = ["A1", "A2k", "A2", "A2m", "A3", "A8", "Q1"]
for j, (var, c, lab) in enumerate((("energy", "#2a78d6", "seed energy"), ("hamming", "#eb6834", "Hamming to optimum"))):
    m = [np.mean(W[(a, var)]) for a in arms]; s = [np.std(W[(a, var)], ddof=1) / np.sqrt(len(W[(a, var)])) for a in arms]
    ax.errorbar(np.arange(len(arms)) + (j - 0.5) * 0.25, m, yerr=s, fmt="o", color=c, ms=3.6, elinewidth=0.8, label=lab)
ax.axhline(0, color=MUTED, lw=0.7); ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, fontsize=7)
ax.set_ylabel(r"Spearman $\rho$ with replicate TTS"); ax.set_title("(c) within-arm seed properties", loc="left")
leg(ax, 1, -0.16, 6.0)
fig.tight_layout(); fig.savefig(f"{OUT}/figA4_mechanism.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5: crossover test and TTS
LINES = {"A0": (1.168, 0.1475), "A2": (2.105, 0.1223), "A2m": (3.178, 0.0936)}     # comment_numbers.txt sec. 4
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.3))
ax = axs[0]; shade(ax); zero(ax)
x = np.arange(27, 49)
for arm in ("A2", "A2m"):
    a0, b0 = LINES["A0"]; a1, b1 = LINES[arm]
    ax.plot(x, (a1 + b1 * x) - (a0 + b0 * x), "--", color=COL[arm], lw=0.9, label=f"{arm}: straight lines on 27–37, extrapolated")
eb(ax, *tcol("A2", 27, 48), "A2", "A2 measured")
eb(ax, *gcol("A2m", Nmax=48), "A2m", "A2m measured")
ax.axvline(47, color=MUTED, ls=":", lw=0.9); ax.text(46.7, 0.62, "published\ncrossover\n$N_\\times\\approx47$", ha="right", fontsize=6.5, color=MUTED)
ax.set_xlabel("$N$"); ax.set_ylabel("$D$"); ax.set_ylim(-0.6, 0.75); ax.set_xlim(26.5, 48.5)
ax.set_title("(a) crossover test up to $N=48$", loc="left"); leg(ax, 2, -0.2)
ax = axs[1]; shade(ax)
NN = sorted(n for n in LQ0 if n >= 20)
odd = [n for n in NN if n % 2]; ev = [n for n in NN if not n % 2]
ax.plot(NN, [LQ0[n] for n in NN], color=MUTED, lw=0.8, zorder=1)
ax.scatter(odd, [LQ0[n] for n in odd], s=16, color="#e34948", zorder=3, label="MTS (A0), odd $N$")
ax.scatter(ev, [LQ0[n] for n in ev], s=16, color="#2a78d6", zorder=3, label="MTS (A0), even $N$")
ax.plot(x, LINES["A0"][0] + LINES["A0"][1] * x, "--", color=INK, lw=0.8, label="fit on 27–37, extrapolated")
pn = sorted(n for (n, m) in F2 if m == "MTS")
ax.scatter(pn, [F2[(n, "MTS")] for n in pn], marker="x", s=14, color=INK, lw=0.8, label="published MTS (Fig. 2 of [1])")
ax.set_xlabel("$N$"); ax.set_ylabel(r"$\log_{10}Q_{0.50}$ (objective evaluations)")
ax.set_title("(b) median TTS of MTS", loc="left"); leg(ax, 2, -0.2)
fig.tight_layout(); fig.savefig(f"{OUT}/figA5_crossover.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 6: seed cost included
fig, ax = plt.subplots(figsize=(3.45, 2.9)); shade(ax); zero(ax)
for arm, lab in (("A2", "A2 + $10^5$"), ("A2m", "A2m + $10^6$"), ("A5", "A5 + $10^5$"), ("A6", "A6 + $10^7$"), ("A3", "A3 + $10^7$")):
    eb(ax, *gcol(arm, cost=1, Nmin=27, Nmax=48), arm, lab, ms=3)
ax.set_xlabel("$N$"); ax.set_ylabel("$D$ with the seed cost counted"); ax.set_xlim(26.5, 48.5)
leg(ax, 3, -0.2, 6.0); ax.set_ylim(bottom=-0.2)
fig.tight_layout(); fig.savefig(f"{OUT}/figA6_seedcost.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig A-app: optima
OPT = list(csv.DictReader(open(f"{EXT}/optima.csv")))
fig, ax = plt.subplots(figsize=(3.45, 2.3))
for r in OPT:
    n = int(r["N"]); k = int(r["n_classes"]); s = int(r["n_skew"])
    ax.bar(n, k, color="#e34948" if n % 2 else "#2a78d6", width=0.7)
    if s: ax.text(n, k + 0.1, "s" if s == k else f"{s}s", ha="center", fontsize=6, color=INK)
ax.set_xlabel("$N$"); ax.set_ylabel("distinct optima reached\n(up to symmetry)"); ax.set_ylim(0, 7)
fig.tight_layout(); fig.savefig(f"{OUT}/figA7_optima.pdf"); plt.close(fig)

with open(f"{OUT}/article_numbers.txt", "w") as f: f.write("\n".join(NUM) + "\n")
print("done")
