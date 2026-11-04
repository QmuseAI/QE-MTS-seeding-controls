"""Fit-window sensitivity and gap-shape analysis of the published QE-MTS vs MTS data
(QMI 8:92, Fig. 2 and Fig. 5), using only values extracted by extract_figs.py.

usage:  python window_analysis.py [datadir] [outdir]
reads:  fig2_q50_digitized.csv, fig5_logratio_vector.csv
writes: window_table.csv, gap_models.txt, summary.txt, step1_figure.png

Quantities
  Q(N,m)  = median-of-medians TTS (Fig.2 black markers), m in {QE-MTS, MTS}
  D(N)    = log10 Q(N,QE-MTS) - log10 Q(N,MTS)    (from Fig.2)   and the Fig.5 bootstrap median of the same
  kappa   = exp(slope of ln Q vs N) over a window [a,b]     (paper: Eq. 9, window 27-37)
  kappa ratio QE/MTS = 10**slope(D)  -> paired fit: the common N-to-N hardness of the single LABS
                                       instance per N cancels in D, so its CI is the honest one.
Note: no per-replicate data are available, so CIs here are OLS t-intervals across N (they include the
N-to-N lack of fit, which the paper's replicate/seed bootstrap does not).
"""
import sys, os
import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = sys.argv[1] if len(sys.argv) > 1 else "../data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../results"
os.makedirs(OUT, exist_ok=True)
LN10 = np.log(10)
LINES = []
def out(s=""):
    print(s); LINES.append(s)

# ---------------- load ----------------
import csv
def load_csv(path):
    with open(path) as fh:
        rdr = csv.DictReader(l for l in fh if not l.startswith("#"))
        rows = list(rdr)
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try: cols[k] = np.array([float(v) for v in vals])
        except ValueError: cols[k] = np.array(vals)
    return cols
f2 = load_csv(os.path.join(DATA, "fig2_q50_digitized.csv"))
f2["N"] = f2["N"].astype(int)
Ns = np.array(sorted(set(f2["N"])))
LQ = {m: np.array([f2["log10_Q50"][(f2["N"] == N) & (f2["method"] == m)][0] for N in Ns]) for m in ("QE-MTS", "MTS")}
D2 = LQ["QE-MTS"] - LQ["MTS"]
f5 = load_csv(os.path.join(DATA, "fig5_logratio_vector.csv"))
assert np.all(f5["N"] == Ns)
D5 = f5["median_D"]

def ols(x, y):
    r = stats.linregress(x, y); n = len(x)
    t = stats.t.ppf(0.975, n - 2) if n > 2 else np.nan
    return r.slope, r.intercept, r.stderr, t, r.rvalue ** 2

# ---------------- A. reproduce the paper's fits (sanity check of the digitisation) ----------------
out("=" * 92)
out("A. Sanity check: OLS on the digitised Q0.50 markers, window 27-37 (paper: QE-MTS 1.24, MTS 1.37)")
for m in ("QE-MTS", "MTS"):
    b, a, se, t, r2 = ols(Ns, LQ[m] * LN10)
    out(f"   {m:7s} kappa = {np.exp(b):.3f}   OLS 95% CI [{np.exp(b-t*se):.3f}, {np.exp(b+t*se):.3f}]   R2 = {r2:.3f}")
out("   (paper's bootstrap CIs [1.23,1.25] / [1.36,1.37] ignore N-to-N lack of fit; R2 0.86-0.89 / 0.85-0.87)")

# ---------------- B. all windows ----------------
rows = []
for a0 in Ns:
    for b0 in Ns:
        if b0 - a0 < 4: continue
        sel = (Ns >= a0) & (Ns <= b0); x = Ns[sel]
        r = dict(Nmin=a0, Nmax=b0, n=sel.sum())
        for m, tag in (("QE-MTS", "QE"), ("MTS", "MTS")):
            b, a, se, t, r2 = ols(x, LQ[m][sel] * LN10)
            r[f"kappa_{tag}"] = np.exp(b); r[f"kappa_{tag}_lo"] = np.exp(b - t * se); r[f"kappa_{tag}_hi"] = np.exp(b + t * se)
            r[f"R2_{tag}"] = r2
        for D, tag in ((D2, "fig2"), (D5, "fig5")):
            b, a, se, t, r2 = ols(x, D[sel])
            r[f"ratio_{tag}"] = 10 ** b; r[f"ratio_{tag}_lo"] = 10 ** (b - t * se); r[f"ratio_{tag}_hi"] = 10 ** (b + t * se)
            r[f"p_slope_{tag}"] = 2 * stats.t.sf(abs(b / se), len(x) - 2)
            r[f"Ncross_{tag}"] = (-a / b) if b < 0 else np.inf      # where the fitted D line hits 0
        rows.append(r)
keys = list(rows[0])
with open(os.path.join(OUT, "window_table.csv"), "w") as f:
    f.write("# kappa_*: exp(OLS slope of ln Q50); ratio_*: kappa_QE/kappa_MTS from the paired fit of D (fig2 or fig5 medians)\n")
    f.write("# *_lo/_hi: OLS 95% t-interval; p_slope: two-sided p for slope(D)=0; Ncross: N where fitted D line reaches 0\n")
    f.write(",".join(keys) + "\n")
    for r in rows: f.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")

out()
out("=" * 92)
out("B. Window sensitivity (OLS on Q0.50; ratio = kappa_QE/kappa_MTS from paired fit of Fig.5 medians)")
out(f"{'window':>8} {'n':>2} | {'kappa QE [95% CI]':>22} | {'kappa MTS [95% CI]':>22} | {'ratio (Fig5) [95% CI]':>24} {'p':>7} | {'N_cross':>7}")
key_windows = [(27, 37), (28, 37), (29, 37), (30, 37), (31, 37), (32, 37), (33, 37), (27, 31), (27, 32), (32, 36)]
for (a0, b0) in key_windows:
    r = next(r for r in rows if r["Nmin"] == a0 and r["Nmax"] == b0)
    nc = r["Ncross_fig5"]
    out(f"{a0}-{b0:<5} {r['n']:>2} | {r['kappa_QE']:.3f} [{r['kappa_QE_lo']:.3f},{r['kappa_QE_hi']:.3f}]   "
        f"| {r['kappa_MTS']:.3f} [{r['kappa_MTS_lo']:.3f},{r['kappa_MTS_hi']:.3f}]   "
        f"| {r['ratio_fig5']:.4f} [{r['ratio_fig5_lo']:.4f},{r['ratio_fig5_hi']:.4f}] {r['p_slope_fig5']:7.3g} | "
        f"{('%.1f' % nc) if np.isfinite(nc) else 'never':>7}")

# leave-out: drop the three sizes where MTS is visibly low (28, 30, 31)
sel = ~np.isin(Ns, [28, 30, 31])
out()
out("   Drop N=28,30,31 (MTS anomalously fast there in Fig.2), window 27-37 otherwise:")
for m in ("QE-MTS", "MTS"):
    b, a, se, t, r2 = ols(Ns[sel], LQ[m][sel] * LN10)
    out(f"     {m:7s} kappa = {np.exp(b):.3f} [{np.exp(b-t*se):.3f},{np.exp(b+t*se):.3f}]  R2={r2:.3f}")
b, a, se, t, r2 = ols(Ns[sel], D5[sel])
out(f"     ratio (Fig5) = {10**b:.4f} [{10**(b-t*se):.4f},{10**(b+t*se):.4f}]  N_cross = {(-a/b) if b<0 else np.inf:.1f}")

# ---------------- C. shape of the gap D(N) ----------------
out()
out("=" * 92)
out("C. Shape of the gap D(N) = log10(TTS_QE/TTS_MTS)  (least squares, Gaussian residuals, AICc)")
x = Ns.astype(float); u = x - 27

def aicc(rss, n, k):
    k1 = k + 1                                   # + residual variance
    return n * np.log(rss / n) + 2 * k1 + 2 * k1 * (k1 + 1) / (n - k1 - 1)

def fit_models(D, label):
    n = len(D); res = {}
    # M1 linear (the paper's implicit model: D decreases linearly -> crosses 0)
    b, a, *_ = stats.linregress(u, D); pred = a + b * u
    res["M1 linear: D=a+b(N-27)"] = (2, np.sum((D - pred) ** 2), pred, dict(a=a, b=b, N_cross=27 - a / b if b < 0 else np.inf))
    # M2 additive overhead: TTS_QE = TTS_MTS + Delta(N), Delta/TTS_MTS = R rho^(N-27)  -> D>0 always
    f2m = lambda uu, R, rho: np.log10(1 + R * rho ** uu)
    p, _ = optimize.curve_fit(f2m, u, D, p0=[1.5, 0.8], bounds=([1e-6, 1e-3], [1e3, 2.0]))
    pred = f2m(u, *p)
    res["M2 overhead: D=log10(1+R*rho^(N-27))"] = (2, np.sum((D - pred) ** 2), pred, dict(R=p[0], rho=p[1], N_cross=np.inf))
    # M4 two levels with a fitted breakpoint (descriptive)
    best = None
    for bp in Ns[1:-1]:
        lo, hi = D[Ns < bp], D[Ns >= bp]
        pred = np.where(Ns < bp, lo.mean(), hi.mean()); rss = np.sum((D - pred) ** 2)
        if best is None or rss < best[1]: best = (bp, rss, pred, lo.mean(), hi.mean())
    res["M4 two-level step (breakpoint fitted)"] = (3, best[1], best[2], dict(break_at=best[0], level_low_N=best[3], level_high_N=best[4], N_cross=np.inf))
    out(f"  data: {label}")
    ac = {k: aicc(v[1], n, v[0]) for k, v in res.items()}; amin = min(ac.values())
    for k, (kp, rss, pred, par) in res.items():
        ps = ", ".join(f"{q}={w:.3g}" for q, w in par.items())
        out(f"    {k:42s} k={kp}  RMS={np.sqrt(rss/n):.4f}  dAICc={ac[k]-amin:6.2f}   {ps}")
    return res

R5 = fit_models(D5, "Fig.5 bootstrap medians (vector, precise)")
R2 = fit_models(D2, "Fig.2 Q0.50 markers (raster)")

# plateau test on the last sizes: is the gap still shrinking for N>=33?
out()
for a0 in (32, 33):
    sel = Ns >= a0
    b, a, se, t, r2 = ols(Ns[sel], D5[sel])
    out(f"  Fig.5 medians, N={a0}-37: slope of D = {b:+.4f} /N  [{b-t*se:+.4f},{b+t*se:+.4f}],  mean D = {D5[sel].mean():+.3f}"
        f"  -> TTS_QE/TTS_MTS = {10**D5[sel].mean():.2f}x;  linear extrapolation hits 0 at N = {(-a/b) if b<0 else np.inf:.1f}")

out()
out("  Fig.5 per-N: median D, KDE-body area with D<0, lowest bootstrap draw:")
for i, N in enumerate(Ns):
    out(f"    N={N}: median {D5[i]:+.3f}   P_body(D<0) = {f5['body_frac_below0'][i]:.2f}   min {f5['min_D'][i]:+.2f}")

# ---------------- D. tail N>=33: linear (crosses) vs geometric decay (never crosses) ----------------
out()
out("=" * 92)
out("D. Tail of the gap, N=33-37 only: two 2-parameter models that both fit 5 points, very different futures")
sel = Ns >= 33; ut = (Ns[sel] - 33).astype(float); Dt = D5[sel]
b, a, *_ = stats.linregress(ut, Dt); rl = Dt - (a + b * ut); tail_a, tail_b = a, b
pg, _ = optimize.curve_fit(lambda uu, A, r: A * r ** uu, ut, Dt, p0=[0.13, 0.85])
rg = Dt - pg[0] * pg[1] ** ut
out(f"   linear   D = {a:.4f} {b:+.4f}(N-33)         RMS {np.sqrt(np.mean(rl**2)):.4f}   -> D=0 at N = {33 - a/b:.1f}")
out(f"   geometric D = {pg[0]:.4f} * {pg[1]:.3f}^(N-33)    RMS {np.sqrt(np.mean(rg**2)):.4f}   -> D>0 for all N (overhead-type)")
out("   predicted D (log10 TTS_QE/TTS_MTS):")
out("     N    linear   geometric")
for N in range(38, 48):
    out(f"     {N}  {a + b*(N-33):+.3f}    {pg[0]*pg[1]**(N-33):+.3f}")
out("   -> the two hypotheses differ by >=0.05 dex from N~42 and ~0.1 dex by N~45, i.e. testable at or below the paper's N_x=47.")

# ---------------- E. what the paper's lines imply at N=47 ----------------
out()
out("=" * 92)
out("E. Extrapolation to N=47 (paper's crossover) from Q0.50 fits:")
for (a0, b0) in [(27, 37), (33, 37)]:
    sel = (Ns >= a0) & (Ns <= b0); v = {}
    for m in ("QE-MTS", "MTS"):
        b, a, *_ = ols(Ns[sel], LQ[m][sel]); v[m] = a + b * 47
    out(f"   window {a0}-{b0}: log10 Q50(47)  QE-MTS {v['QE-MTS']:.2f}  MTS {v['MTS']:.2f}   D(47) = {v['QE-MTS']-v['MTS']:+.2f}")

with open(os.path.join(OUT, "summary.txt"), "w") as f: f.write("\n".join(LINES) + "\n")

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
cQ, cM = "#1f9e89", "#e8684a"
for m, c in (("QE-MTS", cQ), ("MTS", cM)):
    ax[0].plot(Ns, LQ[m], "o", color=c, label=f"{m} Q0.50 (digitised Fig.2)")
    for (a0, b0), ls in (((27, 37), "--"), ((33, 37), "-")):
        sel = (Ns >= a0) & (Ns <= b0); b, a, *_ = ols(Ns[sel], LQ[m][sel])
        xx = np.array([a0, 47]); ax[0].plot(xx, a + b * xx, ls, color=c, lw=1.2,
                                              label=f"{m} fit {a0}-{b0}: kappa={np.exp(b*LN10):.2f}")
ax[0].axvline(47, color="0.6", lw=0.8, ls=":"); ax[0].text(46.6, ax[0].get_ylim()[0] + 0.1, "paper's N_x", ha="right", fontsize=8)
ax[0].set_xlabel("N"); ax[0].set_ylabel("log10 Q0.50 (objective evaluations)"); ax[0].legend(fontsize=7)
ax[0].set_title("Fitted kappa depends strongly on the N window")
ax[1].vlines(Ns, f5["body_bottom_D"], f5["body_top_D"], color="0.75", lw=6, label="Fig.5 KDE body")
ax[1].plot(Ns, D5, "ko", ms=5, label="Fig.5 median D")
ax[1].plot(Ns, D2, "x", color="m", ms=6, label="Fig.2: log10(Q_QE/Q_MTS)")
uu = np.linspace(0, 20, 300)
k1 = "M1 linear: D=a+b(N-27)"; k2 = "M2 overhead: D=log10(1+R*rho^(N-27))"; k4 = "M4 two-level step (breakpoint fitted)"
p1 = R5[k1][3]; ax[1].plot(27 + uu, p1["a"] + p1["b"] * uu, "--", color="0.3", lw=1, label=f"linear fit N=27-37 (D=0 at N={p1['N_cross']:.0f})")
ut2 = np.linspace(0, 14, 200)
ax[1].plot(33 + ut2, tail_a + tail_b * ut2, "--", color="tab:red", lw=1.2, label=f"linear fit N=33-37 (D=0 at N={33 - tail_a/tail_b:.0f})")
ax[1].plot(33 + ut2, pg[0] * pg[1] ** ut2, "-", color="tab:blue", lw=1.2, label="geometric decay fit N=33-37 (never crosses)")
p4 = R5[k4][3]; ax[1].step([27, p4["break_at"] - 0.5, 47], [p4["level_low_N"], p4["level_high_N"], p4["level_high_N"]],
                           where="post", color="tab:orange", lw=1, label=f"two-level step (break at N={p4['break_at']})")
ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlim(26.5, 47.5); ax[1].set_ylim(-0.3, 0.75)
ax[1].set_xlabel("N"); ax[1].set_ylabel("D = log10(TTS_QE-MTS / TTS_MTS)"); ax[1].legend(fontsize=7)
ax[1].set_title("Gap D(N): models agree on N<=37, disagree beyond")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "step1_figure.png"), dpi=150); plt.close(fig)
