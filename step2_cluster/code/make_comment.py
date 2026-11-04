"""Numbers, tables and figures for the Comment on Gomez Cadavid et al., QMI 8:92 (QE-MTS for LABS).

usage: python make_comment.py --run RUN_DIR --step1 STEP1_DATA_DIR --out OUT_DIR [--analysis DIR]
                              [--seedstats seeds/seed_stats_all.csv] [--nboot 2000]
  RUN_DIR       step2_cluster/run_main          (needs data/*.txt; kappa_table.csv in --analysis)
  STEP1_DATA    step1_digitize/data             (fig5_logratio_vector.csv, fig2_q50_digitized.csv)

Statistics are those of analyze.py (replicate median over seeds, quantiles over replicates, two-stage
bootstrap; censored runs = +inf). On top of analyze.py this script
  1. writes a de-duplicated kappa table and a per-N table for the paper (A0, A1, A2, A5, A6, A7a, paper);
  2. pooled test of the gap beyond the paper's window: mean D over N = 38..48 with a bootstrap CI;
  3. one-parameter additive-overhead model D(N) = log10(1 + Delta / T_ref(N)), fitted on N = 27..37 to
     (a) our A2 vs A0 and (b) the paper's QE-MTS vs MTS (digitised Fig. 2 medians, Fig. 5 gap);
  4. Fig. 1 (gap), Fig. 2 (extrapolated vs measured crossover), Fig. 3 (absolute TTS, N = 20..48);
  5. if the seed-quality arms (A2k, A2m, A3, A8) are present: overhead Delta vs seed energy, Fig. 4;
  6. replicate-level spread of the gap compared with the violins of the paper's Fig. 5 (Fig. S1).
Small N where a seeded arm can contain the optimum itself (TTS = 1 in some replicates) are left out of the
figures (N < 23).
"""
import argparse, os, sys, glob, re, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--step1", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--analysis", default="", help="analysis folder with kappa_table.csv (default RUN/analysis_extra)")
ap.add_argument("--seedstats", default="", help="seed_stats csv (seed energy per arm, N, replicate)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(20261022)
REF, B = "A0", a.nboot
W0, W1 = 27, 37                    # paper's fit window
TAIL = (38, 48)                    # beyond the paper's window
LOG = []
def say(s=""):
    print(s); LOG.append(s)

# ------------------------------------------------------------------ load raw TTS
Y = {}
for f in glob.glob(os.path.join(a.run, "data", "*_N*.txt")):
    m = re.match(r"(.+)_N(\d+)\.txt", os.path.basename(f)); arm, N = m.group(1), int(m.group(2))
    d = np.loadtxt(f, ndmin=2)
    if d.size == 0: continue
    reps = {}
    for r, s, e in d[:, :3]: reps.setdefault(int(r), []).append(e)
    S = min(len(v) for v in reps.values())
    M = np.array([np.array(v[:S], float) for _, v in sorted(reps.items())])
    Y[(arm, N)] = np.where(M < 0, np.inf, np.maximum(M, 1.0))
NS = {}
for (arm, N) in Y: NS.setdefault(arm, []).append(N)
NS = {k: np.array(sorted(v)) for k, v in NS.items()}

def boot_q50(M):
    """two-stage bootstrap of log10 Q0.50 (median over replicates of the median over seeds)"""
    R, S = M.shape; outq = np.empty(B)
    for b0 in range(0, B, 250):
        b1 = min(B, b0 + 250); nb = b1 - b0
        ir = rng.integers(0, R, (nb, R)); isd = rng.integers(0, S, (nb, R, S))
        med = np.median(M[ir[:, :, None], isd], axis=2)
        outq[b0:b1] = np.median(med, axis=1)
    with np.errstate(divide="ignore"):
        return np.log10(outq)

L, LB, CENS = {}, {}, {}
for key, M in sorted(Y.items()):
    rm = np.median(M, axis=1)
    L[key] = np.log10(np.median(rm)); LB[key] = boot_q50(M)
    CENS[key] = int(np.isinf(rm).sum())

# ------------------------------------------------------------------ paper data (step 1)
P5 = {}
with open(os.path.join(a.step1, "fig5_logratio_vector.csv")) as fh:
    for r in csv.DictReader(l for l in fh if not l.startswith("#")):
        P5[int(r["N"])] = dict(D=float(r["median_D"]), lo=float(r["body_bottom_D"]), hi=float(r["body_top_D"]),
                               min=float(r["min_D"]), fneg=float(r["body_frac_below0"]))
P2 = {}
with open(os.path.join(a.step1, "fig2_q50_digitized.csv")) as fh:
    for r in csv.DictReader(l for l in fh if not l.startswith("#")):
        P2[(r["method"], int(r["N"]))] = float(r["Q50_evals"])
NP = sorted(P5)

# ------------------------------------------------------------------ 1. gap table
def gap(arm, N):
    d = L[(arm, N)] - L[(REF, N)]
    db = LB[(arm, N)] - LB[(REF, N)]
    db = db[np.isfinite(db)]
    return d, np.percentile(db, 2.5), np.percentile(db, 97.5)

SWEEP = [x for x in ("A2k", "A2m", "A3", "A8") if any((x, N) in L for N in range(27, 38))]
ARMS = ["A1", "A2", "A5", "A6", "A7a"] + SWEEP
rows = []
for N in range(20, 49):
    if (REF, N) not in L: continue
    row = dict(N=N, log10_Q50_A0=L[(REF, N)], cens_rep_medians_A0=CENS[(REF, N)])
    for arm in ARMS:
        if (arm, N) in L:
            d, lo, hi = gap(arm, N); row[f"D_{arm}"] = d; row[f"D_{arm}_lo"] = lo; row[f"D_{arm}_hi"] = hi
    if N in P5: row["D_paper"] = P5[N]["D"]
    rows.append(row)
keys = ["N", "log10_Q50_A0", "cens_rep_medians_A0"] + [f"D_{x}{s}" for x in ARMS for s in ("", "_lo", "_hi")] + ["D_paper"]
with open(os.path.join(a.out, "table_gap_perN.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows: f.write(",".join(("%.4f" % r[k] if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in keys) + "\n")

say("=" * 90)
say("1. Paired gap D = log10 Q50(arm) - log10 Q50(A0), 95% two-stage bootstrap CI")
for r in rows:
    if r["N"] < 27: continue
    s = f"  N={r['N']:2d}  A0 log10Q50={r['log10_Q50_A0']:.2f}" + (f" (cens.rep.med {r['cens_rep_medians_A0']})" if r['cens_rep_medians_A0'] else "")
    for arm in ("A1", "A2", "A5") + tuple(SWEEP):
        if f"D_{arm}" in r: s += f"  {arm} {r[f'D_{arm}']:+.3f}[{r[f'D_{arm}_lo']:+.3f},{r[f'D_{arm}_hi']:+.3f}]"
    if "D_paper" in r: s += f"  paper {r['D_paper']:+.3f}"
    say(s)

# ------------------------------------------------------------------ 2. pooled test beyond the window
say(); say("=" * 90)
say(f"2. Pooled gap over N={TAIL[0]}-{TAIL[1]} (mean of per-N D; bootstrap over the independent per-N draws)")
POOL = {}
for arm in ("A1", "A2", "A5") + tuple(SWEEP):
    Ns = [N for N in range(TAIL[0], TAIL[1] + 1) if (arm, N) in L and (REF, N) in L]
    if len(Ns) < 3: continue
    dpt = np.mean([L[(arm, N)] - L[(REF, N)] for N in Ns])
    db = np.mean([LB[(arm, N)] - LB[(REF, N)] for N in Ns], axis=0)
    db = db[np.isfinite(db)]
    lo, hi = np.percentile(db, [2.5, 97.5]); p_le0 = float(np.mean(db <= 0))
    npos = sum(L[(arm, N)] > L[(REF, N)] for N in Ns)
    POOL[arm] = dict(Ns=f"{Ns[0]}-{Ns[-1]}", n=len(Ns), mean=dpt, lo=lo, hi=hi, frac_boot_le0=p_le0, n_pos=npos)
    say(f"  {arm}: N={Ns[0]}-{Ns[-1]} (n={len(Ns)})  mean D = {dpt:+.3f} [{lo:+.3f},{hi:+.3f}]  "
        f"(x{10**dpt:.2f});  bootstrap P(mean D <= 0) = {p_le0:.4f};  D>0 at {npos}/{len(Ns)} N")
    nneg = [N for N in Ns if np.percentile((LB[(arm, N)] - LB[(REF, N)])[np.isfinite(LB[(arm, N)] - LB[(REF, N)])], 97.5) < 0]
    say(f"       N with D significantly < 0 (upper 95% bound < 0): {nneg if nneg else 'none'}")

# ------------------------------------------------------------------ 3. additive-overhead model
say(); say("=" * 90)
say("3. One-parameter additive overhead: T_x(N) = T_ref(N) + Delta  <=>  D(N) = log10(1 + Delta/T_ref(N))")
def fit_delta(T, D):
    T, D = np.asarray(T, float), np.asarray(D, float)
    f = lambda ld: np.sum((np.log10(1 + 10 ** ld / T) - D) ** 2)
    r = optimize.minimize_scalar(f, bounds=(2, 9), method="bounded")
    pred = np.log10(1 + 10 ** r.x / T)
    return 10 ** r.x, np.sqrt(np.mean((pred - D) ** 2)), pred
def fit_lin(N, D):
    s = stats.linregress(N, D); pred = s.intercept + s.slope * np.asarray(N)
    return s, np.sqrt(np.mean((pred - D) ** 2))
AD = {}
Nw = np.arange(W0, W1 + 1)
# (a) our A2 (and A1)
SEEDE = {}
if a.seedstats and os.path.exists(a.seedstats):
    import collections
    from labs_common import EOPT
    acc = collections.defaultdict(list)
    with open(a.seedstats) as fh:
        for r in csv.DictReader(fh):
            if W0 <= int(r["N"]) <= W1:
                acc[r["arm"]].append((float(r["seedE_min"]) / EOPT[int(r["N"])], float(r["ham_bestseed_to_opt"])))
    SEEDE = {k: (np.mean([v[0] for v in vv]), np.mean([v[1] for v in vv])) for k, vv in acc.items()}
for arm in ("A2", "A1") + tuple(SWEEP):
    T = np.array([10 ** L[(REF, N)] for N in Nw]); D = np.array([L[(arm, N)] - L[(REF, N)] for N in Nw])
    dl, rms, pred = fit_delta(T, D); sl, rms_lin = fit_lin(Nw, D)
    perN = T * (10 ** D - 1)
    # out of sample on 38-48
    Nt = np.array([N for N in range(TAIL[0], TAIL[1] + 1) if (arm, N) in L])
    Tt = np.array([10 ** L[(REF, N)] for N in Nt]); Dt = np.array([L[(arm, N)] - L[(REF, N)] for N in Nt])
    rms_out_add = np.sqrt(np.mean((np.log10(1 + dl / Tt) - Dt) ** 2))
    rms_out_lin = np.sqrt(np.mean((sl.intercept + sl.slope * Nt - Dt) ** 2))
    AD[arm] = dict(Delta=dl, rms=rms, rms_lin=rms_lin, perN=perN.tolist(), rms_out_add=rms_out_add,
                   rms_out_lin=rms_out_lin, lin_zero=-sl.intercept / sl.slope, meanD=float(D.mean()),
                   kratio=10 ** sl.slope, r_paper=float(np.corrcoef(D, [P5[N]["D"] for N in Nw])[0, 1]),
                   seedE=SEEDE.get(arm, (np.nan, np.nan))[0], seedH=SEEDE.get(arm, (np.nan, np.nan))[1])
    say(f"  {arm} vs A0, N={W0}-{W1}: seed E/Eopt {AD[arm]['seedE']:.2f}, Hamming to optimum {AD[arm]['seedH']:.1f}; "
        f"mean D {D.mean():+.3f}, kappa ratio {10**sl.slope:.4f}, r(paper) {AD[arm]['r_paper']:+.2f}")
    say(f"       Delta = {dl:.3g} evals, RMS {rms:.3f} dex (1 param);  linear D(N): RMS {rms_lin:.3f} (2 params), "
        f"crosses 0 at N={-sl.intercept/sl.slope:.1f}")
    say(f"       per-N Delta = T_ref(10^D - 1): " + " ".join(f"{N}:{v:.2g}" for N, v in zip(Nw, perN)))
    say(f"       out of sample N={TAIL[0]}-{TAIL[1]}: RMS additive {rms_out_add:.3f}, linear {rms_out_lin:.3f}")
# (b) the paper: Fig.2 medians (T_MTS, T_QE) and Fig.5 gap
Tm = np.array([P2[("MTS", N)] for N in NP]); Tq = np.array([P2[("QE-MTS", N)] for N in NP])
for lab, D in (("Fig.5 gap", np.array([P5[N]["D"] for N in NP])), ("Fig.2 medians", np.log10(Tq / Tm))):
    dl, rms, pred = fit_delta(Tm, D); sl, rms_lin = fit_lin(np.array(NP), D)
    AD["paper_" + ("fig5" if "5" in lab else "fig2")] = dict(Delta=dl, rms=rms, rms_lin=rms_lin, lin_zero=-sl.intercept / sl.slope,
                                                             perN=(Tm * (10 ** D - 1)).tolist())
    say(f"  paper QE-MTS vs MTS ({lab}): Delta = {dl:.3g} evals, RMS {rms:.3f} dex (1 param);  linear: RMS {rms_lin:.3f} (2 params), "
        f"crosses 0 at N={-sl.intercept/sl.slope:.1f}")
    say(f"       per-N Delta: " + " ".join(f"{N}:{v:.2g}" for N, v in zip(NP, Tm * (10 ** D - 1))))
dq = AD["paper_fig2"]["perN"]
say(f"  paper per-N Delta (Fig.2): N=27-31 median {np.median(dq[:5]):.2g}, N=33-37 median {np.median(dq[6:]):.2g}; "
    f"ratio Delta_QE/Delta_A2 = {AD['paper_fig5']['Delta']/AD['A2']['Delta']:.1f}")
# paper's prediction at N=47 under the additive model, using OUR A0 T(47) rescaled by the A0/paper offset
off = np.mean([L[(REF, N)] - np.log10(P2[("MTS", N)]) for N in NP])
for N in (40, 44, 47, 48):
    Tp = 10 ** (L[(REF, N)] - off)
    say(f"  additive model with the paper's Delta: predicted QE-MTS gap at N={N}: D = {np.log10(1 + AD['paper_fig5']['Delta']/Tp):+.4f}"
        f"  (T_MTS ~ {Tp:.2g}, from our A0 shifted by {-off:+.2f} dex)")

# ------------------------------------------------------------------ 4. extrapolations from the window
say(); say("=" * 90)
say("4. Q50 straight lines on N=27-37 (paper's method) extrapolated, vs measurement")
def line(arm, lo=W0, hi=W1):
    Ns = np.arange(lo, hi + 1); s = stats.linregress(Ns, [L[(arm, N)] for N in Ns]); return s
EXT = {}
for arm in ("A0", "A2") + tuple(x for x in SWEEP if x == "A2m"):
    s = line(arm); EXT[arm] = s
    say(f"  {arm}: log10 Q50 = {s.intercept:.3f} + {s.slope:.4f} N   (kappa {10**s.slope:.3f})")
for arm in [x for x in EXT if x != "A0"]:
    x0 = (EXT[arm].intercept - EXT["A0"].intercept) / (EXT["A0"].slope - EXT[arm].slope)
    say(f"  {arm} and A0 lines cross at N = {x0:.1f}")
    for N in range(38, 49):
        if (arm, N) in L:
            pr = EXT[arm].intercept - EXT["A0"].intercept + (EXT[arm].slope - EXT["A0"].slope) * N
            say(f"     N={N}: predicted D_{arm} = {pr:+.3f}, measured {L[(arm, N)] - L[(REF, N)]:+.3f}")
for N in (40, 44, 47, 48):
    pr = EXT["A2"].intercept + EXT["A2"].slope * N - (EXT["A0"].intercept + EXT["A0"].slope * N)
    say(f"  N={N}: predicted D_A2 = {pr:+.3f}, measured {L[('A2', N)] - L[(REF, N)]:+.3f}")
for N in (40, 44, 47, 48):
    pr = EXT["A0"].intercept + EXT["A0"].slope * N
    say(f"  N={N}: A0 line predicts log10 Q50 = {pr:.2f}, measured {L[(REF, N)]:.2f}  (diff {L[(REF, N)] - pr:+.2f} dex)")
pm = stats.linregress(NP, [np.log10(P2[("MTS", N)]) for N in NP]); pq = stats.linregress(NP, [np.log10(P2[("QE-MTS", N)]) for N in NP])
xp = (pq.intercept - pm.intercept) / (pm.slope - pq.slope)
say(f"  paper (Fig.2 Q50 lines): MTS kappa {10**pm.slope:.3f}, QE kappa {10**pq.slope:.3f}, lines cross at N = {xp:.1f}")
oddeven = [(N, L[(REF, N)]) for N in range(38, 49)]
say("  A0 log10 Q50, N=38-48: " + " ".join(f"{N}:{v:.2f}" for N, v in oddeven))
up = [N for N in range(39, 48) if L[(REF, N)] > L[(REF, N - 1)] and L[(REF, N)] > L[(REF, N + 1)]]
say(f"  N in 39-47 that are local maxima of A0 TTS: {up}")
jumps = [abs(L[(REF, N + 1)] - L[(REF, N)]) for N in range(38, 48)]
say(f"  largest |log10 Q50(N+1) - log10 Q50(N)| on 38-48: {max(jumps):.2f} dex")
s = stats.linregress(range(27, 49), [L[(REF, N)] for N in range(27, 49)])
say(f"  A0 kappa on 27-48 (Q50 OLS): {10**s.slope:.3f}  R2={s.rvalue**2:.2f}")

# ------------------------------------------------------------------ 5. de-duplicated kappa table
kin = os.path.join(a.analysis or os.path.join(a.run, "analysis_extra"), "kappa_table.csv")
if os.path.exists(kin):
    with open(kin) as fh: lines = fh.read().splitlines()
    seen, outl = set(), []
    for l in lines:
        if l not in seen: seen.add(l); outl.append(l)
    with open(os.path.join(a.out, "kappa_table_dedup.csv"), "w") as f: f.write("\n".join(outl) + "\n")
    say(); say(f"5. kappa_table: {len(lines)-1} rows -> {len(outl)-1} after removing exact duplicates")

say(); say("6. Censoring notes")
for (arm, N), c in sorted(CENS.items()):
    if c: say(f"  {arm} N={N}: {c}/100 replicate medians censored (Q50 exact, upper quantiles are lower bounds)")
missing = [N for N in range(43, 46) if ("A5", N) not in L]
if missing: say(f"  A5 not run for N={missing}; A5 is reported for N<=42 only")

# ------------------------------------------------------------------ 7. replicate-level spread vs paper Fig. 5
say(); say("=" * 90)
say("7. Spread of the gap over replicates: log10(median_r(arm)/median_r'(A0)) over all replicate pairs (r, r'),")
say("   compared with the violins of the paper's Fig. 5 (body = KDE body, frac<0 = body area with D<0)")
SPREAD = {}
SP_ARMS = ["A2"] + [x for x in SWEEP if x == "A2m"]
for arm in SP_ARMS:
    for N in NP:
        if (arm, N) not in Y: continue
        ma = np.log10(np.median(Y[(arm, N)], 1)); mb = np.log10(np.median(Y[(REF, N)], 1))
        pairs = (ma[:, None] - mb[None, :]).ravel(); pairs = pairs[np.isfinite(pairs)]
        SPREAD[(arm, N)] = dict(q05=np.percentile(pairs, 5), q50=np.median(pairs), q95=np.percentile(pairs, 95),
                                lo=pairs.min(), frac_neg=float(np.mean(pairs < 0)))
for N in NP:
    s_ = f"  N={N}: paper body [{P5[N]['lo']:+.2f},{P5[N]['hi']:+.2f}] min {P5[N].get('min', np.nan):+.2f} frac<0 {P5[N]['fneg']:.2f}"
    for arm in SP_ARMS:
        if (arm, N) in SPREAD:
            v = SPREAD[(arm, N)]
            s_ += f" | {arm} 5-95% [{v['q05']:+.2f},{v['q95']:+.2f}] min {v['lo']:+.2f} frac<0 {v['frac_neg']:.2f}"
    say(s_)

with open(os.path.join(a.out, "comment_numbers.txt"), "w") as f: f.write("\n".join(LOG) + "\n")
json.dump(dict(pool=POOL, additive={k: {kk: vv for kk, vv in v.items()} for k, v in AD.items()}),
          open(os.path.join(a.out, "comment_numbers.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})
C = dict(paper="#000000", A2="#D55E00", A1="#0072B2", A5="#009E73", A6="#CC79A7", A7a="#7F7F7F", A0="#0072B2",
         A2k="#E69F00", A2m="#B5651D", A3="#8B1A1A", A8="#56B4E9")
MK = dict(paper="s", A2="o", A1="^", A5="D", A6="v", A7a="P", A2k="<", A2m=">", A3="h", A8="X")
LAB = dict(paper="published QE-MTS (DCQO seed ×100)", A2="A2: best of 1e5 uniform ×100",
           A1="A1: one random string ×100", A5="A5: 100 best distinct of 1e5",
           A6="A6: 100 best distinct of 1e7", A7a="A7a: optimum with N/4 flips ×100",
           A2k="A2k: best of 1e3 ×100", A2m="A2m: best of 1e6 ×100", A3="A3: best of 1e7 ×100",
           A8="A8: tabu-walk low-energy seed ×100")

def plot_gap(ax, arm, Nlo=23, Nhi=48, dx=0.0, ms=3.6):
    Ns = [N for N in range(Nlo, Nhi + 1) if (arm, N) in L]
    g = np.array([gap(arm, N) for N in Ns])
    ax.errorbar(np.array(Ns) + dx, g[:, 0], yerr=[g[:, 0] - g[:, 1], g[:, 2] - g[:, 0]], fmt=MK[arm] + "-",
                color=C[arm], ms=ms, lw=1.0, elinewidth=0.7, capsize=0, label=LAB[arm], zorder=3)

# Fig. 1: gap in and beyond the paper's window
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
ax.axvspan(W0 - 0.5, W1 + 0.5, color="0.93", zorder=0)
ax.fill_between(NP, [P5[N]["lo"] for N in NP], [P5[N]["hi"] for N in NP], color="0.75", alpha=0.45, lw=0, zorder=1)
ax.plot(NP, [P5[N]["D"] for N in NP], MK["paper"] + "-", color=C["paper"], ms=3.6, lw=1.2, label=LAB["paper"], zorder=4)
STRONG = [x for x in ("A2m",) if x in SWEEP]                # classical replicated seed that matches the published gap
for x in STRONG: plot_gap(ax, x, Nhi=42, dx=-0.15)
plot_gap(ax, "A2", dx=0.0); plot_gap(ax, "A1", dx=0.15)
ax.axhline(0, color="0.3", lw=0.6)
ax.set_xlabel("N"); ax.set_ylabel(r"$D=\log_{10}\,[\mathrm{TTS}_{\rm seeded}/\mathrm{TTS}_{\rm MTS}]$")
ax.set_title("(a) replicated single seed", loc="left", fontsize=9)
ax.set_xlim(22.5, 48.5); ax.set_ylim(-0.35, 0.75 if not STRONG else 0.85)
ax.legend(loc="upper right", fontsize=7, handlelength=1.6)
ax = axs[1]
ax.axvspan(W0 - 0.5, W1 + 0.5, color="0.93", zorder=0)
ax.plot(NP, [P5[N]["D"] for N in NP], MK["paper"] + "-", color=C["paper"], ms=3.6, lw=1.2, zorder=4, label="published QE-MTS")
plot_gap(ax, "A5", Nhi=42, dx=0.0); plot_gap(ax, "A6", dx=0.15); plot_gap(ax, "A7a", dx=-0.15)
ax.axhline(0, color="0.3", lw=0.6)
ax.set_xlabel("N"); ax.set_title("(b) diverse or near-optimal seeds", loc="left", fontsize=9)
ax.set_xlim(22.5, 45.5); ax.set_ylim(-0.75, 0.6)
ax.legend(loc="lower right", fontsize=7, handlelength=1.6)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig1_gap.pdf")); fig.savefig(os.path.join(a.out, "fig1_gap.png"))
plt.close(fig)

# Fig. 2: extrapolated crossover vs measurement (A2, and A2m if run)
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.axvspan(W0 - 0.5, W1 + 0.5, color="0.93", zorder=0)
Nx = np.arange(W0, 49)
hs, ls_ = [], []
for arm, nhi in [("A2", 48)] + [("A2m", 42)] * ("A2m" in EXT):
    lin = EXT[arm].intercept - EXT["A0"].intercept + (EXT[arm].slope - EXT["A0"].slope) * Nx
    h1, = ax.plot(Nx, lin, "--", color=C[arm], lw=1.0)
    plot_gap(ax, arm, Nlo=27, Nhi=nhi, dx=0.0 if arm == "A2" else 0.15, ms=3.0)
    hs.append(ax.containers[-1]); ls_.append(f"{arm} measured (95% CI)")
    hs.append(h1); ls_.append(f"{arm}: straight lines on N=27–37, extrapolated")
h2, = ax.plot(Nx, np.log10(1 + AD["A2"]["Delta"] / np.array([10 ** L[(REF, N)] for N in Nx])), ":", color="0.25", lw=1.2)
hs.append(h2); ls_.append(r"A2: constant overhead $\Delta$ (fit 27–37)")
ax.axhline(0, color="0.3", lw=0.6)
ax.axvline(47, color="0.4", lw=0.7, ls=(0, (1, 2)), zorder=1)
ax.text(46.8, 0.66, "published\ncrossover\n$N_\\times\\approx47$", fontsize=6.5, ha="right", va="top", color="0.25")
ax.set_xlabel("N"); ax.set_ylabel(r"$D$")
ax.set_xlim(26.5, 48.5); ax.set_ylim(-0.45, 0.70 if "A2m" in EXT else 0.40)
ax.legend(hs, ls_, loc="lower left", fontsize=5.8, handlelength=1.8)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig2_crossover.pdf")); fig.savefig(os.path.join(a.out, "fig2_crossover.png"))
plt.close(fig)

# Fig. 3: absolute TTS, N = 20-48
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.axvspan(W0 - 0.5, W1 + 0.5, color="0.93", zorder=0)
for arm, dx in (("A0", 0.0), ("A2", 0.15)):
    Ns = [N for N in range(20, 49) if (arm, N) in L]
    if arm == "A2": Ns = [N for N in Ns if N >= 23]
    y = np.array([L[(arm, N)] for N in Ns]); lb = np.array([np.percentile(LB[(arm, N)], [2.5, 97.5]) for N in Ns])
    ax.errorbar(np.array(Ns) + dx, y, yerr=[y - lb[:, 0], lb[:, 1] - y], fmt=("o" if arm == "A0" else "s") + "-",
                color=(C["A0"] if arm == "A0" else C["A2"]), ms=3.0, lw=0.9, elinewidth=0.7,
                label=("A0: MTS, random population" if arm == "A0" else "A2: best of 1e5 ×100"), zorder=3)
ax.plot(NP, [np.log10(P2[("MTS", N)]) for N in NP], "x", color="0.2", ms=4, label="published MTS (Fig. 2)")
ax.plot(NP, [np.log10(P2[("QE-MTS", N)]) for N in NP], "+", color="0.2", ms=5, label="published QE-MTS (Fig. 2)")
ax.plot(Nx, EXT["A0"].intercept + EXT["A0"].slope * Nx, "--", color=C["A0"], lw=0.8, label="A0 fit on N=27–37, extrapolated")
ax.set_xlabel("N"); ax.set_ylabel(r"$\log_{10} Q_{0.50}$  (objective evaluations)")
ax.set_xlim(19.5, 48.5)
ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig3_tts.pdf")); fig.savefig(os.path.join(a.out, "fig3_tts.png"))
plt.close(fig)
# Fig. 4: seed-quality sweep of the replicated protocol
if SWEEP:
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    ax.axvspan(W0 - 0.5, W1 + 0.5, color="0.93", zorder=0)
    ax.plot(NP, [P5[N]["D"] for N in NP], MK["paper"] + "-", color=C["paper"], ms=3.6, lw=1.2, zorder=4, label="published QE-MTS")
    for i, x in enumerate([y for y in ("A2k", "A2", "A2m", "A3") if y == "A2" or y in SWEEP]):
        plot_gap(ax, x, Nlo=27, Nhi=42, dx=0.1 * (i - 1.5), ms=3.0)          # A8 (off scale) only in panel (b)
    ax.axhline(0, color="0.3", lw=0.6); ax.set_xlabel("N"); ax.set_ylabel(r"$D$")
    ax.set_title("(a) replicated seeds of increasing quality", loc="left", fontsize=9)
    ax.set_xlim(26.5, 42.5); ax.legend(loc="upper right", fontsize=6.3, handlelength=1.6)
    ax = axs[1]
    for x in ("A1", "A2k", "A2", "A2m", "A3", "A8"):
        if x in AD and np.isfinite(AD[x]["seedE"]):
            ax.plot(AD[x]["seedE"], AD[x]["Delta"], MK[x], color=C[x], ms=6, zorder=3)
            ax.annotate(x, (AD[x]["seedE"], AD[x]["Delta"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.axhline(AD["paper_fig5"]["Delta"], color=C["paper"], lw=1.0, ls="--")
    ax.text(0.03, AD["paper_fig5"]["Delta"] * 1.12, "published QE-MTS", transform=ax.get_yaxis_transform(), ha="left", fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(FixedLocator([1.2, 1.5, 2, 3, 5, 8])); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xlabel(r"replicated seed energy $E/E_{\rm opt}$ (mean, N=27–37)"); ax.set_ylabel(r"overhead $\Delta$ (evaluations)")
    ax.invert_xaxis(); ax.set_title("(b) overhead vs seed quality", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig4_seed_quality.pdf")); fig.savefig(os.path.join(a.out, "fig4_seed_quality.png"))
    plt.close(fig)

# Fig. S1: spread of the gap over replicates vs the paper's Fig. 5 violins
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axs[0]
ax.fill_between(NP, [P5[N]["lo"] for N in NP], [P5[N]["hi"] for N in NP], color="0.75", alpha=0.6, lw=0, label="published QE-MTS (Fig. 5 body)")
ax.plot(NP, [P5[N]["D"] for N in NP], "s-", color=C["paper"], ms=3, lw=1.0)
for x in [y for y in SP_ARMS if (y, NP[0]) in SPREAD]:
    Ns = [N for N in NP if (x, N) in SPREAD]
    ax.fill_between(Ns, [SPREAD[(x, N)]["q05"] for N in Ns], [SPREAD[(x, N)]["q95"] for N in Ns], color=C[x], alpha=0.25, lw=0,
                    label=f"{x}: 5–95% over replicate pairs")
    ax.plot(Ns, [SPREAD[(x, N)]["q50"] for N in Ns], MK[x] + "-", color=C[x], ms=3, lw=1.0)
ax.axhline(0, color="0.3", lw=0.6); ax.set_xlabel("N"); ax.set_ylabel(r"$D$ per replicate"); ax.set_ylim(-1.2, 1.6)
ax.set_title("(a) spread over replicates", loc="left", fontsize=9); ax.legend(fontsize=6.3, loc="upper right")
ax = axs[1]
ax.plot(NP, [P5[N]["fneg"] for N in NP], "s-", color=C["paper"], ms=3.5, lw=1.0, label="published QE-MTS")
for x in [y for y in SP_ARMS if (y, NP[0]) in SPREAD]:
    Ns = [N for N in NP if (x, N) in SPREAD]
    ax.plot(Ns, [SPREAD[(x, N)]["frac_neg"] for N in Ns], MK[x] + "-", color=C[x], ms=3.5, lw=1.0, label=x)
ax.set_xlabel("N"); ax.set_ylabel("fraction with $D<0$"); ax.set_ylim(0, 0.6)
ax.set_title("(b) seeded replicates faster than MTS", loc="left", fontsize=9); ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "figS1_replicate_spread.pdf")); fig.savefig(os.path.join(a.out, "figS1_replicate_spread.png"))
plt.close(fig)
print("wrote", a.out)
