"""Full analysis of the classical-seeding control experiment (paper-identical statistics + checks).

usage: python analyze.py --data RUN/data --out RUN/analysis [--ref A0] [--nboot 2000]
                         [--paper ../../step1_digitize/data/fig5_logratio_vector.csv]
                         [--seedstats ../seeds/seed_stats.csv]

Statistics follow Gomez Cadavid et al. (QMI 8:92) Sec. 3.3 exactly:
  per replicate: median over seeds; per (N, arm): quantiles Q_p of the replicate medians, p in {.05,.10,.50,.90,.95};
  log Q_p = alpha + beta N; kappa = exp(beta); two-stage bootstrap (replicates, then seeds), B draws;
  conservative crossover N_x = (alpha_ref,0.05 - alpha_arm,0.95) / (beta_arm,0.95 - beta_ref,0.05)   (their Eq. 10).
Added on top (because each N is a single LABS instance and the paper's bootstrap ignores N-to-N lack of fit):
  - OLS t-intervals for kappa (include lack of fit);
  - paired gap D_x(N) = log10 Q50(arm) - log10 Q50(ref) with bootstrap CIs, and kappa ratio = 10^slope(D);
  - fit-window sensitivity; tail models for D on N=33-37 (linear vs geometric) and their out-of-sample test at N>37;
  - direct test of every predicted crossover against the measured data beyond 37;
  - comparison with the published QE-MTS gap (Fig. 5 medians) if --paper is given;
  - aspiration on/off sensitivity (A0na vs A0, A2na vs A2) when present.
Censored runs (evals < 0: early-stopped by mts v3 or capped) are treated as +inf. The replicate medians and
their seed-level bootstrap are exact as long as they never fall on a censored value; this is checked and flagged.
"""
import argparse, os, sys, glob, re, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ref", default="A0"); ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--paper", default=""); ap.add_argument("--seedstats", default="")
ap.add_argument("--paperfig2", default="", help="step1 fig2_q50_digitized.csv: compare our A0 with the paper's MTS")
ap.add_argument("--window", default="27-37", help="paper fit window")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
W0, W1 = map(int, a.window.split("-"))
PQ = np.array([0.05, 0.10, 0.50, 0.90, 0.95]); iQ = {p: i for i, p in enumerate(PQ)}
LN10 = np.log(10)
rng = np.random.default_rng(12345)
TXT = []
def out(s=""):
    print(s); TXT.append(s)

# ---------------------------------------------------------------- load
files = glob.glob(os.path.join(a.data, "*_N*.txt"))
arms = sorted({re.match(r"(.+)_N(\d+)\.txt", os.path.basename(f)).group(1) for f in files},
              key=lambda s: (s != a.ref, s))
Y, NS = {}, {}
for f in files:
    m = re.match(r"(.+)_N(\d+)\.txt", os.path.basename(f)); arm, N = m.group(1), int(m.group(2))
    d = np.loadtxt(f, ndmin=2)
    if d.size == 0: continue
    reps = {}
    for r, s, e in d[:, :3]: reps.setdefault(int(r), []).append(e)
    S = min(len(v) for v in reps.values())
    M = np.array([np.array(v[:S], float) for _, v in sorted(reps.items())])
    M = np.where(M < 0, np.inf, np.maximum(M, 1.0))
    Y[(arm, N)] = M
for arm in arms: NS[arm] = np.array(sorted(N for (aa, N) in Y if aa == arm))
assert a.ref in arms, f"reference arm {a.ref} not found"

def fit(x, y):
    r = stats.linregress(x, y); t = stats.t.ppf(0.975, len(x) - 2)
    return r.slope, r.intercept, r.stderr, t, r.rvalue ** 2

def qstats(M):
    med = np.median(M, axis=1)
    return np.quantile(med, PQ)

def boot(M, B):
    R, S = M.shape; outq = np.empty((B, len(PQ)))
    for b0 in range(0, B, 250):
        b1 = min(B, b0 + 250); nb = b1 - b0
        ir = rng.integers(0, R, (nb, R)); isd = rng.integers(0, S, (nb, R, S))
        med = np.median(M[ir[:, :, None], isd], axis=2)
        outq[b0:b1] = np.quantile(med, PQ, axis=1).T
    return outq

out("=" * 100)
out(f"arms: {arms}   reference: {a.ref}   bootstrap draws: {a.nboot}")
LQ, LQB = {}, {}            # log10 quantiles (point) and bootstrap draws
rows_perN = []
for (arm, N), M in sorted(Y.items()):
    q = qstats(M); qb = boot(M, a.nboot)
    LQ[(arm, N)] = np.log10(q); LQB[(arm, N)] = np.log10(qb)
    ncap = int(np.isinf(M).sum())                                  # censored runs (expected with early stop)
    nbad = int(np.isinf(np.median(M, axis=1)).sum())               # replicate medians that are censored
    fbad = float(np.mean(~np.isfinite(qb[:, -1])))                 # bootstrap draws with Q0.95 censored
    rows_perN.append(dict(arm=arm, N=N, R=M.shape[0], S=M.shape[1], censored=ncap, cens_rep_medians=nbad,
                          **{f"Q{int(p*100):02d}": q[i] for i, p in enumerate(PQ)},
                          Q50_lo=10 ** np.percentile(np.log10(qb[:, 2]), 2.5),
                          Q50_hi=10 ** np.percentile(np.log10(qb[:, 2]), 97.5)))
    if nbad or fbad > 0:
        out(f"WARNING {arm} N={N}: {nbad} of {M.shape[0]} replicate medians censored; "
            f"Q0.95 censored in {100*fbad:.2f}% of bootstrap draws (raise --cap / --kfrac)")
with open(os.path.join(a.out, "perN_quantiles.csv"), "w") as f:
    k = list(rows_perN[0]); f.write(",".join(k) + "\n")
    for r in rows_perN: f.write(",".join(f"{r[x]:.6g}" if isinstance(r[x], float) else str(r[x]) for x in k) + "\n")


def ols_rows(x, Yb):
    """OLS slope/intercept of every row of Yb (B x n) against x; non-finite rows give nan/inf, no exception"""
    x = np.asarray(x, float); xc = x - x.mean()
    with np.errstate(invalid="ignore"):
        sl = (Yb @ xc) / (xc @ xc); ic = Yb.mean(1) - sl * x.mean()
    return sl, ic

def slope_boot(arm, Ns, qi):
    Yb = np.stack([LQB[(arm, N)][:, qi] for N in Ns], 1) * LN10        # B x n
    return ols_rows(Ns, Yb)                                            # slopes (B,), intercepts (B,)

# ---------------------------------------------------------------- 0. is our MTS the paper's MTS?
if a.paperfig2 and os.path.exists(a.paperfig2):
    out(); out("=" * 100)
    out(f"0. Implementation check: our {a.ref} vs the paper's MTS (digitised Fig.2 Q0.50), log10 difference per N")
    pm = {}
    with open(a.paperfig2) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            if r["method"] == "MTS": pm[int(r["N"])] = float(r["log10_Q50"])
    Nc = [N for N in sorted(pm) if (a.ref, N) in LQ]
    dd = np.array([LQ[(a.ref, N)][2] - pm[N] for N in Nc])
    out("   " + "  ".join(f"N{N}:{d:+.2f}" for N, d in zip(Nc, dd)))
    out(f"   mean offset {dd.mean():+.3f} dex (x{10**dd.mean():.2f}), sd {dd.std(ddof=1):.3f} dex, corr of log Q50 over N: "
        f"{np.corrcoef([LQ[(a.ref, N)][2] for N in Nc], [pm[N] for N in Nc])[0,1]:.3f}")
    kp = np.exp(fit(np.array(Nc), np.array([pm[N] for N in Nc]) * LN10)[0])
    ko = np.exp(fit(np.array(Nc), np.array([LQ[(a.ref, N)][2] for N in Nc]) * LN10)[0])
    out(f"   kappa on N={min(Nc)}-{max(Nc)}: ours {ko:.3f}, paper's MTS {kp:.3f}")

# ---------------------------------------------------------------- 1. kappa per arm & window
out(); out("=" * 100)
out("1. Scaling base kappa of Q0.50 (also Q0.10, Q0.90 in kappa_table.csv)")
out("   bootCI = paper's two-stage bootstrap (no N-to-N lack of fit); olsCI = t-interval incl. lack of fit")
kt = []
def windows_for(arm):
    Ns = NS[arm]; w = [(W0, W1), (Ns.min(), W1), (33, W1)]
    if Ns.max() > W1: w += [(W1 + 1, Ns.max()), (Ns.min(), Ns.max()), (W0, Ns.max())]
    return [(lo, hi) for lo, hi in w if ((Ns >= lo) & (Ns <= hi)).sum() >= 5]
for arm in arms:
    for (lo, hi) in windows_for(arm):
        Ns = NS[arm][(NS[arm] >= lo) & (NS[arm] <= hi)]
        for p in (0.10, 0.50, 0.90):
            qi = iQ[p]; y = np.array([LQ[(arm, N)][qi] for N in Ns]) * LN10
            if not np.all(np.isfinite(y)): continue
            b, c, se, t, r2 = fit(Ns, y); sb, _ = slope_boot(arm, Ns, qi); sb = sb[np.isfinite(sb)]
            row = dict(arm=arm, window=f"{lo}-{hi}", n=len(Ns), q=p, kappa=np.exp(b),
                       boot_lo=np.exp(np.percentile(sb, 2.5)), boot_hi=np.exp(np.percentile(sb, 97.5)),
                       ols_lo=np.exp(b - t * se), ols_hi=np.exp(b + t * se), R2=r2)
            kt.append(row)
            if p == 0.50:
                out(f"   {arm:5s} {lo}-{hi:<3} n={len(Ns):2d}  kappa={row['kappa']:.3f}  bootCI[{row['boot_lo']:.3f},{row['boot_hi']:.3f}]"
                    f"  olsCI[{row['ols_lo']:.3f},{row['ols_hi']:.3f}]  R2={r2:.3f}")
with open(os.path.join(a.out, "kappa_table.csv"), "w") as f:
    k = list(kt[0]); f.write(",".join(k) + "\n")
    for r in kt: f.write(",".join(f"{r[x]:.5g}" if isinstance(r[x], float) else str(r[x]) for x in k) + "\n")

# ---------------------------------------------------------------- 2. paired gap D and kappa ratio
out(); out("=" * 100)
out(f"2. Paired gap D_x(N) = log10 Q50(x) - log10 Q50({a.ref})  [95% bootstrap CI]")
paper = {}
if a.paper and os.path.exists(a.paper):
    with open(a.paper) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")): paper[int(r["N"])] = float(r["median_D"])
D, DB = {}, {}
others = [x for x in arms if x != a.ref and not x.endswith("na")]
allN = sorted(set().union(*[set(NS[x]) for x in arms]))
hdr = "   N   " + "".join(f"{x:>24s}" for x in others) + ("   paper(QE-MTS)" if paper else "")
out(hdr)
for N in allN:
    line = f"   {N:<4}"
    for x in others:
        if (x, N) in LQ and (a.ref, N) in LQ:
            d = LQ[(x, N)][2] - LQ[(a.ref, N)][2]; db = LQB[(x, N)][:, 2] - LQB[(a.ref, N)][:, 2]
            D[(x, N)] = d; DB[(x, N)] = db
            line += f"  {d:+.3f} [{np.percentile(db,2.5):+.3f},{np.percentile(db,97.5):+.3f}]"
        else: line += f"{'-':>24s}"
    if paper: line += f"   {paper[N]:+.3f}" if N in paper else "        -"
    out(line)

out(); out("   kappa ratio kappa_x/kappa_ref = 10^slope(D) on windows  [OLS 95% CI] (boot 95% CI)")
for x in others:
    for (lo, hi) in [(W0, W1), (33, W1)] + ([(W1 + 1, NS[x].max()), (W0, NS[x].max())] if NS[x].max() > W1 else []):
        Ns = np.array([N for N in allN if lo <= N <= hi and (x, N) in D])
        if len(Ns) < 5: continue
        y = np.array([D[(x, N)] for N in Ns]); b, c, se, t, _ = fit(Ns, y)
        Yb = np.stack([DB[(x, N)] for N in Ns], 1); sb = ols_rows(Ns, Yb)[0]; sb = sb[np.isfinite(sb)]
        out(f"   {x:5s} {lo}-{hi:<3} ratio={10**b:.4f} [{10**(b-t*se):.4f},{10**(b+t*se):.4f}] "
            f"({10**np.percentile(sb,2.5):.4f},{10**np.percentile(sb,97.5):.4f})  p(slope=0)={2*stats.t.sf(abs(b/se),len(Ns)-2):.2g}")
if paper:
    Np = np.array(sorted(paper)); b, c, se, t, _ = fit(Np, np.array([paper[N] for N in Np]))
    out(f"   paper QE-MTS {Np.min()}-{Np.max()} ratio={10**b:.4f} [{10**(b-t*se):.4f},{10**(b+t*se):.4f}]  (from Fig.5 medians)")
    out(); out("   similarity of D_x(N) to the published QE-MTS gap on the paper's N range:")
    for x in others:
        Nc = [N for N in Np if (x, N) in D]
        if len(Nc) < 5: continue
        dx = np.array([D[(x, N)] for N in Nc]); dp = np.array([paper[N] for N in Nc])
        out(f"   {x:5s}: RMS(D_x - D_paper) = {np.sqrt(np.mean((dx-dp)**2)):.3f} dex   Pearson r = {np.corrcoef(dx,dp)[0,1]:+.2f}"
            f"   mean D_x = {dx.mean():+.3f} vs paper {dp.mean():+.3f}")

# ---------------------------------------------------------------- 3. crossover predictions and direct tests
out(); out("=" * 100)
out(f"3. Crossover predicted from the {W0}-{W1} window, and what the data beyond {W1} say")
cross_rows = []
for x in others:
    Nw = np.array([N for N in NS[x] if W0 <= N <= W1 and N in NS[a.ref]])
    if len(Nw) < 5: continue
    # paper Eq. 10 (Q0.95 of arm vs Q0.05 of ref), per bootstrap draw
    bx, ax_ = slope_boot(x, Nw, iQ[0.95]); br, ar = slope_boot(a.ref, Nw, iQ[0.05])
    with np.errstate(divide="ignore", invalid="ignore"):
        Nx = (ar - ax_) / (bx - br)
    Nx = np.where(bx < br, Nx, np.inf)
    # median-level crossing (Q0.50 lines)
    b50x, a50x = slope_boot(x, Nw, iQ[0.50]); b50r, a50r = slope_boot(a.ref, Nw, iQ[0.50])
    with np.errstate(divide="ignore", invalid="ignore"):
        N50 = np.where(b50x < b50r, (a50r - a50x) / (b50x - b50r), np.inf)
    frac_cross = np.mean(np.isfinite(Nx))
    fin = Nx[np.isfinite(Nx)]
    s = (f"   {x:5s}: paper-style N_x (Q95 vs Q05): median {np.median(Nx):.1f}  95% CI [{np.percentile(Nx,2.5):.1f},"
         f"{np.percentile(Nx,97.5):.1f}]  (finite in {100*frac_cross:.0f}% of draws);  Q50-line crossing: median {np.median(N50):.1f}")
    out(s)
    beyond = [N for N in NS[x] if N > W1 and (x, N) in D]
    if beyond:
        for N in beyond:
            db = DB[(x, N)]; q95x = LQ[(x, N)][iQ[0.95]]; q05r = LQ[(a.ref, N)][iQ[0.05]]
            verdict = "D<0 (arm faster)" if np.percentile(db, 97.5) < 0 else ("D>0 (arm slower)" if np.percentile(db, 2.5) > 0 else "D~0")
            out(f"        N={N}: D={D[(x,N)]:+.3f} [{np.percentile(db,2.5):+.3f},{np.percentile(db,97.5):+.3f}] {verdict:17s}"
                f" | Q95(arm) < Q05(ref)? {'YES' if q95x < q05r else 'no'}")
            cross_rows.append(dict(arm=x, N=N, D=D[(x, N)], D_lo=np.percentile(db, 2.5), D_hi=np.percentile(db, 97.5),
                                   pred_Nx_median=np.median(Nx), pred_N50_median=np.median(N50),
                                   bulk_crossed=int(q95x < q05r)))
if cross_rows:
    with open(os.path.join(a.out, "crossover_tests.csv"), "w") as f:
        k = list(cross_rows[0]); f.write(",".join(k) + "\n")
        for r in cross_rows: f.write(",".join(f"{r[q]:.5g}" if isinstance(r[q], float) else str(r[q]) for q in k) + "\n")

# ---------------------------------------------------------------- 4. tail models: fit 33-37, predict beyond
out(); out("=" * 100)
out(f"4. Tail of D_x: fit on 33-{W1}, linear (crosses 0) vs geometric decay (never crosses); out-of-sample check")
for x in others:
    Nt = np.array([N for N in range(33, W1 + 1) if (x, N) in D]); Nb = np.array([N for N in NS[x] if N > W1 and (x, N) in D])
    if len(Nt) < 4: continue
    y = np.array([D[(x, N)] for N in Nt]); u = Nt - 33.0
    bl, al, *_ = stats.linregress(u, y)
    try:
        pg, _ = optimize.curve_fit(lambda uu, A, r: A * r ** uu, u, y, p0=[max(y[0], 1e-3), 0.85], bounds=([-10, 0.0], [10, 2.0]))
    except Exception: pg = (np.nan, np.nan)
    msg = f"   {x:5s}: linear D=0 at N={33 - al/bl if bl < 0 else np.inf:.1f};  geometric A={pg[0]:+.3f} r={pg[1]:.3f}"
    if len(Nb):
        yb = np.array([D[(x, N)] for N in Nb]); ub = Nb - 33.0
        el = np.sqrt(np.mean((yb - (al + bl * ub)) ** 2)); eg = np.sqrt(np.mean((yb - pg[0] * pg[1] ** ub) ** 2))
        msg += f"  | out-of-sample RMS on N={Nb.min()}-{Nb.max()}: linear {el:.3f}, geometric {eg:.3f}"
    out(msg)

# ---------------------------------------------------------------- 5. aspiration sensitivity
pairs = [(x + "na", x) for x in arms if x + "na" in arms]
if pairs:
    out(); out("=" * 100); out("5. Tabu aspiration on/off (paper Alg.2 as printed has none; App. F text says it has)")
    for na, wa in pairs:
        Nc = [N for N in NS[na] if N in NS[wa]]
        dd = np.array([LQ[(na, N)][2] - LQ[(wa, N)][2] for N in Nc])
        kna = np.exp(fit(np.array(Nc), np.array([LQ[(na, N)][2] for N in Nc]) * LN10)[0])
        kwa = np.exp(fit(np.array(Nc), np.array([LQ[(wa, N)][2] for N in Nc]) * LN10)[0])
        out(f"   {na} vs {wa} (N={min(Nc)}-{max(Nc)}): kappa {kna:.3f} vs {kwa:.3f}; mean log10 Q50 shift {dd.mean():+.3f}")
    if ("A0na" in arms) and ("A2na" in arms):
        Nc = [N for N in NS["A2na"] if N in NS["A0na"] and W0 <= N <= W1]
        y = np.array([LQ[("A2na", N)][2] - LQ[("A0na", N)][2] for N in Nc]); b, c, se, t, _ = fit(np.array(Nc), y)
        out(f"   without aspiration: kappa ratio A2na/A0na = {10**b:.4f} [{10**(b-t*se):.4f},{10**(b+t*se):.4f}]")

# ---------------------------------------------------------------- 6. seed quality
if a.seedstats and os.path.exists(a.seedstats):
    out(); out("=" * 100); out("6. Seed quality (mean over replicates): best seed energy / Eopt, Hamming distance of best seed to known optimum")
    import collections
    acc = collections.defaultdict(list)
    with open(a.seedstats) as fh:
        for r in csv.DictReader(fh): acc[(r["arm"], int(r["N"]))].append((float(r["seedE_min"]), float(r["ham_bestseed_to_opt"])))
    from labs_common import EOPT
    for arm in sorted({k[0] for k in acc}):
        Ns = sorted(N for (aa, N) in acc if aa == arm)
        s = "   " + f"{arm:4s}" + " ".join(f"N{N}:{np.mean([v[0] for v in acc[(arm,N)]])/EOPT[N]:.2f}/{np.mean([v[1] for v in acc[(arm,N)]]):.1f}"
                                    for N in Ns if N in (27, 32, 37, 42, 47, 50))
        out(s)

with open(os.path.join(a.out, "summary.txt"), "w") as f: f.write("\n".join(TXT) + "\n")

# ---------------------------------------------------------------- figures
cols = plt.cm.tab10(np.linspace(0, 1, 10))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for i, arm in enumerate(arms):
    Ns = NS[arm]; y = np.array([LQ[(arm, N)][2] for N in Ns])
    lo = np.array([np.percentile(LQB[(arm, N)][:, 2], 2.5) for N in Ns]); hi = np.array([np.percentile(LQB[(arm, N)][:, 2], 97.5) for N in Ns])
    ax[0].errorbar(Ns + 0.08 * (i - len(arms) / 2), y, [y - lo, hi - y], fmt="o-", ms=3, lw=0.8, capsize=2, color=cols[i % 10], label=arm)
ax[0].axvspan(W0, W1, color="0.92", zorder=0); ax[0].set_xlabel("N"); ax[0].set_ylabel("log10 Q0.50 (objective evaluations)")
ax[0].legend(fontsize=7, ncol=2); ax[0].set_title("Median-of-medians TTS (95% two-stage bootstrap)")
for i, x in enumerate(others):
    Ns = np.array([N for N in allN if (x, N) in D]); y = np.array([D[(x, N)] for N in Ns])
    lo = np.array([np.percentile(DB[(x, N)], 2.5) for N in Ns]); hi = np.array([np.percentile(DB[(x, N)], 97.5) for N in Ns])
    ax[1].errorbar(Ns + 0.08 * i, y, [y - lo, hi - y], fmt="o-", ms=3, lw=0.8, capsize=2, color=cols[(arms.index(x)) % 10], label=f"{x} vs {a.ref}")
if paper:
    Np = sorted(paper); ax[1].plot(Np, [paper[N] for N in Np], "ks--", ms=5, label="paper: QE-MTS vs MTS (Fig.5)")
ax[1].axhline(0, color="k", lw=0.6); ax[1].axvspan(W0, W1, color="0.92", zorder=0)
ax[1].set_xlabel("N"); ax[1].set_ylabel(f"D = log10 Q50(arm) - log10 Q50({a.ref})"); ax[1].legend(fontsize=7)
ax[1].set_title("Paired gap; paper's window shaded")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_scaling_and_gap.png"), dpi=150); plt.close(fig)
print(f"\nwrote {a.out}/summary.txt, perN_quantiles.csv, kappa_table.csv, crossover_tests.csv, fig_scaling_and_gap.png")
