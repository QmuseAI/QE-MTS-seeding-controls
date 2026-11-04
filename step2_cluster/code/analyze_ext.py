"""Analysis of the extension runs, together with run_main.

usage: python analyze_ext.py --data ../run_main/data ../run_ext/data --out ../run_ext/analysis_ext \
           --seedstats ../seeds/seed_stats_all.csv [../seeds/seed_stats_ext.csv ../seeds/seed_stats_dcqo.csv] \
           [--paper ../../step1_digitize/data/fig5_logratio_vector.csv] [--nboot 2000]

Statistics as in analyze.py / make_comment.py: replicate medians over seeds, Q50 over replicates, two-stage bootstrap
(replicates, then seeds), paired gap D_x(N) = log10 Q50(x) - log10 Q50(ref), kappa ratio = 10^(OLS slope of D on
N=27-37), additive overhead D = log10(1 + Delta/T_ref). Censored runs (evals < 0) are +inf.
Sections (each is skipped if its arms are missing):
  0 inventory and censoring      1 validation of mts4 (V0 = A0, V2m = A2m line by line)
  2 seed cost included in TTS    3 seed properties within an arm (replicate-level correlations)
  4 odd/even hardness and the optima found (degeneracy, skew symmetry)
  5 how to use a seed (k copies + random strings, perturbed copies)
  6 MTS hyperparameters          7 population-free iterated tabu search
  8 A2m up to N=48 (crossover test with the arm that matches the published curve)
  9 DCQO seeds (gpu/dcqo_seeds.py)
Outputs: summary_ext.txt, gaps_ext.csv, within_arm.csv, optima.csv, fig_ext_*.png
"""
import argparse, os, sys, glob, re, csv
import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labs_common import EOPT, RLE, rle_to_spins, spins_to_int, energy_ints

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True, help="data folders; later folders add (arm, N) files")
ap.add_argument("--out", required=True)
ap.add_argument("--seedstats", nargs="*", default=[])
ap.add_argument("--paper", default="")
ap.add_argument("--nboot", type=int, default=2000)
ap.add_argument("--ref", default="A0")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(20261023)
W0, W1 = 27, 37
TXT = []
def out(s=""):
    print(s); TXT.append(s)
def hdr(s):
    out(); out("=" * 110); out(s)

# ------------------------------------------------------------------ load
Y, RAW = {}, {}          # (arm, N) -> R x S matrix of TTS (inf = censored);  raw int array (r, s, evals, E[, cfg])
for d in a.data:
    for f in sorted(glob.glob(os.path.join(d, "*_N*.txt"))):
        m = re.match(r"(.+)_N(\d+)\.txt$", os.path.basename(f))
        if not m: continue
        arm, N = m.group(1), int(m.group(2))
        if os.path.getsize(f) < 4: continue
        try:
            d0 = np.loadtxt(f, dtype=np.int64, ndmin=2)
        except ValueError:
            print(f"skip unreadable {f}"); continue
        if d0.size == 0: continue
        d0 = d0[np.lexsort((d0[:, 1], d0[:, 0]))]
        reps = {}
        for row in d0: reps.setdefault(int(row[0]), []).append(row[2])
        S = min(len(v) for v in reps.values())
        M = np.array([np.array(v[:S], float) for _, v in sorted(reps.items())])
        Y[(arm, N)] = np.where(M < 0, np.inf, np.maximum(M, 1.0)); RAW[(arm, N)] = d0
ARMS = sorted({k[0] for k in Y})
def Ns(arm, lo=0, hi=99): return sorted(N for (aa, N) in Y if aa == arm and lo <= N <= hi)
def has(arm): return any(k[0] == arm for k in Y)
assert has(a.ref), "reference arm A0 not found"

def boot_q50(M, B):
    R, S = M.shape; outq = np.empty(B)
    for b0 in range(0, B, 250):
        b1 = min(B, b0 + 250); nb = b1 - b0
        ir = rng.integers(0, R, (nb, R)); isd = rng.integers(0, S, (nb, R, S))
        med = np.median(M[ir[:, :, None], isd], axis=2)
        outq[b0:b1] = np.quantile(med, 0.5, axis=1)
    return outq

LQ, LQB, MED = {}, {}, {}
for k, M in Y.items():
    med = np.median(M, axis=1); MED[k] = med
    LQ[k] = np.log10(np.quantile(med, 0.5))
    with np.errstate(divide="ignore"):
        LQB[k] = np.log10(boot_q50(M, a.nboot))

def lq_cost(k, c):                     # log10 Q50 of (replicate median + seed cost c), point and bootstrap
    if c == 0: return LQ[k], LQB[k]
    return np.log10(10 ** LQ[k] + c), np.log10(10 ** LQB[k] + c)

def gap(arm, ref, N, cost=0.0):
    p, b = lq_cost((arm, N), cost); d = p - LQ[(ref, N)]; db = b - LQB[(ref, N)]
    db = db[np.isfinite(db)]
    lo, hi = (np.percentile(db, [2.5, 97.5]) if len(db) else (np.nan, np.nan))
    return d, lo, hi, db

def common(arm, ref, lo=0, hi=99):
    return [N for N in Ns(arm, lo, hi) if (ref, N) in Y]

def mean_gap(arm, ref, lo, hi, cost_fn=None):
    NN = common(arm, ref, lo, hi)
    if not NN: return None
    pts, draws = [], []
    for N in NN:
        d, _, _, db = gap(arm, ref, N, cost_fn(N) if cost_fn else 0.0)
        pts.append(d); draws.append(rng.choice(db, a.nboot) if len(db) else np.full(a.nboot, np.nan))
    mb = np.nanmean(np.array(draws), axis=0)
    return dict(n=len(NN), N=NN, mean=float(np.mean(pts)), lo=float(np.nanpercentile(mb, 2.5)),
                hi=float(np.nanpercentile(mb, 97.5)), p_le0=float(np.mean(mb <= 0)))

def kappa_ratio(arm, ref, lo=W0, hi=W1):
    NN = common(arm, ref, lo, hi)
    if len(NN) < 4: return None
    D = np.array([LQ[(arm, N)] - LQ[(ref, N)] for N in NN])
    r = stats.linregress(NN, D); t = stats.t.ppf(0.975, len(NN) - 2)
    return 10 ** r.slope, 10 ** (r.slope - t * r.stderr), 10 ** (r.slope + t * r.stderr), r.pvalue

def overhead(arm, ref, lo=W0, hi=W1, cost_fn=None):
    NN = common(arm, ref, lo, hi)
    if len(NN) < 3: return None
    D = np.array([gap(arm, ref, N, cost_fn(N) if cost_fn else 0.0)[0] for N in NN])
    T = np.array([10 ** LQ[(ref, N)] for N in NN])
    f = lambda L: np.sum((D - np.log10(1 + 10 ** L / T)) ** 2)
    res = optimize.minimize_scalar(f, bounds=(0, 13), method="bounded")
    return 10 ** res.x, float(np.sqrt(res.fun / len(NN)))

GAPROWS = []
def report(arm, ref, label="", cost_fn=None, lo=W0, hi=W1, tail=(38, 48)):
    if not has(arm) or not common(arm, ref): out(f"   {arm:9s}: not available"); return None
    kr = kappa_ratio(arm, ref, lo, hi); mg = mean_gap(arm, ref, lo, hi, cost_fn)
    mt = mean_gap(arm, ref, tail[0], tail[1], cost_fn); ov = overhead(arm, ref, lo, hi, cost_fn)
    s = f"   {arm:9s} vs {ref:7s} {label:28s}"
    if mg: s += f" mean D {lo}-{hi} {mg['mean']:+.3f} [{mg['lo']:+.3f},{mg['hi']:+.3f}]"
    if kr: s += f"  kappa ratio {kr[0]:.3f} [{kr[1]:.3f},{kr[2]:.3f}]"
    if ov: s += f"  Delta {ov[0]:.2e} (RMS {ov[1]:.3f})"
    if mt: s += f"  | mean D {mt['N'][0]}-{mt['N'][-1]} (n={mt['n']}) {mt['mean']:+.3f} [{mt['lo']:+.3f},{mt['hi']:+.3f}]"
    out(s)
    for N in common(arm, ref):
        d, l, h, _ = gap(arm, ref, N, cost_fn(N) if cost_fn else 0.0)
        GAPROWS.append(dict(arm=arm, ref=ref, cost=int(cost_fn(N)) if cost_fn else 0, N=N, D=d, lo=l, hi=h))
    return dict(kr=kr, mg=mg, mt=mt, ov=ov)

def perN_line(arm, ref, lo=0, hi=99, cost_fn=None):
    return "  ".join(f"{N}:{gap(arm, ref, N, cost_fn(N) if cost_fn else 0)[0]:+.3f}" for N in common(arm, ref, lo, hi))

# seed statistics
SEED = {}                                   # (arm, N, r) -> (seedE_min, seedE_mean, ham_best, ham_min, n_distinct)
for f in a.seedstats:
    if not os.path.exists(f): continue
    with open(f) as fh:
        for row in csv.DictReader(fh):
            try:
                SEED[(row["arm"], int(row["N"]), int(row["r"]))] = (float(row["seedE_min"]), float(row["seedE_mean"]),
                    float(row["ham_bestseed_to_opt"]), float(row["ham_min_to_opt"]), float(row["n_distinct"]))
            except (KeyError, ValueError):
                continue
def seed_mean(arm, lo=W0, hi=W1, col=0, rel=True):
    v = [x[col] / (EOPT[N] if rel else 1) for (aa, N, r), x in SEED.items() if aa == arm and lo <= N <= hi]
    return float(np.mean(v)) if v else np.nan

# ------------------------------------------------------------------ 0
hdr("0. Inventory (arm: N range, replicates) and censored replicate medians")
for arm in ARMS:
    NN = Ns(arm); R = {Y[(arm, N)].shape[0] for N in NN}
    out(f"   {arm:9s} N={NN[0]}-{NN[-1]} ({len(NN)} sizes)  R={sorted(R)}")
for k in sorted(Y):
    nb = int(np.isinf(MED[k]).sum())
    if nb: out(f"   WARNING {k[0]} N={k[1]}: {nb} of {len(MED[k])} replicate medians censored (Q50 exact while < 50%)")

# ------------------------------------------------------------------ 1
hdr("1. Validation: mts4 without settings must reproduce mts v3 (same seeds, same random streams)")
for v, ref in (("V0", "A0"), ("V2m", "A2m")):
    if not has(v): out(f"   {v}: not run"); continue
    tot = same = 0
    for N in Ns(v):
        if (ref, N) not in RAW: continue
        A = RAW[(v, N)]; B = RAW[(ref, N)]
        bmap = {(int(r), int(s)): int(e) for r, s, e in B[:, :3]}
        for r, s, e in A[:, :3]:
            if (int(r), int(s)) in bmap: tot += 1; same += bmap[(int(r), int(s))] == int(e)
    out(f"   {v} vs {ref}: {same}/{tot} runs identical  -> {'OK' if tot and same == tot else 'MISMATCH - check mts4 before using the extension results'}")

# ------------------------------------------------------------------ 2
SEEDCOST = {"A1": lambda N: 0, "A2k": lambda N: 1e3, "A2": lambda N: 1e5, "A2m": lambda N: 1e6, "A3": lambda N: 1e7,
            "A5": lambda N: 1e5, "A6": lambda N: 1e7, "A8": lambda N: 1000.0 * N * N,
            "M2m_k1": lambda N: 1e6, "M2m_k10": lambda N: 1e6, "M2m_k50": lambda N: 1e6, "P2m_d8": lambda N: 1e6,
            "P2m_d4": lambda N: 1e6, "M3_k1": lambda N: 1e7, "P3_d8": lambda N: 1e7}
hdr("2. Seed cost counted: T = TTS + cost of the seed (uniform samples: 1 evaluation each; A8: 1000 N moves x N evaluations)")
out("   (QE-MTS: the DCQO cost is 1e5 shots plus the circuit simulation and is not in objective evaluations)")
for arm in ("A1", "A2k", "A2", "A2m", "A3", "A8", "A5", "A6"):
    if not has(arm): continue
    res = report(arm, a.ref, "with seed cost", cost_fn=SEEDCOST[arm])
    if res:
        neg = [N for N in common(arm, a.ref) if gap(arm, a.ref, N, SEEDCOST[arm](N))[2] < 0]
        out(f"             N with D significantly < 0 after counting the seed: {neg if neg else 'none'}")

# ------------------------------------------------------------------ 3
hdr("3. Seed properties within an arm: Spearman correlation of log10(replicate median TTS) with the seed energy and with the\n"
    "   Hamming distance of the seed to the optimum, per N, pooled as the mean over N (permutation test within N, 1000 perms)")
WROWS = []
for arm in ("A1", "A2k", "A2", "A2m", "A3", "A8", "Q1"):
    NN = [N for N in Ns(arm, 27, 48) if (arm, N, 0) in SEED]
    if not NN: continue
    per = {}
    for N in NN:
        med = MED[(arm, N)]; R = len(med)
        x = np.array([SEED.get((arm, N, r), (np.nan,) * 5) for r in range(R)])
        ok = np.isfinite(med) & np.isfinite(x[:, 0])
        if ok.sum() < 10: continue
        per[N] = (np.log10(med[ok]), x[ok, 0], x[ok, 2])
    if not per: continue
    res = {}
    for j, name in ((1, "energy"), (2, "hamming")):
        rho = {N: stats.spearmanr(v[0], v[j]).correlation for N, v in per.items()}
        obs = np.nanmean(list(rho.values()))
        null = []
        for _ in range(1000):
            null.append(np.nanmean([stats.spearmanr(v[0], rng.permutation(v[j])).correlation for v in per.values()]))
        p = float(np.mean(np.abs(null) >= abs(obs)))
        res[name] = (obs, p)
        for N, rr in rho.items(): WROWS.append(dict(arm=arm, N=N, variable=name, rho=rr))
    out(f"   {arm:5s} N={min(per)}-{max(per)}: rho(TTS, seed energy) = {res['energy'][0]:+.3f} (p={res['energy'][1]:.3f});"
        f"  rho(TTS, Hamming to optimum) = {res['hamming'][0]:+.3f} (p={res['hamming'][1]:.3f})")
out("   (negative rho with energy = lower-energy seeds give LONGER runs inside the same arm)")
with open(os.path.join(a.out, "within_arm.csv"), "w") as f:
    f.write("arm,N,variable,rho\n"); f.writelines(f"{r['arm']},{r['N']},{r['variable']},{r['rho']:.4f}\n" for r in WROWS)

# ------------------------------------------------------------------ 4
hdr("4. Odd/even hardness and the optima reached")
def resid_parity(arm, lo, hi):
    NN = Ns(arm, lo, hi)
    if len(NN) < 4: return None
    x = np.array(NN, float); y = np.array([LQ[(arm, N)] for N in NN]); yb = np.stack([LQB[(arm, N)] for N in NN], 1)
    odd = (x % 2 == 1)
    def diff(yy):
        r = yy - np.polyval(np.polyfit(x, yy, 1), x); return r[odd].mean() - r[~odd].mean()
    d = diff(y); db = np.array([diff(row) for row in yb if np.all(np.isfinite(row))])
    return d, np.percentile(db, 2.5), np.percentile(db, 97.5)
for lo, hi in ((27, 37), (38, 48)):
    r = resid_parity(a.ref, lo, hi)
    if r: out(f"   A0 log10 Q50, N={lo}-{hi}: odd minus even, after removing a linear trend: {r[0]:+.2f} dex [{r[1]:+.2f},{r[2]:+.2f}]")
for arm in ("A1", "A2", "A2m"):
    NN = common(arm, a.ref, 38, 48)
    if len(NN) < 4: continue
    od = [N for N in NN if N % 2]; ev = [N for N in NN if not N % 2]
    if not od or not ev: continue
    def mdraw(L): return np.mean([rng.choice(gap(arm, a.ref, N)[3], a.nboot) for N in L], axis=0)
    dd = mdraw(od) - mdraw(ev)
    po = np.mean([gap(arm, a.ref, N)[0] for N in od]); pe = np.mean([gap(arm, a.ref, N)[0] for N in ev])
    out(f"   D_{arm} on N=38-48: odd N mean {po:+.3f}, even N mean {pe:+.3f}, difference {po-pe:+.3f} "
        f"[{np.percentile(dd, 2.5):+.3f},{np.percentile(dd, 97.5):+.3f}]")

def bitrev(x, N):
    r = np.zeros_like(x)
    for i in range(N): r |= ((x >> np.uint64(i)) & np.uint64(1)) << np.uint64(N - 1 - i)
    return r
def canon(x, N):
    x = np.asarray(x, dtype=np.uint64); full = np.uint64((1 << N) - 1)
    alt = np.uint64(sum(1 << i for i in range(1, N, 2)))
    imgs = []
    for aa in (x, bitrev(x, N)):
        for b in (aa, aa ^ alt): imgs += [b, b ^ full]
    return np.min(np.stack(imgs), axis=0)
def skew(x, N):
    if N % 2 == 0: return False
    s = [-1 if (int(x) >> i) & 1 else 1 for i in range(N)]; c = (N - 1) // 2
    return all(s[c + l] == (-1) ** l * s[c - l] for l in range(1, c + 1))
OPT = []
for N in sorted({N for (arm, N) in RAW}):
    hits_c0, allc = [], []
    for (arm, NN), d0 in RAW.items():
        if NN != N or d0.shape[1] < 5: continue
        cf = d0[(d0[:, 2] > 0) & (d0[:, 4] > 0), 4].astype(np.uint64)
        if not len(cf): continue
        allc.append(cf)
        if arm == "C0": hits_c0.append(cf)
    if not allc: continue
    allc = np.concatenate(allc)
    E = energy_ints(allc, N); bad = int((E != EOPT[N]).sum())
    cl = canon(allc, N); classes = np.unique(cl)
    ref = int(canon(np.array([spins_to_int(rle_to_spins(RLE[N]))], dtype=np.uint64), N)[0]) if N in RLE else None
    frac = ""
    if hits_c0:
        c0 = canon(np.concatenate(hits_c0), N); u, cnt = np.unique(c0, return_counts=True)
        frac = " ".join(f"{c/len(c0):.2f}" for c in sorted(cnt, reverse=True)[:6])
    nskew = sum(skew(c, N) for c in classes)
    OPT.append(dict(N=N, n_hits=len(allc), n_classes=len(classes), n_skew=nskew,
                    ref_found=(ref in set(int(c) for c in classes)) if ref is not None else "", bad_energy=bad))
    out(f"   N={N}: {len(allc)} optima reached, {len(classes)} distinct up to symmetry ({nskew} skew-symmetric)"
        f"{'; P&M optimum among them' if OPT[-1]['ref_found'] is True else ''}{'; C0 class frequencies ' + frac if frac else ''}"
        f"{'  ENERGY CHECK FAILED x' + str(bad) if bad else ''}")
if OPT:
    with open(os.path.join(a.out, "optima.csv"), "w") as f:
        k = list(OPT[0]); f.write(",".join(k) + "\n"); f.writelines(",".join(str(r[x]) for x in k) + "\n" for r in OPT)
    out("   (number of classes is a lower bound on the degeneracy; compare with the list of Packebusch & Mertens 2016)")

# ------------------------------------------------------------------ 5
hdr("5. How to use a seed. A2m seed: k copies + (100-k) random strings (k=100 is A2m), and 100 copies each with d random flips")
RES5 = {}
for arm, lab in (("A2m", "k=100 (published protocol)"), ("M2m_k50", "k=50"), ("M2m_k10", "k=10"), ("M2m_k1", "k=1"),
                 ("P2m_d8", "copies + round(N/8) flips"), ("P2m_d4", "copies + round(N/4) flips"),
                 ("A3", "A3 seed, k=100"), ("M3_k1", "A3 seed, k=1"), ("P3_d8", "A3 seed, copies + N/8 flips")):
    RES5[arm] = report(arm, a.ref, lab, tail=(38, 42))
out("   with the seed cost counted:")
for arm in ("A2m", "M2m_k1", "P2m_d8", "A3", "M3_k1", "P3_d8"):
    if has(arm): report(arm, a.ref, "+ seed cost", cost_fn=SEEDCOST[arm], tail=(38, 42))

# ------------------------------------------------------------------ 6
hdr("6. MTS hyperparameters: replicated A2m seed vs random population with the SAME settings")
report("A2m", a.ref, "default (K=100, p=1/N, tabu x1)", lo=27, hi=37, tail=(38, 40))
for v, lab in (("K50", "K=50"), ("K200", "K=200"), ("pm2", "p_mut=2/N"), ("tb2", "tabu length x2")):
    report(f"H2m_{v}", f"H0_{v}", lab, lo=27, hi=37, tail=(38, 40))
out("   efficiency of the settings themselves (random population vs default MTS):")
for v in ("K50", "K200", "pm2", "tb2"): report(f"H0_{v}", a.ref, "", lo=27, hi=37, tail=(38, 40))

# ------------------------------------------------------------------ 7
hdr("7. Population-free iterated tabu search (ITS): seeded start vs random start")
for arm, lab in (("I2", "A2 seed"), ("I2m", "A2m seed"), ("I3", "A3 seed")): report(arm, "I0", lab, tail=(38, 40))
out("   ITS (random start) vs MTS:"); report("I0", a.ref, "", tail=(38, 40))

# ------------------------------------------------------------------ 8
hdr("8. A2m up to N=48: the classical arm that reproduces the published curve, at the published crossover")
LINES = {}
for arm in ("A0", "A2", "A2m"):
    NN = Ns(arm, W0, W1)
    if len(NN) >= 5: LINES[arm] = np.polyfit(NN, [LQ[(arm, N)] for N in NN], 1)
for arm in ("A2", "A2m"):
    if arm not in LINES or not has(arm): continue
    cr = (LINES["A0"][1] - LINES[arm][1]) / (LINES[arm][0] - LINES["A0"][0])
    out(f"   {arm}: Q50 lines fitted on N=27-37 cross at N={cr:.1f}")
    for N in common(arm, a.ref, 38, 48):
        d, l, h, _ = gap(arm, a.ref, N)
        pred = np.polyval(LINES[arm], N) - np.polyval(LINES["A0"], N)
        out(f"      N={N}: predicted {pred:+.3f}, measured {d:+.3f} [{l:+.3f},{h:+.3f}]{'  <0 significant' if h < 0 else ''}")
    mg = mean_gap(arm, a.ref, 38, 48)
    if mg: out(f"      mean D over N={mg['N'][0]}-{mg['N'][-1]} (n={mg['n']}): {mg['mean']:+.3f} [{mg['lo']:+.3f},{mg['hi']:+.3f}], P(mean<=0)={mg['p_le0']:.3f}")

# ------------------------------------------------------------------ 9
hdr("9. DCQO seeds (QE-MTS protocol with real DCQO samples)")
PAPER = {}
if a.paper and os.path.exists(a.paper):
    with open(a.paper) as fh:
        for row in csv.DictReader(l for l in fh if not l.startswith("#")):
            try: PAPER[int(row["N"])] = float(row.get("median_D") or row["median"])
            except (KeyError, ValueError): pass
for arm, lab in (("Q1", "best of 1e5 DCQO shots x100"), ("Q5", "100 best distinct DCQO shots"),
                 ("PQ1_d8", "DCQO seed, copies + N/8 flips")):
    report(arm, a.ref, lab, tail=(34, 36))
if has("Q1"):
    for arm in ("A2k", "A2", "A2m", "A3", "Q1"):
        out(f"   {arm:4s}: seed energy / Eopt (mean, N=27-33) {seed_mean(arm, 27, 33):.3f};  Hamming to optimum {seed_mean(arm, 27, 33, 2, False):.2f}")
    # equivalent number of uniform samples, per N, by log-interpolation of the mean seed energy of A2k/A2/A2m/A3
    NS_ = {"A2k": 1e3, "A2": 1e5, "A2m": 1e6, "A3": 1e7}
    eq = []
    for N in Ns("Q1", 27, 42):
        xs = [(np.log10(n), seed_mean(arm, N, N)) for arm, n in NS_.items() if not np.isnan(seed_mean(arm, N, N))]
        q = seed_mean("Q1", N, N)
        if len(xs) >= 2 and not np.isnan(q):
            xs.sort(key=lambda t: -t[1]); e = [t[1] for t in xs]; ln = [t[0] for t in xs]
            eq.append((N, 10 ** np.interp(-q, [-v for v in e], ln)))
    if eq: out("   DCQO seed quality = best of n uniform samples with n ~ " + "  ".join(f"N{N}:{v:.1e}" for N, v in eq)
               + "  (clipped to 1e3-1e7)")
    if PAPER:
        NN = [N for N in common("Q1", a.ref) if N in PAPER]
        if len(NN) >= 3:
            dq = np.array([gap("Q1", a.ref, N)[0] for N in NN]); dp = np.array([PAPER[N] for N in NN])
            out(f"   Q1 vs published QE-MTS gap on N={NN[0]}-{NN[-1]}: RMS {np.sqrt(np.mean((dq-dp)**2)):.3f} dex, r={np.corrcoef(dq, dp)[0,1]:+.2f}, "
                f"mean D {dq.mean():+.3f} vs published {dp.mean():+.3f}")
    out("   overhead vs seed energy (N=27-33) for all replicated seeds:")
    for arm in ("A1", "A2k", "A2", "A2m", "A3", "A8", "Q1"):
        ov = overhead(arm, a.ref, 27, 33) if has(arm) else None
        if ov: out(f"      {arm:4s} E/Eopt {seed_mean(arm, 27, 33):.2f}  Delta {ov[0]:.2e}")

# ------------------------------------------------------------------ write tables
with open(os.path.join(a.out, "gaps_ext.csv"), "w") as f:
    f.write("arm,ref,seed_cost_counted,N,D,lo,hi\n")
    seen = set()
    for r in GAPROWS:
        key = (r["arm"], r["ref"], r["cost"] > 0, r["N"])
        if key in seen: continue
        seen.add(key); f.write(f"{r['arm']},{r['ref']},{int(r['cost'] > 0)},{r['N']},{r['D']:.4f},{r['lo']:.4f},{r['hi']:.4f}\n")
with open(os.path.join(a.out, "summary_ext.txt"), "w") as f: f.write("\n".join(TXT) + "\n")

# ------------------------------------------------------------------ figures
def plot_gap(ax, arm, ref, lab, cost_fn=None, **kw):
    NN = common(arm, ref, 23, 48)
    if not NN: return
    g = [gap(arm, ref, N, cost_fn(N) if cost_fn else 0) for N in NN]
    d = np.array([x[0] for x in g]); lo = np.array([x[1] for x in g]); hi = np.array([x[2] for x in g])
    ax.errorbar(NN, d, yerr=[np.clip(d - lo, 0, None), np.clip(hi - d, 0, None)], marker="o", ms=3, lw=1.2, capsize=0, label=lab, **kw)
def finish(ax, title, xl="N", yl="D = log10 TTS ratio"):
    ax.axhline(0, color="0.4", lw=0.8); ax.axvspan(26.5, 37.5, color="0.92", zorder=0)
    ax.set_title(title, fontsize=9, loc="left"); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=7, frameon=False)

if any(has(x) for x in ("M2m_k1", "P2m_d8", "M3_k1")):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for arm, lab in (("A2m", "A2m: k=100 copies"), ("M2m_k50", "k=50 + 50 random"), ("M2m_k10", "k=10 + 90 random"),
                     ("M2m_k1", "k=1 + 99 random"), ("P2m_d8", "100 copies, N/8 flips"), ("P2m_d4", "100 copies, N/4 flips")):
        if has(arm): plot_gap(axs[0], arm, a.ref, lab)
    finish(axs[0], "(a) A2m seed (best of 1e6): replicated vs mixed vs perturbed")
    for arm, lab in (("A3", "A3: 100 copies"), ("M3_k1", "1 copy + 99 random"), ("P3_d8", "100 copies, N/8 flips")):
        if has(arm): plot_gap(axs[1], arm, a.ref, lab)
    finish(axs[1], "(b) A3 seed (best of 1e7)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ext_seed_usage.png"), dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3.8))
for arm in ("A2", "A2m", "A3", "A5", "A6"):
    if has(arm): plot_gap(ax, arm, a.ref, f"{arm} + seed cost", cost_fn=SEEDCOST[arm])
finish(ax, "seed cost counted in the TTS"); fig.tight_layout()
fig.savefig(os.path.join(a.out, "fig_ext_seedcost.png"), dpi=160); plt.close(fig)

if any(has(f"H2m_{v}") for v in ("K50", "K200", "pm2", "tb2")) or has("I2"):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    if has("A2m"): plot_gap(axs[0], "A2m", a.ref, "default MTS")
    for v, lab in (("K50", "K=50"), ("K200", "K=200"), ("pm2", "p_mut=2/N"), ("tb2", "tabu x2")):
        if has(f"H2m_{v}"): plot_gap(axs[0], f"H2m_{v}", f"H0_{v}", lab)
    finish(axs[0], "(a) replicated A2m seed, other MTS settings")
    for arm, lab in (("I2", "ITS from A2 seed"), ("I2m", "ITS from A2m seed"), ("I3", "ITS from A3 seed")):
        if has(arm): plot_gap(axs[1], arm, "I0", lab)
    finish(axs[1], "(b) population-free ITS, vs ITS from a random start")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ext_robustness.png"), dpi=160); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
NN = Ns(a.ref, 20, 48); yy = [LQ[(a.ref, N)] for N in NN]
axs[0].plot(NN, yy, color="0.6", lw=0.8)
axs[0].scatter([N for N in NN if N % 2], [LQ[(a.ref, N)] for N in NN if N % 2], c="C3", s=18, label="odd N")
axs[0].scatter([N for N in NN if not N % 2], [LQ[(a.ref, N)] for N in NN if not N % 2], c="C0", s=18, label="even N")
axs[0].set_xlabel("N"); axs[0].set_ylabel("log10 Q50 (A0)"); axs[0].legend(fontsize=7, frameon=False)
axs[0].set_title("(a) MTS median TTS", fontsize=9, loc="left")
if OPT:
    axs[1].bar([r["N"] for r in OPT], [r["n_classes"] for r in OPT],
               color=["C3" if r["N"] % 2 else "C0" for r in OPT])
    axs[1].set_xlabel("N"); axs[1].set_ylabel("distinct optima found (up to symmetry)")
    axs[1].set_title("(b) optima reached by the runs", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ext_parity.png"), dpi=160); plt.close(fig)

if has("A2m") and Ns("A2m", 43, 48):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for arm, c in (("A2", "C1"), ("A2m", "C5")):
        if not has(arm): continue
        plot_gap(ax, arm, a.ref, f"{arm} measured", color=c)
        if arm in LINES:
            x = np.arange(27, 49); ax.plot(x, np.polyval(LINES[arm], x) - np.polyval(LINES["A0"], x), "--", color=c, lw=1,
                                           label=f"{arm}: lines on 27-37, extrapolated")
    ax.axvline(47, color="0.5", ls=":", lw=1); ax.set_ylim(-0.6, 0.8)
    finish(ax, "crossover test up to N=48"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_ext_a2m48.png"), dpi=160); plt.close(fig)

if has("Q1"):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for arm, lab in (("Q1", "DCQO best of 1e5 x100"), ("Q5", "100 best distinct DCQO"), ("PQ1_d8", "DCQO seed, N/8 flips"),
                     ("A2m", "A2m (classical)")):
        if has(arm): plot_gap(axs[0], arm, a.ref, lab)
    if PAPER: axs[0].plot(sorted(PAPER), [PAPER[N] for N in sorted(PAPER)], "ks-", ms=4, lw=1.2, label="published QE-MTS")
    finish(axs[0], "(a) DCQO seeds in the QE-MTS protocol")
    pts = []
    for arm in ("A1", "A2k", "A2", "A2m", "A3", "A8", "Q1"):
        ov = overhead(arm, a.ref, 27, 33) if has(arm) else None
        if ov: pts.append((arm, seed_mean(arm, 27, 33), ov[0]))
    for arm, e, d in pts:
        axs[1].scatter(e, d, s=40, marker="*" if arm == "Q1" else "o", c="k" if arm == "Q1" else "C0")
        axs[1].annotate(arm, (e, d), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axs[1].set_xscale("log"); axs[1].set_yscale("log"); axs[1].invert_xaxis()
    axs[1].set_xlabel("seed energy / Eopt (N=27-33)"); axs[1].set_ylabel("overhead Delta (N=27-33)")
    axs[1].set_title("(b) overhead vs seed quality, with DCQO", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ext_dcqo.png"), dpi=160); plt.close(fig)

print(f"\nwritten to {a.out}")
