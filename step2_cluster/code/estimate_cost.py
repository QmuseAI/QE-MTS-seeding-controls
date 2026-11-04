"""Project the cost of plan.txt from the timing job (10_timing.sbatch), for mts v3 (early stop per replicate).

usage (login node, from step2_cluster, takes seconds; reuses the existing run_main/timing data):
    python code/estimate_cost.py --timing run_main/timing --plan slurm/plan.txt --nodes 4 --cpus 500 \
                                 [--kfrac 0.70] [--threads 8]

Model: with early stop, a replicate costs ~ S * F * median TTS, where F = mean(min(T, Q_kfrac)) / median
(measured on the pooled timing runs, x1.2 for replicate-to-replicate spread). median TTS(N) = exp(a + b N)
through the measured medians at N=37 and 42 (arm A0); time per evaluation proportional to N, measured under
full node load. Seeded arms are charged 1.5x (slower than A0 at small N) -> conservative.
"""
import argparse, glob, os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--timing", required=True); ap.add_argument("--plan", required=True)
ap.add_argument("--cpus", type=int, default=500, help="worker threads per node")
ap.add_argument("--nodes", type=int, default=4)
ap.add_argument("--kfrac", type=float, default=0.70); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
med, tev, F = {}, {}, {}
for N in (37, 42):
    ev, sec, T = 0, 0.0, []
    for f in glob.glob(os.path.join(a.timing, f"N{N}_r*.txt")):
        L = [l.split() for l in open(f) if l.strip()]
        tf = f[:-4] + ".time"
        if not L or not os.path.exists(tf): continue
        _, s, e = open(tf).read().split(); sec += float(e) - float(s)
        ev += sum(abs(int(x[2])) for x in L); T += [int(x[2]) for x in L]
    T = np.array(T, float); T[T < 0] = np.inf
    med[N] = np.median(T); tev[N] = sec / ev
    qk = np.quantile(T, a.kfrac); F[N] = np.mean(np.minimum(T, qk)) / med[N]
    print(f"N={N}: {len(T)} runs, median TTS {med[N]:.3g}, mean {np.mean(T[np.isfinite(T)]):.3g} evals, "
          f"early-stop factor F={F[N]:.2f}, {tev[N]*1e9:.1f} ns/eval per thread under full load")
b = (np.log(med[42]) - np.log(med[37])) / 5; A = np.log(med[37]) - 37 * b
c = np.mean([tev[N] / N for N in tev]); Ff = 1.2 * max(F.values())
print(f"median-TTS growth exp(b) = {np.exp(b):.3f} per N (paper's MTS: 1.37); F used = {Ff:.2f}")
tot, longest, rows = 0.0, 0.0, []
for line in open(a.plan):
    w = line.split("#")[0].split()
    if not w: continue
    arm, n0, n1, R, S = w[0], int(w[1]), int(w[2]), int(w[3]), int(w[4])
    f = 1.0 if arm in ("A0", "A0na") else 1.5
    cs = 0.0
    for N in range(n0, n1 + 1):
        per_task = f * S * Ff * np.exp(A + b * N) * c * N
        cs += R * per_task; longest = max(longest, 3 * per_task / a.threads / 0.85)
    tot += cs; rows.append((arm, n0, n1, cs / 3600))
for r in rows: print(f"  {r[0]:5s} N={r[1]}-{r[2]}: {r[3]:8.1f} thread-hours")
P = a.cpus * a.nodes
wall = max(tot / P / 0.85, longest)
print(f"TOTAL {tot/3600:.0f} thread-hours (before subtracting tasks already finished). On {a.nodes} x {a.cpus} = {P}: "
      f"ideal {tot/P/3600:.2f} h; slowest task ~{longest/3600:.2f} h (3x a typical replicate, {a.threads} threads); "
      f"expected wall-clock ~{wall/3600:.1f} h (+-50%)")
