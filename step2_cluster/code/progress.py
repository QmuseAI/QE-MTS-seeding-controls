"""Progress and ETA of the production run (run on the login node, any time):
    python code/progress.py            (from the step2_cluster directory; RUN=run_main by default)

Counts finished tasks (complete part files) per arm and N, weights them by the same cost model used for
load balancing (N * 1.37^N), and extrapolates the remaining wall time from the time elapsed since the run
array started (first line of logs/qemts_run_*_0.log). The cost model is rough, so treat the ETA as +-50 %;
the largest-N tasks run first, so the cost-weighted fraction rises slowly at the start and fast at the end.
"""
import csv, glob, os, sys, time, datetime, collections

run = os.environ.get("RUN", "run_main")
man = os.path.join(run, "tasks", "manifest.csv")
if not os.path.exists(man): sys.exit(f"no {man}: has the prep job finished?")
rows = list(csv.DictReader(open(man)))
S = {}
for line in open(os.environ.get("PLAN", "slurm/plan.txt")):
    w = line.split("#")[0].split()
    if w: S[w[0]] = int(w[4])

done_c = tot_c = 0.0; per = collections.defaultdict(lambda: [0, 0]); running = 0
for r in rows:
    c = float(r["cost"]); tot_c += c
    p = os.path.join(run, "parts", f"{r['arm']}_N{r['N']}_r{r['r']}.txt")
    ok = os.path.exists(p) and sum(1 for _ in open(p)) == S[r["arm"]]
    running += os.path.exists(p + ".tmp")
    per[(r["arm"], int(r["N"]))][1] += 1
    if ok: done_c += c; per[(r["arm"], int(r["N"]))][0] += 1
ndone = sum(v[0] for v in per.values())
print(f"tasks: {ndone}/{len(rows)} finished, {running} running;  cost-weighted progress {100*done_c/tot_c:.1f}%")

logs = sorted(glob.glob("logs/qemts_run_*_0.log"), key=os.path.getmtime)
if logs:
    first = open(logs[-1]).readline().split()[0]
    t0 = datetime.datetime.fromisoformat(first).timestamp(); el = time.time() - t0
    print(f"elapsed since run start: {el/3600:.2f} h", end="")
    if done_c > 0:
        print(f";  estimated remaining: {el*(tot_c-done_c)/done_c/3600:.2f} h  (rough, +-50%)")
    else:
        print(";  no task finished yet (largest N first) - check again later, or see run_main/timing/estimate.txt")

arms = sorted({k[0] for k in per}); Ns = sorted({k[1] for k in per})
print("\nfinished replicates per (arm, N):")
print("N    " + "".join(f"{a:>8s}" for a in arms))
for N in Ns:
    print(f"{N:<5}" + "".join(f"{per[(a,N)][0]:>4d}/{per[(a,N)][1]:<3d}" if (a, N) in per else f"{'':>8s}" for a in arms))
