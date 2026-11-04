"""Progress and ETA of an extension run (login node, from step2_cluster/, any time):
    python code/progress_ext.py                                   # TAG=ext, plan_ext.txt
    python code/progress_ext.py --tag hard --plan slurm/plan_hard.txt
    python code/progress_ext.py --tag dcqo --plan slurm/plan_dcqo.txt --table     # + replicates per (arm, N)

Counts finished part files of the tasks in RUN/tasks/manifest_<tag>.csv and weights them by the evaluations that
each N actually cost in run_main (A0, per replicate) -> a better ETA than N*1.37^N, still only +-50 %.
Tasks are interleaved by replicate (r = 0 of every (arm, N), then r = 1, ...), so the fraction rises steadily.
"""
import argparse, csv, glob, os, time, datetime, collections

# evaluations spent by A0 in run_main per N (100 replicates); used as relative cost per task
A0COST = {20: 3.95e8, 21: 5.99e8, 22: 1.00e8, 23: 4.95e8, 24: 4.07e8, 25: 6.98e8, 26: 2.96e8, 27: 3.67e9, 28: 1.93e9,
          29: 2.82e9, 30: 2.33e9, 31: 2.36e9, 32: 7.60e9, 33: 1.74e10, 34: 2.03e10, 35: 3.49e10, 36: 3.83e10,
          37: 4.74e10, 38: 5.26e10, 39: 3.50e12, 40: 5.90e11, 41: 2.18e13, 42: 3.64e12, 43: 2.76e14, 44: 8.08e12,
          45: 3.98e13, 46: 1.46e12, 47: 1.71e13, 48: 9.55e13}

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="ext"); ap.add_argument("--plan", default="slurm/plan_ext.txt")
ap.add_argument("--run", default=os.environ.get("RUN", "run_ext")); ap.add_argument("--table", action="store_true")
a = ap.parse_args()
man = os.path.join(a.run, "tasks", f"manifest_{a.tag}.csv")
if not os.path.exists(man): raise SystemExit(f"no {man}: has the prep job of '{a.tag}' finished?")
S = {}
for line in open(a.plan):
    w = line.split("#")[0].split()
    if w: S[w[0]] = int(w[4])
rows = list(csv.DictReader(open(man)))
done_c = tot_c = 0.0; per = collections.defaultdict(lambda: [0, 0]); running = 0
for r in rows:
    N = int(r["N"]); c = A0COST.get(N, 1e14) / 100; tot_c += c
    p = os.path.join(a.run, "parts", f"{r['arm']}_N{N}_r{r['r']}.txt")
    ok = os.path.exists(p) and sum(1 for _ in open(p)) == S.get(r["arm"], 100)
    running += os.path.exists(p + ".tmp")
    per[(r["arm"], N)][1] += 1
    if ok: done_c += c; per[(r["arm"], N)][0] += 1
ndone = sum(v[0] for v in per.values())
print(f"[{a.tag}] tasks: {ndone}/{len(rows)} finished, {running} running;  cost-weighted progress {100*done_c/tot_c:.1f}%")

# elapsed time: start of the first shard of this run array (time spent waiting in the queue is not counted)
t_man = os.path.getmtime(man)
# the run array of this tag = logs whose first line reports exactly as many tasks as shard_<tag>_<k>.txt
import re
nshard = {}
for f in glob.glob(os.path.join(a.run, "tasks", f"shard_{a.tag}_*.txt")):
    k = int(re.search(r"_(\d+)\.txt$", f).group(1)); nshard[k] = sum(1 for _ in open(f))
starts = []
for f in glob.glob("logs/qemts_run_*_*.log"):
    try:
        first = open(f).readline()
        ts = datetime.datetime.fromisoformat(first.split()[0]).timestamp()
        m = re.search(r"shard (\d+)/\d+: (\d+) tasks", first)
        if ts >= t_man - 60 and m and nshard.get(int(m.group(1))) == int(m.group(2)): starts.append(ts)
    except (ValueError, IndexError, OSError):
        pass
if starts:
    el = time.time() - min(starts)
    msg = f"elapsed since this run array started: {el/3600:.2f} h"
    if done_c > 0: msg += f";  estimated remaining: {el*(tot_c-done_c)/done_c/3600:.2f} h  (rough, +-50%)"
    print(msg)
else:
    print("run array not started yet (squeue shows it pending)")

arms = []
for r in rows:
    if r["arm"] not in arms: arms.append(r["arm"])
print("\nper arm: finished replicate tasks / queued   (sizes still open)")
for arm in arms:
    ks = sorted(k for k in per if k[0] == arm)
    f = sum(per[k][0] for k in ks); t = sum(per[k][1] for k in ks)
    open_N = [k[1] for k in ks if per[k][0] < per[k][1]]
    print(f"  {arm:9s} {f:5d}/{t:<5d}" + (f"  open: N={','.join(map(str, open_N))}" if open_N else "  done"))
if a.table:
    Ns = sorted({k[1] for k in per})
    print("\nN    " + "".join(f"{x:>10s}" for x in arms))
    for N in Ns:
        print(f"{N:<5}" + "".join(f"{per[(x,N)][0]:>6d}/{per[(x,N)][1]:<3d}" if (x, N) in per else f"{'':>10s}" for x in arms))
