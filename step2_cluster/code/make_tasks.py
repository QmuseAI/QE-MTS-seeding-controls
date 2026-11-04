"""Build the task list for the SLURM array (one task = one (arm, N, replicate) with S seeds).

usage: python make_tasks.py --plan ../slurm/plan.txt --run /path/to/RUN --code /path/to/code \
                            --seeds /path/to/seeds --shards 4 [--cap 3000000000] [--kfrac 0.70] [--tag X] [--interleave]
Writes RUN/tasks/shard_<k>.txt (one shell command per line) and RUN/tasks/manifest.csv.
Plan lines:  arm  Nmin  Nmax  R  S  [cap]  [kfrac]     (the optional last two columns override --cap / --kfrac
             for that line only; cap may be written as 5e10)
--kfrac: early stop per replicate (mts v3): stop once the ceil(kfrac*S) smallest TTS are known exactly; the
         rest are right-censored. Medians and the seed-level bootstrap stay exact for kfrac >= 0.7 (S=100).
         --kfrac 1 = old behaviour (every run to completion or cap).
--cap:   per-run safety cap on evaluations (bounds the longest task); default 3e9.
--tag:   write RUN/tasks/shard_<tag>_<k>.txt and manifest_<tag>.csv instead, so a second task list can be built
         while another array job is still reading shard_<k>.txt (20_run.sbatch reads the tagged files if TAG is set).
--interleave: order the tasks replicate by replicate (r=0 of every (arm, N), then r=1, ...) instead of longest
         first, so that a job stopped early leaves every (arm, N) with about the same number of finished replicates.
Tasks whose output file already holds S complete lines are skipped -> re-running this script after an
interrupted job only re-queues what is missing (safe restart).
Load balancing: tasks are sorted by estimated cost (N * 1.37^N) and dealt to shards by longest-processing-time
first; inside a node, xargs takes them in that order.
"""
import argparse, os, math

# arm -> (mts mode, seed-file tag or None, RNG base, aspiration)
ARMS = {
    "A0":   (0, None,  1, 1),   # random population (paper's MTS baseline)
    "A1":   (1, "A1",  2, 1),   # one random string x K
    "A2":   (1, "A2",  3, 1),   # best of 1e5 uniform x K  (paper's QE-MTS protocol, classical sampler)
    "A5":   (2, "A5",  6, 1),   # best 100 distinct of 1e5 uniform
    "A6":   (2, "A6",  7, 1),   # best 100 distinct of 1e7 uniform
    "A7a":  (1, "A7a", 8, 1),   # optimum with N/4 flips x K (oracle)
    "A7b":  (1, "A7b", 9, 1),   # optimum with N/8 flips x K (oracle)
    "A0na": (0, None,  1, 0),   # = A0 without aspiration (same random streams as A0 -> paired)
    "A2na": (1, "A2",  3, 0),   # = A2 without aspiration (same seeds and streams as A2 -> paired)
    # seed-quality sweep of the replicated protocol (2026-10-22)
    "A2k":  (1, "A2k", 10, 1),  # best of 1e3 uniform x K
    "A2m":  (1, "A2m", 11, 1),  # best of 1e6 uniform x K
    "A3":   (1, "A3",  12, 1),  # best of 1e7 uniform x K (= best member of the A6 population)
    "A8":   (1, "A8",  13, 1),  # best non-optimal string of a 1000*N-move tabu walk x K
    # early-stop validation: same seeds and RNG streams as A0 / A2, run with kfrac = 1 (no early stop)
    "A0f":  (0, None,  1, 1),
    "A2f":  (1, "A2",  3, 1),
}

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--code", required=True); ap.add_argument("--seeds", required=True)
ap.add_argument("--shards", type=int, default=4); ap.add_argument("--cap", default="3000000000")
ap.add_argument("--kfrac", type=float, default=0.70)
ap.add_argument("--tag", default="")
ap.add_argument("--interleave", action="store_true")
a = ap.parse_args()
run, code, seeds = map(os.path.abspath, (a.run, a.code, a.seeds))
eopt = dict(tuple(map(int, l.split())) for l in open(os.path.join(code, "eopt_all.txt")) if l.strip())
os.makedirs(os.path.join(run, "parts"), exist_ok=True); os.makedirs(os.path.join(run, "tasks"), exist_ok=True)

tasks, n_done, n_total = [], 0, 0
for line in open(a.plan):
    line = line.split("#")[0].split()
    if not line: continue
    arm, nmin, nmax, R, S = line[0], int(line[1]), int(line[2]), int(line[3]), int(line[4])
    cap = str(int(float(line[5]))) if len(line) > 5 else str(int(float(a.cap)))     # mts reads it with atoll
    kfrac = float(line[6]) if len(line) > 6 else a.kfrac
    mode, tag, base, asp = ARMS[arm]
    count_init = 1 if mode == 0 else 0                       # = mts default, now passed explicitly
    kstop = min(S, max(S // 2 + 1, math.ceil(kfrac * S - 1e-9)))
    n_line = 0
    for N in range(nmin, nmax + 1):
        if N not in eopt: raise SystemExit(f"no Eopt for N={N}")
        sf = os.path.join(seeds, f"seeds_{tag}_N{N}.txt") if tag else "none"
        if tag and not os.path.exists(sf): raise SystemExit(f"missing seed file {sf}")
        for r in range(R):
            n_total += 1
            out = os.path.join(run, "parts", f"{arm}_N{N}_r{r}.txt")
            if os.path.exists(out) and sum(1 for _ in open(out)) == S:
                n_done += 1; continue
            cmd = (f"{code}/mts {N} {eopt[N]} {mode} 1 {S} {sf} {base} {cap} {r} {asp} {count_init} {kstop} > {out}.tmp "
                   f"&& mv {out}.tmp {out}")
            tasks.append((N * 1.37 ** N, arm, N, r, cmd)); n_line += 1
    print(f"  {arm:5s} N={nmin}-{nmax}  cap={cap}  kstop={kstop}  queued {n_line}")

if a.interleave: tasks.sort(key=lambda t: (t[3], -t[0]))
else: tasks.sort(key=lambda t: -t[0])
load = [0.0] * a.shards; shard = [[] for _ in range(a.shards)]
for t in tasks:
    k = min(range(a.shards), key=lambda i: load[i]); load[k] += t[0]; shard[k].append(t)
pre = f"shard_{a.tag}_" if a.tag else "shard_"
for k in range(a.shards):
    with open(os.path.join(run, "tasks", f"{pre}{k}.txt"), "w") as f:
        f.write("".join(t[4] + "\n" for t in shard[k]))
with open(os.path.join(run, "tasks", f"manifest_{a.tag}.csv" if a.tag else "manifest.csv"), "w") as f:
    f.write("shard,arm,N,r,cost\n")
    for k in range(a.shards):
        for t in shard[k]: f.write(f"{k},{t[1]},{t[2]},{t[3]},{t[0]:.4g}\n")
print(f"default cap={a.cap}, default kfrac={a.kfrac}; shard files {pre}<k>.txt"
      f"{' (interleaved by replicate)' if a.interleave else ''}")
print(f"{n_total} tasks in plan, {n_done} already complete, {len(tasks)} queued over {a.shards} shards "
      f"(relative loads: {', '.join(f'{l/max(load):.3f}' for l in load) if tasks else '-'})")
