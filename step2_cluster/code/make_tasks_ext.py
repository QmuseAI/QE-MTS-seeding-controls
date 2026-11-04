"""Task list for the extension runs (mts4 options, new arms). Same conventions as make_tasks.py.

usage: python make_tasks_ext.py --plan ../slurm/plan_ext.txt --run RUN --code CODE --seeds SEEDS --shards 4 \
                                --tag ext [--cap 5e10] [--kfrac 0.70] [--interleave]
Plan lines:  arm  Nmin  Nmax  R  S  [cap]  [kfrac]
Writes RUN/tasks/shard_<tag>_<k>.txt and RUN/tasks/manifest_<tag>.csv; finished (arm, N, r) part files are skipped.
Every mts4 task also writes the configuration that reached the optimum (cfg=1, 5th output column).
"""
import argparse, os, math

# arm -> (binary, mts mode, seed-file tag or None, RNG base, aspiration, mts4 settings; {d8},{d4} = round(N/8),round(N/4))
# Bases equal to those of A0 (1), A2m (11), A3 (12) give the same MTS random streams as those arms (paired runs).
ARMS = {
    # validation: mts4 without settings must reproduce A0 / A2m of run_main exactly
    "V0":      ("mts4", 0, None,  1, 1, ""),
    "V2m":     ("mts4", 1, "A2m", 11, 1, ""),
    # collector of optimal configurations (= A0 runs, replicates 0..R-1, plus the optimum that was reached)
    "C0":      ("mts4", 0, None,  1, 1, ""),
    # how to use a seed: k copies + (K-k) random strings, and perturbed copies (A2m seed)
    "M2m_k1":  ("mts4", 1, "A2m", 11, 1, "kcopies=1"),
    "M2m_k10": ("mts4", 1, "A2m", 11, 1, "kcopies=10"),
    "M2m_k50": ("mts4", 1, "A2m", 11, 1, "kcopies=50"),
    "P2m_d8":  ("mts4", 1, "A2m", 11, 1, "nflip={d8}"),
    "P2m_d4":  ("mts4", 1, "A2m", 11, 1, "nflip={d4}"),
    # the same fixes for the deepest classical seed (A3 = best of 1e7)
    "M3_k1":   ("mts4", 1, "A3",  12, 1, "kcopies=1"),
    "P3_d8":   ("mts4", 1, "A3",  12, 1, "nflip={d8}"),
    # MTS hyperparameters: (reference, replicated A2m seed) pairs
    "H0_K50":   ("mts4", 0, None,  1, 1, "K=50"),
    "H2m_K50":  ("mts4", 1, "A2m", 11, 1, "K=50"),
    "H0_K200":  ("mts4", 0, None,  1, 1, "K=200"),
    "H2m_K200": ("mts4", 1, "A2m", 11, 1, "K=200"),
    "H0_pm2":   ("mts4", 0, None,  1, 1, "pmut=2"),
    "H2m_pm2":  ("mts4", 1, "A2m", 11, 1, "pmut=2"),
    "H0_tb2":   ("mts4", 0, None,  1, 1, "tabu=2"),
    "H2m_tb2":  ("mts4", 1, "A2m", 11, 1, "tabu=2"),
    # population-free iterated tabu search (no diversity to lose)
    "I0":   ("mts4", 0, None,  31, 1, "K=1 solver=1 pmut=4"),
    "I2":   ("mts4", 1, "A2",  32, 1, "K=1 solver=1 pmut=4"),
    "I2m":  ("mts4", 1, "A2m", 33, 1, "K=1 solver=1 pmut=4"),
    "I3":   ("mts4", 1, "A3",  34, 1, "K=1 solver=1 pmut=4"),
    # A2m beyond N=42 (same binary and streams as the A2m arm of run_main)
    "A2m":  ("mts",  1, "A2m", 11, 1, None),
    # DCQO seeds from gpu/dcqo_seeds.py
    "Q1":     ("mts4", 1, "Q1", 41, 1, ""),             # best of 1e5 DCQO shots x K (the QE-MTS protocol)
    "Q5":     ("mts4", 2, "Q5", 42, 1, ""),             # 100 best distinct DCQO shots
    "PQ1_d8": ("mts4", 1, "Q1", 41, 1, "nflip={d8}"),  # DCQO seed, perturbed copies
}

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--code", required=True); ap.add_argument("--seeds", required=True)
ap.add_argument("--shards", type=int, default=4); ap.add_argument("--cap", default="5e10")
ap.add_argument("--kfrac", type=float, default=0.70)
ap.add_argument("--tag", default="ext")
ap.add_argument("--interleave", action="store_true")
ap.add_argument("--list-missing", action="store_true",
                help="only print the missing seed files as TAG:N (one per line) and exit")
a = ap.parse_args()
run, code, seeds = map(os.path.abspath, (a.run, a.code, a.seeds))
eopt = dict(tuple(map(int, l.split())) for l in open(os.path.join(code, "eopt_all.txt")) if l.strip())
if a.list_missing:
    miss = set()
    for line in open(a.plan):
        line = line.split("#")[0].split()
        if not line: continue
        tag = ARMS[line[0]][2]
        for N in range(int(line[1]), int(line[2]) + 1):
            if tag and not os.path.exists(os.path.join(seeds, f"seeds_{tag}_N{N}.txt")): miss.add((tag, N))
    print("\n".join(f"{t}:{n}" for t, n in sorted(miss)))
    raise SystemExit(0)
os.makedirs(os.path.join(run, "parts"), exist_ok=True); os.makedirs(os.path.join(run, "tasks"), exist_ok=True)

tasks, n_done, n_total = [], 0, 0
for line in open(a.plan):
    line = line.split("#")[0].split()
    if not line: continue
    arm, nmin, nmax, R, S = line[0], int(line[1]), int(line[2]), int(line[3]), int(line[4])
    if arm not in ARMS: raise SystemExit(f"unknown arm {arm}")
    cap = str(int(float(line[5]))) if len(line) > 5 else str(int(float(a.cap)))
    kfrac = float(line[6]) if len(line) > 6 else a.kfrac
    binary, mode, tag, base, asp, sets = ARMS[arm]
    count_init = 1 if mode == 0 else 0
    kstop = min(S, max(S // 2 + 1, math.ceil(kfrac * S - 1e-9)))
    n_line = 0
    for N in range(nmin, nmax + 1):
        if N not in eopt: raise SystemExit(f"no Eopt for N={N}")
        sf = os.path.join(seeds, f"seeds_{tag}_N{N}.txt") if tag else "none"
        if tag and not os.path.exists(sf): raise SystemExit(f"missing seed file {sf}")
        extra = ""
        if sets is not None:
            extra = (sets.format(d8=max(1, round(N / 8)), d4=max(1, round(N / 4))) + " cfg=1").strip()
        for r in range(R):
            n_total += 1
            out = os.path.join(run, "parts", f"{arm}_N{N}_r{r}.txt")
            if os.path.exists(out) and sum(1 for _ in open(out)) == S:
                n_done += 1; continue
            cmd = (f"{code}/{binary} {N} {eopt[N]} {mode} 1 {S} {sf} {base} {cap} {r} {asp} {count_init} {kstop} {extra}"
                   .rstrip() + f" > {out}.tmp && mv {out}.tmp {out}")
            tasks.append((N * 1.37 ** N, arm, N, r, cmd)); n_line += 1
    print(f"  {arm:8s} N={nmin}-{nmax}  {binary} {('[' + extra + ']') if extra else ''}  cap={cap}  kstop={kstop}  queued {n_line}")

if a.interleave: tasks.sort(key=lambda t: (t[3], -t[0]))
else: tasks.sort(key=lambda t: -t[0])
load = [0.0] * a.shards; shard = [[] for _ in range(a.shards)]
for t in tasks:
    k = min(range(a.shards), key=lambda i: load[i]); load[k] += t[0]; shard[k].append(t)
for k in range(a.shards):
    with open(os.path.join(run, "tasks", f"shard_{a.tag}_{k}.txt"), "w") as f:
        f.write("".join(t[4] + "\n" for t in shard[k]))
with open(os.path.join(run, "tasks", f"manifest_{a.tag}.csv"), "w") as f:
    f.write("shard,arm,N,r,cost\n")
    for k in range(a.shards):
        for t in shard[k]: f.write(f"{k},{t[1]},{t[2]},{t[3]},{t[0]:.4g}\n")
print(f"{n_total} tasks in plan, {n_done} already complete, {len(tasks)} queued over {a.shards} shards")
