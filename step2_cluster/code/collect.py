"""Merge per-replicate part files into one file per (arm, N) and check completeness.

usage: python collect.py --run RUN --plan ../slurm/plan.txt
Writes RUN/data/<arm>_N<N>.txt  (columns: r s evals best_energy; evals<0 means censored after |evals|)
Prints missing / incomplete tasks, censored runs, and the total number of objective evaluations spent.
"""
import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--plan", required=True)
a = ap.parse_args()
os.makedirs(os.path.join(a.run, "data"), exist_ok=True)
missing, capped, total = [], 0, 0
for line in open(a.plan):
    line = line.split("#")[0].split()
    if not line: continue
    arm, nmin, nmax, R, S = line[0], int(line[1]), int(line[2]), int(line[3]), int(line[4])
    for N in range(nmin, nmax + 1):
        rows = []
        for r in range(R):
            p = os.path.join(a.run, "parts", f"{arm}_N{N}_r{r}.txt")
            L = open(p).read().split("\n") if os.path.exists(p) else []
            L = [l for l in L if l.strip()]
            if len(L) != S: missing.append(f"{arm} N={N} r={r} ({len(L)}/{S} lines)")
            rows += L
        for l in rows:
            e = int(l.split()[2]); capped += e < 0; total += abs(e)
        with open(os.path.join(a.run, "data", f"{arm}_N{N}.txt"), "w") as f: f.write("\n".join(rows) + "\n")
print(f"total objective evaluations spent: {total:.4g}; censored runs (early stop or cap): {capped}")
if missing:
    print(f"{len(missing)} incomplete tasks, e.g.:"); print("\n".join(missing[:20]))
else:
    print("all tasks complete")
