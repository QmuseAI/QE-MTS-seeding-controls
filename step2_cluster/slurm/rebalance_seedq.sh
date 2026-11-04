#!/bin/bash
# Rebalance the seed-quality run when one shard of the run array lags behind.
# (2026-10-22: shard 3 of array 10182 still had 1522 of 1671 tasks, because make_tasks.py --interleave deals
#  equal-cost tasks in plan order, so every 4th task -- the slow A8 arm -- landed on the same shard.)
#
#   DROP="10183 10184" bash slurm/rebalance_seedq.sh 10182 3
#        $1      = job id of the running seedq array
#        $2...   = lagging array element(s) to cancel (their unfinished tasks are redistributed)
#        DROP    = job ids of the old, still pending analyze + comment jobs (cancelled and resubmitted)
# Optional: EXCLUDE_RE='A8_N4[12]_' leaves matching tasks out of the new list (e.g. to skip A8 at N=41-42).
#           TIME (default 24:00:00), THREADS (default 8), node options as in submit_all.sh (N, EXCLUDE, EXTRA, WORKERS).
#
# What it does:
#   1. cancels the lagging element(s) and the jobs in DROP, waits until they have left the queue;
#   2. deletes the half-written .tmp files of the cancelled shard(s) only (runs are deterministic -> rerun = same result);
#   3. rebuilds the list of unfinished tasks (make_tasks.py skips complete outputs), drops tasks whose .tmp file still
#      exists (= running on the other, untouched elements), shuffles the rest with a fixed seed and deals them
#      round-robin to the shards -> RUN/tasks/shard_seedq2_<k>.txt;
#   4. submits run (TAG=seedq2) + analysis + comment; the analysis waits for the old array AND the new one.
set -euo pipefail
cd "$(dirname "$0")/.."
OLD=${1:?usage: DROP=\"ids\" bash slurm/rebalance_seedq.sh ARRAY_ID ELEMENT...}; shift
[ $# -gt 0 ] || { echo "give the lagging array element(s), e.g. 3"; exit 1; }
LAG=("$@")
DROP=${DROP:?set DROP to the pending analyze/comment job ids of the old chain, e.g. DROP=\"10183 10184\"}
mkdir -p logs
source slurm/common.sh
SLURM_SUBMIT_DIR=$(pwd) source slurm/config.sh
module load Core/anaconda3-2024.02-1; eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV:-qecsb}"
THREADS=${THREADS:-8}; TIME=${TIME:-24:00:00}; EXCLUDE_RE=${EXCLUDE_RE:-}

# 1.
for j in $DROP; do scancel "$j" || true; done
for e in "${LAG[@]}"; do scancel "${OLD}_${e}" || true; done
for e in "${LAG[@]}"; do
  while [ -n "$(squeue -h -j "${OLD}_${e}" 2>/dev/null || true)" ]; do sleep 5; done
done
echo "cancelled: ${LAG[*]/#/${OLD}_}  and  $DROP"
sleep 10                                   # let the killed processes release their files

# 2.
nrm=0
for e in "${LAG[@]}"; do
  f=$RUN/tasks/shard_seedq_${e}.txt
  for t in $(grep -o '[^ ]*parts/[^ ]*\.txt\.tmp' "$f" | sort -u); do
    if [ -e "$t" ]; then rm -f "$t"; nrm=$((nrm + 1)); fi
  done
done
echo "removed $nrm half-written .tmp files of the cancelled shard(s)"

# 3.
(cd "$CODE" && $PY make_tasks.py --plan "$ROOT/slurm/plan_seedq.txt" --run "$RUN" --code "$CODE" --seeds "$SEEDS" \
                                 --shards "$N" --tag seedq2)
$PY - "$RUN/tasks" "$N" "$EXCLUDE_RE" <<'EOF'
import os, re, sys, random
tdir, n, excl = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines = []
for k in range(n):
    with open(os.path.join(tdir, f"shard_seedq2_{k}.txt")) as f: lines += [l for l in f if l.strip()]
keep, running, skipped = [], 0, 0
for l in lines:
    tmp = re.search(r"(\S*parts/\S*\.txt\.tmp)", l).group(1)
    if os.path.exists(tmp): running += 1; continue          # still running on an untouched element
    if excl and re.search(excl, l): skipped += 1; continue
    keep.append(l)
random.Random(20261022).shuffle(keep)
for k in range(n):
    with open(os.path.join(tdir, f"shard_seedq2_{k}.txt"), "w") as f: f.writelines(keep[k::n])
print(f"{len(lines)} unfinished tasks: {running} still running elsewhere (left alone), {skipped} excluded, "
      f"{len(keep)} dealt to {n} shards ({', '.join(str(len(keep[k::n])) for k in range(n))})")
EOF

# 4.
j1=$(sbatch --parsable $EXTRA --export=ALL,TAG=seedq2,NSHARDS=$N,WORKERS=$WORKERS,THREADS=$THREADS \
     --time=$TIME --cpus-per-task=$CPUS --array=0-$((N-1)) slurm/20_run.sbatch)
j2=$(sbatch --parsable $EXTRA \
     --export=ALL,PLAN=$ROOT/slurm/plan_all2.txt,ANALYSIS=analysis_seedq,SEEDSTATS=$SEEDS/seed_stats_all.csv \
     --dependency=afterany:$OLD:$j1 slurm/30_analyze.sbatch)
j3=$(sbatch --parsable $EXTRA --export=ALL,ANALYSIS=analysis_seedq --dependency=afterok:$j2 slurm/31_comment.sbatch)
echo "submitted: run=$j1 (array 0-$((N-1)), time $TIME)  analyze=$j2 (after $OLD and $j1)  comment=$j3"
