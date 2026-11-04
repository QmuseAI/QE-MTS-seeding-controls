#!/bin/bash
# Preparation of the seed-quality run (submitted by submit_seedq.sh):
#   1. seed files for A2k, A2m, A3, A8 (N=27-42) -> $ROOT/seeds_seedq, copied into $SEEDS (existing files are never
#      overwritten); their statistics are appended to the old ones in $SEEDS/seed_stats_all.csv
#   2. tagged task list RUN/tasks/shard_seedq_<k>.txt from slurm/plan_seedq.txt, interleaved by replicate
#SBATCH --job-name=qemts_prepq
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
source slurm/env.sh
set -euo pipefail
PLANQ=$ROOT/slurm/plan_seedq.txt
OUTQ=$ROOT/seeds_seedq
mkdir -p "$OUTQ"

# 1.
cd "$CODE"
$PY seeds.py --arms A2k A2m A3 A8 --nmin 27 --nmax 42 --R 100 --outdir "$OUTQ" --nproc 64 --stats seed_stats_seedq.csv
for f in "$OUTQ"/seeds_*.txt; do cp -n "$f" "$SEEDS/"; done
{ cat "$SEEDS/seed_stats.csv"; tail -n +2 "$OUTQ/seed_stats_seedq.csv"; } > "$SEEDS/seed_stats_all.csv"
echo "seed files in $SEEDS; statistics in $SEEDS/seed_stats_all.csv"

# 2.
$PY make_tasks.py --plan "$PLANQ" --run "$RUN" --code "$CODE" --seeds "$SEEDS" --shards "$NSHARDS" \
                  --tag seedq --interleave
echo PREP_SEEDQ_OK
