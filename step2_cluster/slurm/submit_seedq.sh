#!/bin/bash
# Seed-quality sweep (A2k, A2m, A3, A8 at N=27-42) + early-stop validation (A0f, A2f at N=27-33, 20 replicates).
#   bash slurm/submit_seedq.sh                 (from step2_cluster, on the login node)
# Chain: prep (19_prep_seedq) -> run array (20_run, TAG=seedq) -> analysis of all data (30_analyze ->
#        run_main/analysis_seedq) -> early-stop check + Comment figures (31_comment -> comment/out)
# Options: AFTER (job ids to wait for, default none), TIME (run wall time, default 24:00:00), THREADS (default 8),
#          and the node options of submit_all.sh (N, EXCLUDE, EXTRA, WORKERS).
# Stop early: scancel the run array; finished replicates are kept and the analysis still runs (afterany).
# Continue later: run this script again; finished replicates are skipped (seed files are not regenerated).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs comment/out
source slurm/common.sh
SLURM_SUBMIT_DIR=$(pwd) source slurm/config.sh
AFTER=${AFTER:-}; TIME=${TIME:-24:00:00}; THREADS=${THREADS:-8}

deps=""
for j in $AFTER; do [ -n "$(squeue -h -j "$j" 2>/dev/null || true)" ] && deps="$deps:$j"; done
DEP0=""; [ -n "$deps" ] && DEP0="--dependency=afterany$deps"

j0=$(sbatch --parsable --hold $EXTRA --export=ALL,NSHARDS=$N $DEP0 slurm/19_prep_seedq.sbatch)
j1=$(sbatch --parsable $EXTRA --export=ALL,TAG=seedq,NSHARDS=$N,WORKERS=$WORKERS,THREADS=$THREADS \
     --time=$TIME --cpus-per-task=$CPUS --array=0-$((N-1)) --dependency=afterok:$j0 slurm/20_run.sbatch)
j2=$(sbatch --parsable $EXTRA \
     --export=ALL,PLAN=$ROOT/slurm/plan_all2.txt,ANALYSIS=analysis_seedq,SEEDSTATS=$SEEDS/seed_stats_all.csv \
     --dependency=afterany:$j1 slurm/30_analyze.sbatch)
j3=$(sbatch --parsable $EXTRA --export=ALL,ANALYSIS=analysis_seedq --dependency=afterok:$j2 slurm/31_comment.sbatch)
scontrol release "$j0"
echo "submitted: prep=$j0 (${DEP0:-no dependency})  run=$j1 (array 0-$((N-1)), time $TIME)  analyze=$j2  comment=$j3"
echo "results: run_main/analysis_seedq/  and  comment/out/"
