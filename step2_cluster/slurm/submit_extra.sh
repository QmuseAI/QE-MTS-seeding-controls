#!/bin/bash
# Extra run: A0/A1/A2 at N=43, 45, 47 (and the few cap-limited replicates of 41) with cap 5e10, N=46 normal.
# Queued behind the running job and its analysis; the final analysis covers everything (N=20-48).
#   bash slurm/submit_extra.sh                 (from step2_cluster, on the login node)
# Chain: prep (19_prep_extra) -> run array (20_run, TAG=extra) -> analysis of all data (30_analyze -> run_main/analysis_extra)
# Options: AFTER (default "10166 10167": wait for these jobs if they are still in the queue), TIME (run wall time,
#          default 24:00:00), THREADS (default 8), and the node options of submit_all.sh (N, EXCLUDE, EXTRA, WORKERS).
# Stop early: scancel the run array; finished replicates are kept and the analysis still runs (afterany).
# Continue later: run this script again; finished replicates are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
source slurm/common.sh
SLURM_SUBMIT_DIR=$(pwd) source slurm/config.sh
AFTER=${AFTER:-"10166 10167"}; TIME=${TIME:-24:00:00}; THREADS=${THREADS:-8}

deps=""
for j in $AFTER; do [ -n "$(squeue -h -j "$j" 2>/dev/null || true)" ] && deps="$deps:$j"; done
DEP0=""; [ -n "$deps" ] && DEP0="--dependency=afterany$deps"

# prep is submitted on hold and released only after all three jobs were accepted
j0=$(sbatch --parsable --hold $EXTRA --export=ALL,NSHARDS=$N $DEP0 slurm/19_prep_extra.sbatch)
j1=$(sbatch --parsable $EXTRA --export=ALL,TAG=extra,NSHARDS=$N,WORKERS=$WORKERS,THREADS=$THREADS \
     --time=$TIME --cpus-per-task=$CPUS --array=0-$((N-1)) --dependency=afterok:$j0 slurm/20_run.sbatch)
j2=$(sbatch --parsable $EXTRA --export=ALL,PLAN=$ROOT/slurm/plan_all.txt,ANALYSIS=analysis_extra \
     --dependency=afterany:$j1 slurm/30_analyze.sbatch)
scontrol release "$j0"
echo "submitted: prep=$j0 (${DEP0:-no dependency})  run=$j1 (array 0-$((N-1)), time $TIME)  analyze=$j2"
echo "results will be in run_main/analysis_extra/"
