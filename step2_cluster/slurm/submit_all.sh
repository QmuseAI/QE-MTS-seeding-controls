#!/bin/bash
# Run on the login node from the project root (step2_cluster):   bash slurm/submit_all.sh
# Submits the whole chain with dependencies:
#   prep (1 node) -> timing (1 node, informational) -> run (array, one element per node) -> analyze
# Uses every usable node (set EXCLUDE=nXXX to leave one out), WORKERS=500 processes per node.
#   N=4 bash slurm/submit_all.sh                  fix the number of nodes
#   EXCLUDE=n013 bash slurm/submit_all.sh         leave n013 out
#   EXTRA="--partition=xxx" bash slurm/submit_all.sh
#   SKIP_TIMING=1 bash slurm/submit_all.sh        no timing job
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
source slurm/common.sh
EXP="--export=ALL,NSHARDS=$N,WORKERS=$WORKERS,THREADS=${THREADS:-8}"

j0=$(sbatch --parsable $EXTRA $EXP slurm/00_prep.sbatch)
DEP=$j0
if [ -z "${SKIP_TIMING:-}" ]; then
  jt=$(sbatch --parsable $EXTRA $EXP --dependency=afterok:$j0 slurm/10_timing.sbatch); DEP=$jt
fi
j1=$(sbatch --parsable $EXTRA $EXP --cpus-per-task=$CPUS --array=0-$((N-1)) --dependency=afterok:$DEP slurm/20_run.sbatch)
j2=$(sbatch --parsable $EXTRA $EXP --dependency=afterany:$j1 slurm/30_analyze.sbatch)
echo "submitted: prep=$j0 timing=${jt:-skipped} run=$j1 (array 0-$((N-1))) analyze=$j2"
echo "watch: squeue -u $USER    logs: logs/    estimate: run_main/timing/estimate.txt"
