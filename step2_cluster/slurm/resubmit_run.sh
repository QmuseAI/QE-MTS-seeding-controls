#!/bin/bash
# After a timeout / killed node: rebuild the task list with only the missing tasks and resubmit
# the run array plus the analysis. Completed (arm, N, replicate) files are kept and skipped.
#   bash slurm/resubmit_run.sh            (same node options as submit_all.sh: N, EXCLUDE, EXTRA, WORKERS)
# v3 options: CAP (per-run cap, default 3e9), KFRAC (early stop, default 0.70), THREADS (per task, default 8)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
source slurm/common.sh
SLURM_SUBMIT_DIR=$(pwd) source slurm/config.sh
module load Core/anaconda3-2024.02-1; eval "$(conda shell.bash hook)"; conda activate "${CONDA_ENV:-qecsb}"
CAP=${CAP:-3000000000}; KFRAC=${KFRAC:-0.70}; THREADS=${THREADS:-8}
(cd "$CODE" && $PY make_tasks.py --plan "$PLAN" --run "$RUN" --code "$CODE" --seeds "$SEEDS" --shards "$N" \
                                 --cap "$CAP" --kfrac "$KFRAC")
EXP="--export=ALL,NSHARDS=$N,WORKERS=$WORKERS,THREADS=$THREADS"
j1=$(sbatch --parsable $EXTRA $EXP --cpus-per-task=$CPUS --array=0-$((N-1)) slurm/20_run.sbatch)
j2=$(sbatch --parsable $EXTRA $EXP --dependency=afterany:$j1 slurm/30_analyze.sbatch)
echo "resubmitted: run=$j1 analyze=$j2   (cap=$CAP kfrac=$KFRAC threads=$THREADS)"
