#!/bin/bash
# Production run, job array: element i = one node = shard i of the (arm, N, replicate) task list.
# v3: every task (one replicate, S runs) uses $THREADS OpenMP threads (default 8), and WORKERS/THREADS tasks
# run at the same time -> WORKERS threads per node (default 500 of 512 logical CPUs; node held exclusively).
# TTS is counted in objective evaluations, so CPU speed / SMT / thread count change only the wall time, never the results.
# Restart-safe: completed tasks are skipped when the task list is rebuilt (bash slurm/resubmit_run.sh).
# TAG (optional): read the tagged task list RUN/tasks/shard_<TAG>_<i>.txt (make_tasks.py --tag), e.g. TAG=extra.
#SBATCH --job-name=qemts_run
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=512
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=08:00:00
source slurm/env.sh
set -euo pipefail
THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS OMP_WAIT_POLICY=PASSIVE OMP_PROC_BIND=false
P=$(( WORKERS / THREADS )); [ "$P" -lt 1 ] && P=1
SHARD=$RUN/tasks/shard_${TAG:+${TAG}_}${SLURM_ARRAY_TASK_ID}.txt
echo "$(date -Is) $(hostname) shard $SLURM_ARRAY_TASK_ID/$NSHARDS: $(wc -l < "$SHARD") tasks, $P tasks x $THREADS threads"
xargs -d '\n' -P "$P" -I{} bash -c '{}' < "$SHARD"
echo "$(date -Is) shard $SLURM_ARRAY_TASK_ID done"
