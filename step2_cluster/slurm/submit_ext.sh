#!/bin/bash
# Extension runs (from step2_cluster, on the login node). Results go to run_ext/ (run_main/ is only read).
#   bash slurm/submit_ext.sh                                                   # plan_ext.txt   (~3-4 h)
#   TAG=hard THREADS=32 PLANX=slurm/plan_hard.txt bash slurm/submit_ext.sh     # plan_hard.txt  (~5-7 h)
#   TAG=dcqo PLANX=slurm/plan_dcqo.txt bash slurm/submit_ext.sh                # after the DCQO seeds are in seeds/
# Chain: prep (19_prep_ext) -> run array (20_run, TAG) -> collect + analysis (32_analyze_ext -> run_ext/analysis_ext)
# Options: AFTER (job ids to wait for), TIME (run wall time, default 24:00:00), THREADS (default 8),
#          and the node options of submit_all.sh (N, EXCLUDE, EXTRA, WORKERS).
# Stop/continue: as submit_seedq.sh (finished replicates are kept and skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
source slurm/common.sh
export RUN=${RUN:-$(pwd)/run_ext}
SLURM_SUBMIT_DIR=$(pwd) source slurm/config.sh
PLANX=$(readlink -f "${PLANX:-slurm/plan_ext.txt}"); TAG=${TAG:-ext}
AFTER=${AFTER:-}; TIME=${TIME:-24:00:00}; THREADS=${THREADS:-8}
deps=""
for j in $AFTER; do [ -n "$(squeue -h -j "$j" 2>/dev/null || true)" ] && deps="$deps:$j"; done
DEP0=""; [ -n "$deps" ] && DEP0="--dependency=afterany$deps"
j0=$(sbatch --parsable --hold $EXTRA --export=ALL,RUN=$RUN,PLANX=$PLANX,TAG=$TAG,NSHARDS=$N $DEP0 slurm/19_prep_ext.sbatch)
j1=$(sbatch --parsable $EXTRA --export=ALL,RUN=$RUN,TAG=$TAG,NSHARDS=$N,WORKERS=$WORKERS,THREADS=$THREADS \
     --time=$TIME --cpus-per-task=$CPUS --array=0-$((N-1)) --dependency=afterok:$j0 slurm/20_run.sbatch)
j2=$(sbatch --parsable $EXTRA --export=ALL,RUN=$RUN --dependency=afterany:$j1 slurm/32_analyze_ext.sbatch)
scontrol release "$j0"
echo "submitted ($TAG, $PLANX): prep=$j0 ${DEP0:-}  run=$j1 (array 0-$((N-1)), $THREADS threads/task, time $TIME)  analyze=$j2"
echo "results: $RUN/analysis_ext/"
