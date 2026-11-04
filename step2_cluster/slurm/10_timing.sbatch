#!/bin/bash
# Evaluation rate with $WORKERS processes busy on one node (same load as the production run), at N=37 and
# N=42 (arm A0, 4 seeds each); then projects the cost of plan.txt -> $RUN/timing/estimate.txt
# (informational only; these runs are not used in the analysis).
#SBATCH --job-name=qemts_timing
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=512
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:00:00
source slurm/env.sh
set -euo pipefail
T=$RUN/timing; mkdir -p "$T"; cd "$CODE"
for N in 37 42; do
  E=$(awk -v n=$N '$1==n{print $2}' eopt_all.txt)
  seq 0 $((WORKERS-1)) | xargs -P "$WORKERS" -I{} bash -c \
    "s=\$(date +%s.%N); ./mts $N $E 0 1 4 none 999 1000000000000 {} 1 > $T/N${N}_r{}.txt; e=\$(date +%s.%N); echo \"{} \$s \$e\" > $T/N${N}_r{}.time"
done
$PY estimate_cost.py --timing "$T" --plan "$PLAN" --cpus "$WORKERS" --nodes "${NSHARDS:?}" | tee "$T/estimate.txt"
