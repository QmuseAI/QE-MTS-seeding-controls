#!/bin/bash
# Compile on a compute node, verify the optimum table, generate all seed files, build the task list
# split into $NSHARDS shards (one per node; NSHARDS is set by submit_all.sh).
#SBATCH --job-name=qemts_prep
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=512
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=03:00:00
source slurm/env.sh
set -euo pipefail
mkdir -p "$RUN"
cd "$CODE"

# 1. compile with the native flags of the compute nodes (x86-64: -march, aarch64: -mcpu)
if gcc -O3 -march=native -fopenmp -o mts mts.c 2>/dev/null; then FLAGS="-O3 -march=native -fopenmp"
else gcc -O3 -mcpu=native -fopenmp -o mts mts.c; FLAGS="-O3 -mcpu=native -fopenmp"; fi

# 2. provenance (reproducibility statement of the paper)
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "slurm job: ${SLURM_JOB_ID:-none}"
  echo "uname: $(uname -a)"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket' || true
  echo "gcc: $(gcc --version | head -1)  flags: $FLAGS"
  echo "python: $($PY --version 2>&1)  numpy: $($PY -c 'import numpy;print(numpy.__version__)')  scipy: $($PY -c 'import scipy;print(scipy.__version__)')"
  echo "sha256:"; sha256sum mts.c mts labs_common.py seeds.py make_tasks.py analyze.py "$PLAN"
} > "$RUN/provenance.txt"

# 3. optimum table: energies recomputed from the published optimal sequences, then MTS runs aimed
#    below the optimum (must never succeed) as an independent check
$PY verify_eopt.py --mts ./mts --runs 4 | tee "$RUN/verify_eopt.txt"

# 4. seeds (deterministic; per-arm N ranges taken from the plan)
NLIST=$(awk '!/^#/ && NF>=5 {print $1":"$2"-"$3}' "$PLAN" | grep -E '^A(1|2|5|6|7a|7b):' | paste -sd, -)
ARMS=$(echo "$NLIST" | tr ',' '\n' | cut -d: -f1 | paste -sd' ' -)
$PY seeds.py --arms $ARMS --nlist "$NLIST" --R 100 --outdir "$SEEDS" --nproc "$WORKERS"

# 5. task list, one shard per node
$PY make_tasks.py --plan "$PLAN" --run "$RUN" --code "$CODE" --seeds "$SEEDS" --shards "${NSHARDS:?}"
echo PREP_OK
