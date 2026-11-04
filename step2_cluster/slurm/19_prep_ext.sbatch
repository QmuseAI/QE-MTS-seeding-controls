#!/bin/bash
# Preparation of an extension run (submitted by submit_ext.sh):
#   1. compile code/mts4 (v4 = v3 + optional settings; code/mts is left untouched)
#   2. generate missing classical seed files (seeds.py, fixed RNG stream per (arm, N) -> identical to a full generation)
#      into $ROOT/seeds_ext and copy them into $SEEDS without overwriting; statistics -> $SEEDS/seed_stats_ext.csv
#      DCQO seeds (Q1, Q5) are not generated here: copy them from the GPU machine (gpu/dcqo_seeds.py).
#   3. tagged task list RUN/tasks/shard_<TAG>_<k>.txt, interleaved by replicate
#SBATCH --job-name=qemts_prepe
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=04:00:00
source slurm/env.sh
set -euo pipefail
: "${PLANX:?}" "${TAG:?}" "${NSHARDS:?}"
mkdir -p "$RUN"
cd "$CODE"

# 1.
if gcc -O3 -march=native -fopenmp -o mts4 mts4.c -lm 2>/dev/null; then :; else gcc -O3 -mcpu=native -fopenmp -o mts4 mts4.c -lm; fi
[ -x mts ] || { gcc -O3 -march=native -fopenmp -o mts mts.c 2>/dev/null || gcc -O3 -mcpu=native -fopenmp -o mts mts.c; }
{ echo "date: $(date -Is)  host: $(hostname)  plan: $PLANX  tag: $TAG"; echo "gcc: $(gcc --version | head -1)"
  sha256sum mts.c mts4.c mts4 make_tasks_ext.py seeds.py "$PLANX"; } >> "$RUN/provenance_ext.txt"

# 2.
MISS=$($PY make_tasks_ext.py --plan "$PLANX" --run "$RUN" --code "$CODE" --seeds "$SEEDS" --list-missing)
if [ -n "$MISS" ]; then
  echo "missing seed files: $(echo "$MISS" | paste -sd' ' -)"
  if echo "$MISS" | grep -qE '^Q'; then echo "ERROR: DCQO seed files missing; run gpu/dcqo_seeds.py and copy them into $SEEDS"; exit 1; fi
  NLIST=$(echo "$MISS" | awk -F: '{if(!($1 in lo)||$2<lo[$1])lo[$1]=$2; if($2>hi[$1])hi[$1]=$2} END{for(t in lo) printf "%s:%d-%d\n", t, lo[t], hi[t]}' | paste -sd, -)
  ARMS=$(echo "$NLIST" | tr ',' '\n' | cut -d: -f1 | paste -sd' ' -)
  mkdir -p "$ROOT/seeds_ext"
  $PY seeds.py --arms $ARMS --nlist "$NLIST" --R 100 --outdir "$ROOT/seeds_ext" --nproc 64 --stats seed_stats_ext_${TAG}.csv
  for f in "$ROOT"/seeds_ext/seeds_*.txt; do cp -n "$f" "$SEEDS/"; done
  if [ -f "$SEEDS/seed_stats_ext.csv" ]; then tail -n +2 "$ROOT/seeds_ext/seed_stats_ext_${TAG}.csv" >> "$SEEDS/seed_stats_ext.csv"
  else cp "$ROOT/seeds_ext/seed_stats_ext_${TAG}.csv" "$SEEDS/seed_stats_ext.csv"; fi
fi

# 3.
$PY make_tasks_ext.py --plan "$PLANX" --run "$RUN" --code "$CODE" --seeds "$SEEDS" --shards "$NSHARDS" \
                      --tag "$TAG" --interleave
echo PREP_EXT_OK
