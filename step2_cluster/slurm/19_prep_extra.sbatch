#!/bin/bash
# Preparation of the extra run (submitted by submit_extra.sh; runs after the previous run AND its analysis, so that
# analysis still sees the old files):
#   1. A0/A1/A2 at N=41,43,45: replicate files with fewer than KSTOP=70 finished runs (the 70th-fastest TTS was cut
#      off by the old 3e9 cap) are moved to RUN/parts_cap3e9/, so make_tasks.py queues them again.
#      A file already present in parts_cap3e9/ is never overwritten (safe to run this job again).
#   2. Seed files for A1/A2 at N=46,47 are generated if missing. seeds.py uses a fixed RNG stream per (arm, N), so
#      these are exactly the files the full seed generation would have written; seed_stats.csv is not touched.
#   3. Tagged task list RUN/tasks/shard_extra_<k>.txt from slurm/plan_extra.txt, interleaved by replicate.
#SBATCH --job-name=qemts_prepx
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
source slurm/env.sh
set -euo pipefail
KSTOP=${KSTOP:-70}
PLANX=$ROOT/slurm/plan_extra.txt

# 1.
mkdir -p "$RUN/parts_cap3e9"
nmv=0
for A in A0 A1 A2; do for n in 41 43 45; do
  for f in "$RUN"/parts/${A}_N${n}_r*.txt; do
    [ -e "$f" ] || continue
    nf=$(awk '$3 > 0' "$f" | wc -l)
    if [ "$nf" -lt "$KSTOP" ] && [ ! -e "$RUN/parts_cap3e9/$(basename "$f")" ]; then
      mv "$f" "$RUN/parts_cap3e9/"; nmv=$((nmv + 1))
    fi
  done
done; done
echo "moved $nmv replicate files (< $KSTOP finished runs under the old cap) to $RUN/parts_cap3e9/"

# 2.
miss=0
for A in A1 A2; do for n in 46 47; do [ -f "$SEEDS/seeds_${A}_N${n}.txt" ] || miss=1; done; done
if [ "$miss" = 1 ]; then
  (cd "$CODE" && $PY seeds.py --arms A1 A2 --nmin 46 --nmax 47 --R 100 --outdir "$ROOT/seeds_extra" --nproc 8)
  for A in A1 A2; do for n in 46 47; do cp -n "$ROOT/seeds_extra/seeds_${A}_N${n}.txt" "$SEEDS/"; done; done
  echo "seed files for N=46,47 written to $SEEDS"
fi

# 3.
cd "$CODE"
$PY make_tasks.py --plan "$PLANX" --run "$RUN" --code "$CODE" --seeds "$SEEDS" --shards "$NSHARDS" \
                  --tag extra --interleave
echo PREP_EXTRA_OK
