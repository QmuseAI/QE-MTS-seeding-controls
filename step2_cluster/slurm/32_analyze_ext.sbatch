#!/bin/bash
# Merge the part files of the extension runs and analyse them together with run_main (single node).
#   output: $RUN/analysis_ext/ (summary_ext.txt, *.csv, fig_ext_*.png)
#SBATCH --job-name=qemts_anaext
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
source slurm/env.sh
set -euo pipefail
OUTDIR=$RUN/analysis_ext
mkdir -p "$OUTDIR"
cd "$CODE"
for P in "$ROOT"/slurm/plan_ext.txt "$ROOT"/slurm/plan_hard.txt "$ROOT"/slurm/plan_dcqo.txt; do
  t=$(basename "$P" .txt)
  ls "$RUN"/parts/ >/dev/null 2>&1 || continue
  $PY collect.py --run "$RUN" --plan "$P" > "$OUTDIR/collect_${t}.txt" || true
done
SS="$SEEDS/seed_stats_all.csv"
[ -f "$SEEDS/seed_stats_ext.csv" ] && SS="$SS $SEEDS/seed_stats_ext.csv"
[ -f "$SEEDS/seed_stats_dcqo.csv" ] && SS="$SS $SEEDS/seed_stats_dcqo.csv"
$PY analyze_ext.py --data "$ROOT/run_main/data" "$RUN/data" --seedstats $SS \
    --paper "$ROOT/../step1_digitize/data/fig5_logratio_vector.csv" --out "$OUTDIR" --nboot 2000
echo ANALYZE_EXT_OK
