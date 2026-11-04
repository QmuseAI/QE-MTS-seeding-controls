#!/bin/bash
# Merge parts, check completeness, run the full analysis (single node, a few cores).
# ANALYSIS (optional): name of the output folder under RUN (default "analysis"), e.g. ANALYSIS=analysis_extra.
#SBATCH --job-name=qemts_analyze
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
source slurm/env.sh
set -euo pipefail
OUTDIR=$RUN/${ANALYSIS:-analysis}
mkdir -p "$OUTDIR"
cd "$CODE"
$PY collect.py --run "$RUN" --plan "$PLAN" | tee "$OUTDIR/collect.txt"
PAPER=$ROOT/../step1_digitize/data/fig5_logratio_vector.csv
PAPER2=$ROOT/../step1_digitize/data/fig2_q50_digitized.csv
OPT=""
[ -f "$PAPER" ] && OPT="$OPT --paper $PAPER"
[ -f "$PAPER2" ] && OPT="$OPT --paperfig2 $PAPER2"
$PY analyze.py --data "$RUN/data" --out "$OUTDIR" --ref A0 --nboot 2000 \
    --seedstats "${SEEDSTATS:-$SEEDS/seed_stats.csv}" $OPT
echo ANALYZE_OK
