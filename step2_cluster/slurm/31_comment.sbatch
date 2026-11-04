#!/bin/bash
# After the analysis: early-stop check and all numbers/figures for the Comment.
#   -> RUN/analysis_seedq/earlystop_check.txt   and   $ROOT/comment/out/ (comment_numbers.txt, fig*.pdf/png, tables)
#SBATCH --job-name=qemts_comment
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
source slurm/env.sh
set -euo pipefail
OUTDIR=$RUN/${ANALYSIS:-analysis_seedq}
cd "$CODE"
$PY check_earlystop.py --run "$RUN" | tee "$OUTDIR/earlystop_check.txt"
$PY make_comment.py --run "$RUN" --analysis "$OUTDIR" --step1 "$ROOT/../step1_digitize/data" \
    --seedstats "$SEEDS/seed_stats_all.csv" --out "$ROOT/comment/out"
echo COMMENT_OK
