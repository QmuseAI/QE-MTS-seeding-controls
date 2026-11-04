# Paths, sourced by slurm/env.sh and the submit scripts. Everything is run from the project root
# (the step2_cluster directory):   bash slurm/submit_all.sh
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
CODE=$ROOT/code
SEEDS=$ROOT/seeds
RUN=${RUN:-$ROOT/run_main}           # results of the main experiment
PLAN=${PLAN:-$ROOT/slurm/plan.txt}
PY=${PY:-python}
