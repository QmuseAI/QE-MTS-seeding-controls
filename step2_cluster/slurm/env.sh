# Sourced by every job script; do not run on its own. Same setup as QEC_SB_paper1/code/slurm/env.sh.
# The analysis needs only numpy, scipy, matplotlib -> the existing conda env "qecsb" is enough
# (set CONDA_ENV=... to use another one).
module load Core/anaconda3-2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-qecsb}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to cd to $SLURM_SUBMIT_DIR"; exit 1; }
source slurm/config.sh
# worker processes per node: a few logical CPUs are left free on purpose (OS, I/O, xargs itself)
WORKERS=${WORKERS:-500}
