#!/bin/bash
# Standalone DCQO seed package. Upload this whole folder (gpu_dcqo/) to the GPU server and run inside it.
#   bash run_dcqo.sh env                      # 0) check python, cudaq, GPUs
#   bash run_dcqo.sh                          # 1) self-test  2) Trotter convergence at N=27  3) time estimate
#   STEPS=16 [SIGN=-1] bash run_dcqo.sh seeds # 4) seeds N=27..33, one GPU per N in parallel, then merge
#   STEPS=16 [SIGN=-1] bash run_dcqo.sh big   # optional: N=34,35,36 on all 8 GPUs (mpirun), one after the other
#   bash run_dcqo.sh pack                     # 5) results -> dcqo_results.tar.gz (send this file back)
# Options: NLIST="27 28 ... 33" (sizes for 'seeds'), GPUS=8, PY=python
set -euo pipefail
cd "$(dirname "$0")"
STEPS=${STEPS:-16}; SIGN=${SIGN:--1}; OUT=seeds_dcqo; PY=${PY:-python}; GPUS=${GPUS:-8}
NLIST=${NLIST:-"27 28 29 30 31 32 33"}
mkdir -p "$OUT" logs
case "${1:-check}" in
  env)
    $PY -c "import sys, numpy, scipy; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
    $PY -c "import cudaq; print('cudaq', cudaq.__version__); cudaq.set_target('nvidia', option='fp32'); print('target nvidia/fp32 OK')"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv ;;
  check)
    $PY dcqo_seeds.py --selftest                                        2>&1 | tee logs/selftest.txt
    CUDA_VISIBLE_DEVICES=0 $PY dcqo_seeds.py --converge 27 --steps 1 2 4 8 16 32 --sign "$SIGN" 2>&1 | tee logs/converge_N27.txt
    CUDA_VISIBLE_DEVICES=0 $PY dcqo_seeds.py --estimate 27 29 31 33 --sign "$SIGN" 2>&1 | tee logs/estimate.txt ;;
  seeds)
    g=0
    for N in $NLIST; do
      CUDA_VISIBLE_DEVICES=$g nohup $PY dcqo_seeds.py --N $N --steps "$STEPS" --sign "$SIGN" --outdir "$OUT" > logs/seeds_N$N.txt 2>&1 &
      g=$(( (g + 1) % GPUS ))
    done
    wait
    $PY dcqo_seeds.py --merge --outdir "$OUT"; tail -n 2 logs/seeds_N*.txt ;;
  big)
    for N in 34 35 36; do
      mpirun -np "$GPUS" $PY dcqo_seeds.py --N $N --steps "$STEPS" --sign "$SIGN" --mgpu --outdir "$OUT" > logs/seeds_N$N.txt 2>&1
    done
    $PY dcqo_seeds.py --merge --outdir "$OUT" ;;
  pack)
    tar czf dcqo_results.tar.gz "$OUT" logs && ls -lh dcqo_results.tar.gz ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
