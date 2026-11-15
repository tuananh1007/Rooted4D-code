#!/bin/bash
#SBATCH --job-name=rooted4d-thmatch
#SBATCH --account=bhio-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
# Rescore TetHeart exactly as E36 scores our own fits: whole cavity boundary set,
# the metric's default samples per face. E49f measured the gap this closes.
NVME=/work/nvme/bhio/avu2/Rooted4D
RES=$NVME/results/22102244_thfinal
cd $NVME/code; mkdir -p logs
export PYTHONPATH=$NVME/code/Codes:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export R4D_ACDC=/work/hdd/bhio/avu2/Rooted4D/data/raw/acdc
export R4D_TET_FRAME_REPORT=$NVME/results/22045910_tetframe/E50_tet_frame.json
export R4D_OUT=$RES
export R4D_WORKERS=8
export R4D_MATCH_E36=1
R4D_PRED_DIR=$RES/preds_test_iter_60000 R4D_OUTNAME=E49g_test_iter_60000_matched.json \
  $NVME/external/.venv-tetheart/bin/python Codes/scripts/experiments/e49d_tetheart_rescore.py
echo "finished $(date -u +%H:%M:%SZ)"
