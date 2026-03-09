#!/usr/bin/env bash
# =============================================================================
# retrain_adapter.sh — Retrain the w200_wmissing_adapter model
#
# Replicates the original two-stage training:
#   Stage 1: broad+adapter — already done in run_fresh.sh
#   Stage 2: Resume from broad+adapter on w200 missingness data (lr=3e-5)
#
# The original was trained by resuming the full broad+adapter model
# (backbone + adapter weights) on the w200 missingness dataset,
# NOT by creating a fresh adapter from w200_wmissing.
#
# Usage:
#   ./scripts/retrain_adapter.sh
# =============================================================================
set -euo pipefail

BASE_DIR="${BASE_DIR:-/sietch_colab/data_share/cxt_scratch}"
CKPT_DIR="${BASE_DIR}/lightning_logs"
CKPT_CACHE="${BASE_DIR}/checkpoints"

GPUS="0 1"
TRAIN_WORKERS=16

export CXT_CHECKPOINT_CACHE="${CKPT_CACHE}"
export CUDA_VISIBLE_DEVICES="${GPUS// /,}"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

VENV_DIR="${BASE_DIR}/.venv"
if [ -d "${VENV_DIR}" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${REPO_ROOT}"

# Stage 1 checkpoint: broad+adapter (trained on w2000, 10 samples, no bitmask)
BROAD_ADAPTER_CKPT="${CKPT_CACHE}/broad+adapter/broad_adapter_epoch=2-step=792.ckpt"
# Stage 2 data: w200, 10 samples, with bitmask
PROCESSED_SWM_N10="${BASE_DIR}/data/ts_large_pop/processed_small_window_missing_data_n10"

if [ ! -f "${BROAD_ADAPTER_CKPT}" ]; then
    echo "ERROR: broad+adapter checkpoint not found at ${BROAD_ADAPTER_CKPT}"
    echo "Run the 'train' stage of run_fresh.sh first."
    exit 1
fi
if [ ! -d "${PROCESSED_SWM_N10}" ]; then
    echo "ERROR: adapter training data not found at ${PROCESSED_SWM_N10}"
    exit 1
fi

echo "============================================"
echo " Retraining w200_wmissing_adapter (stage 2)"
echo " Resume from: ${BROAD_ADAPTER_CKPT}"
echo " Data:        ${PROCESSED_SWM_N10}"
echo " LR:          3e-5 (original)"
echo " Epochs:      10"
echo " GPUs:        ${GPUS}"
echo "============================================"

mkdir -p "${CKPT_DIR}"
touch "${CKPT_DIR}/.train_marker_w200_wmissing_adapter_fix2"

python -m cxt.train \
    --model w200_wmissing \
    --adapter \
    --adapter-samples 10 \
    --resume-adapter "${BROAD_ADAPTER_CKPT}" \
    --dataset-path "${PROCESSED_SWM_N10}" \
    --gpus ${GPUS} \
    --epochs 10 \
    --lr 3e-5 \
    --workers "${TRAIN_WORKERS}" \
    --log-dir "${BASE_DIR}"

# Install the new checkpoint
LATEST=$(find "${CKPT_DIR}" -name "*.ckpt" -path "*/checkpoints/*" \
    -newer "${CKPT_DIR}/.train_marker_w200_wmissing_adapter_fix2" 2>/dev/null \
    | sort | tail -1 || true)

if [ -z "${LATEST}" ]; then
    echo "ERROR: no checkpoint found after training"
    exit 1
fi

DEST="${CKPT_CACHE}/w200_wmissing_adapter"
mkdir -p "${DEST}"

# Back up current checkpoint
if [ -f "${DEST}/w200_wmissing_adapter_epoch=9-step=480.ckpt" ]; then
    mv "${DEST}/w200_wmissing_adapter_epoch=9-step=480.ckpt" \
       "${DEST}/w200_wmissing_adapter_epoch=9-step=480.ckpt.broad_3e4_bak"
    echo "Backed up broad-3e4 checkpoint to .broad_3e4_bak"
fi

cp "${LATEST}" "${DEST}/w200_wmissing_adapter_epoch=9-step=480.ckpt"
echo "Installed: ${DEST}/w200_wmissing_adapter_epoch=9-step=480.ckpt"
echo "Done."
