#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/sietch_colab/data_share/cxt_scratch}"
DATA_DIR="${BASE_DIR}/data"

source "${BASE_DIR}/.venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/home/kkor/cxt"

echo "Preprocessing constant-only dataset into processed_narrow ..."
python -m cxt.preprocess \
    --base_dir "${DATA_DIR}/base_dataset" \
    --out_subdir processed_narrow \
    --window_size 2000 \
    --num_pairs 200 \
    --train_ratio 0.9 \
    --global_seed 12345 \
    --num_workers 80

echo "Done. Output: ${DATA_DIR}/base_dataset/processed_narrow"
