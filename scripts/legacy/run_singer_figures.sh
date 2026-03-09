#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/sietch_colab/data_share/cxt_scratch}"

source "${BASE_DIR}/.venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/home/kkor/cxt"

FIG_OUT_MAIN="${BASE_DIR}/figures/output/main"
FIG_OUT_SUPP="${BASE_DIR}/figures/output/supplementary"
FIG_CACHE="${BASE_DIR}/figures/output/main/cache"

echo "[$(date +%T)] Running Singer+Polegon figures ..."

python -m figures.singer.plot_singer \
    --output-dir-main "${FIG_OUT_MAIN}" \
    --output-dir-supp "${FIG_OUT_SUPP}" \
    --cache-dir "${FIG_CACHE}/singer" \
    --fig4-cache-dir "${FIG_CACHE}/fig4" \
    --fig5-cache-dir "${FIG_CACHE}/fig5"

echo "[$(date +%T)] Singer figures done."
