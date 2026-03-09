#!/bin/bash
# =============================================================================
# copy_checkpoints.sh — Copy trained checkpoints into the repo for LFS commit
#
# Copies checkpoints from a run_fresh.sh output directory (or any directory
# matching the same layout) into the repo's checkpoints/ folder, ready for
# `git add checkpoints/` followed by `git commit`.
#
# Usage:
#   ./scripts/copy_checkpoints.sh                          # default source
#   ./scripts/copy_checkpoints.sh /path/to/run_fresh/dir   # custom source
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_BASE="${1:-/sietch_colab/data_share/cxt_scratch/checkpoints}"
DEST_BASE="${REPO_ROOT}/checkpoints"

MODELS=(
    "broad/broad_epoch=1-step=5280.ckpt"
    "broad+adapter/broad_adapter_epoch=2-step=792.ckpt"
    "narrow/narrow_epoch=5-step=4692.ckpt"
    "broad_w200/broad_w200_epoch=1-step=944.ckpt"
    "w200_wmissing/w200_wmissing_epoch=1-step=944.ckpt"
    "w200_wmissing_adapter/w200_wmissing_adapter_epoch=9-step=480.ckpt"
)

echo "Source:      ${SRC_BASE}"
echo "Destination: ${DEST_BASE}"
echo ""

for entry in "${MODELS[@]}"; do
    model_dir="${entry%%/*}"
    filename="${entry##*/}"
    src="${SRC_BASE}/${entry}"
    dest="${DEST_BASE}/${model_dir}/${filename}"

    mkdir -p "${DEST_BASE}/${model_dir}"

    if [ ! -f "${src}" ]; then
        echo "WARNING: ${src} not found, skipping"
        continue
    fi

    echo "Copying ${model_dir}/${filename} ..."
    cp "${src}" "${dest}"
done

echo ""
echo "Done. Checkpoint layout:"
find "${DEST_BASE}" -name "*.ckpt" -printf "  %p\n" | sort
echo ""
echo "Next steps:"
echo "  git add checkpoints/"
echo "  git commit -m 'Update checkpoints from run_fresh.sh'"
