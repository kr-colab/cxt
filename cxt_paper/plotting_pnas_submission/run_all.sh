#!/usr/bin/env bash
# Run all PNAS figure scripts sequentially.
# Usage: bash run_all.sh [--output-dir OUTPUT_DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="${1:-output}"

echo "============================================"
echo "  PNAS Figure Generation"
echo "  Output directory: $OUTPUT_DIR"
echo "============================================"

for script in \
    fig1_model_schematic.py \
    fig2_benchmark.py \
    fig3_stdpopsim_v2.py \
    fig4_stdpopsim_v3.py \
    fig5_demography.py \
    fig6_human_1kg.py \
    fig7_mosquito.py \
    fig8_inversion.py
do
    echo ""
    echo "--- Running $script ---"
    python "$script" --output-dir "$OUTPUT_DIR" || {
        echo "WARNING: $script failed (exit code $?), continuing..."
    }
done

echo ""
echo "============================================"
echo "  All figures complete."
echo "  Output: $OUTPUT_DIR/"
echo "============================================"
