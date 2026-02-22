#!/usr/bin/env bash
# Run all figure scripts and produce outputs.
# Execute from repo root: ./figures/run_all_figures.sh
#
# Data paths are resolved from figures/paths.py (revision defaults).
# Override via env: AG1000G_DATA_DIR, HG1KG_TSZ_DIR, BENCHMARK_JSONL

cd "$(dirname "$0")/.."

OUT_MAIN="figures/output/main"
OUT_SUPP="figures/output/supplementary"
CACHE_MAIN="figures/output/main/cache"
CACHE_SUPP="figures/output/supplementary/cache"

mkdir -p "$OUT_MAIN" "$OUT_SUPP" "$CACHE_MAIN" "$CACHE_SUPP"

run() {
  local name="$1"
  shift
  echo "=========================================="
  echo "Running: $name"
  echo "=========================================="
  if python -m "$@" 2>&1; then
    echo "[OK] $name"
  else
    echo "[FAIL] $name (exit $?)"
  fi
}

echo "Main figures (1-8)"
run "Fig 1: Model schematic"             figures.main.fig1_model_schematic --output-dir "$OUT_MAIN" --cache-dir "$CACHE_MAIN/fig1"
run "Fig 2: Benchmark comparison"        figures.main.fig2_benchmark_comparison --output-dir "$OUT_MAIN" --cache-dir "$CACHE_MAIN/fig2"
run "Fig 3: stdpopsim v2 coalescence"    figures.main.fig3_stdpopsim_v2_coalescence --output-dir "$OUT_MAIN" --cache-dir "$CACHE_MAIN/fig3"
run "Fig 4: stdpopsim v3 OOD"            figures.main.fig4_stdpopsim_v3_ood --output-dir "$OUT_MAIN" --cache-dir "$CACHE_MAIN/fig4"
run "Fig 5: Demography inference"        figures.main.fig5_demography_inference --output-dir "$OUT_MAIN" --cache-dir "$CACHE_MAIN/fig5"
run "Fig 6: Human 1000 Genomes"          figures.main.fig6_human_1kg --output-dir "$OUT_MAIN" --cache-dir "$CACHE_MAIN/fig6"
run "Fig 7: Mosquito Rdl"                figures.main.fig7_mosquito_rdl --output-dir "$OUT_MAIN" --cache-dir "$CACHE_MAIN/fig7"
run "Fig 8: Inversion coalescence"       figures.main.fig8_inversion_coalescence --output-dir "$OUT_MAIN" --cache-dir "$CACHE_MAIN/fig7"

echo ""
echo "Supplementary figures"
run "Fig S4: Sample size adapter"        figures.supplementary.figS4_sample_size_adapter --output-dir "$OUT_SUPP" --cache-dir "$CACHE_SUPP/figS4"
run "Fig S5: Window resolution"         figures.supplementary.figS5_window_resolution --output-dir "$OUT_SUPP" --cache-dir "$CACHE_SUPP/figS5"
run "Fig S6: Runtime benchmark"         figures.supplementary.figS6_runtime_benchmark --output-dir "$OUT_SUPP" --cache-dir "$CACHE_SUPP/figS6"
run "Fig S9: Mosquito comparison"       figures.supplementary.figS9_mosquito_comparison --output-dir "$OUT_SUPP" --cache-dir "$CACHE_SUPP/figS9"
run "Fig S10: Cross coalescence"        figures.supplementary.figS10_cross_coalescence --output-dir "$OUT_SUPP" --cache-dir "$CACHE_SUPP/figS10"
run "Fig S11: Interpolation grid"        figures.supplementary.figS11_interpolation_grid --output-dir "$OUT_SUPP" --cache-dir "$CACHE_SUPP/figS11"

echo ""
echo "Done. Outputs in $OUT_MAIN and $OUT_SUPP"
