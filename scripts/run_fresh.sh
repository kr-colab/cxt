#!/usr/bin/env bash
# =============================================================================
# run_fresh.sh  —  Complete fresh reproduction of the cxt paper
#
# Bootstraps an isolated uv virtualenv, then runs simulate → preprocess →
# train → figures.  All outputs go to BASE_DIR.  Does NOT touch system
# Python or any existing data.
#
# Usage:
#   ./scripts/run_fresh.sh                    # run everything
#   ./scripts/run_fresh.sh simulate           # only simulations
#   ./scripts/run_fresh.sh preprocess         # only preprocessing
#   ./scripts/run_fresh.sh train              # only training
#   ./scripts/run_fresh.sh figures            # only figures
#   ./scripts/run_fresh.sh train figures      # multiple stages
#
# All output goes to BASE_DIR (default: /home/kkor/cxt_fresh).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — everything under one new directory
# ---------------------------------------------------------------------------
BASE_DIR="${BASE_DIR:-/sietch_colab/data_share/cxt_scratch}"

DATA_DIR="${BASE_DIR}/data"
CKPT_DIR="${BASE_DIR}/lightning_logs"
CKPT_CACHE="${BASE_DIR}/checkpoints"
FIG_OUT_MAIN="${BASE_DIR}/figures/output/main"
FIG_OUT_SUPP="${BASE_DIR}/figures/output/supplementary"
FIG_CACHE_MAIN="${BASE_DIR}/figures/output/main/cache"
FIG_CACHE_SUPP="${BASE_DIR}/figures/output/supplementary/cache"

# ---------------------------------------------------------------------------
# Hardware — GPU 1,2 and 80 CPUs
# ---------------------------------------------------------------------------
GPUS="0 1"
SIM_WORKERS=80
PREPROCESS_WORKERS=80
TRAIN_WORKERS=16

# ---------------------------------------------------------------------------
# External data for figures — defaults to ~/cxt_paper_archive/
# Override CXT_PAPER_ARCHIVE or individual vars to relocate.
# ---------------------------------------------------------------------------
_ARCHIVE="${CXT_PAPER_ARCHIVE:-${HOME}/cxt_paper_archive}"
BITMASK="${BITMASK:-${_ARCHIVE}/ag1000g/agp3.is_accessible.txt.npz}"
export AG1000G_DATA_DIR="${AG1000G_DATA_DIR:-${_ARCHIVE}/ag1000g}"
export AG1000G_ACCESSIBILITY="${AG1000G_ACCESSIBILITY:-${BITMASK}}"
export HG1KG_TSZ_DIR="${HG1KG_TSZ_DIR:-${_ARCHIVE}/hg1kg}"

# Direct figure checkpoint loading to the new cache
export CXT_CHECKPOINT_CACHE="${CKPT_CACHE}"
export CUDA_VISIBLE_DEVICES="${GPUS// /,}"

# ---------------------------------------------------------------------------
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
VENV_DIR="${BASE_DIR}/.venv"

# ---------------------------------------------------------------------------
# Bootstrap: uv virtualenv
# ---------------------------------------------------------------------------
mkdir -p "${BASE_DIR}"
if ! command -v uv &>/dev/null; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtualenv at ${VENV_DIR} ..."
    uv venv "${VENV_DIR}" --python 3.12
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Installing cxt and dependencies ..."
uv pip install --link-mode copy -e "${REPO_ROOT}" \
    msprime tskit stdpopsim \
    torch torchtune torchao lightning einops tqdm \
    numpy scipy pandas \
    matplotlib seaborn \
    tszip requests \
    pytest
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
mkdir -p "${BASE_DIR}"
LOGFILE="${BASE_DIR}/run_fresh_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

FAILED=()
STAGE_TIMES=()
SCRIPT_START=$SECONDS

log()  { echo "[$(date +%H:%M:%S)] $*"; }
pass() { log "  ✓ $1 (${2}s)"; }
fail() { log "  ✗ $1 (exit $2, ${3}s)"; FAILED+=("$1"); }

run_step() {
    local name="$1"; shift
    log "--- $name ---"
    local t0=$SECONDS
    if "$@" 2>&1; then
        pass "$name" "$(( SECONDS - t0 ))"
    else
        fail "$name" "$?" "$(( SECONDS - t0 ))"
    fi
}

log "============================================"
log " Fresh reproduction run (uv virtualenv)"
log " $(date)"
log " BASE_DIR:  ${BASE_DIR}"
log " VENV:      ${VENV_DIR}"
log " GPUs:      ${GPUS}"
log " CPU workers: ${SIM_WORKERS}"
log " Log file:  ${LOGFILE}"
log "============================================"

# ---------------------------------------------------------------------------
# Parse stages
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then
    STAGES=(simulate preprocess train figures)
else
    STAGES=("$@")
fi
log "Stages: ${STAGES[*]}"
log ""

# ===================================================================
#  STAGE 1 — SIMULATE
#  Generates tree sequences (.trees) + X/y arrays for all scenarios.
# ===================================================================
do_simulate() {
    local t0=$SECONDS
    log "============================================"
    log "STAGE 1: SIMULATE"
    log "============================================"

    mkdir -p "${DATA_DIR}"

    local SIM_SCRIPT="cxt/simulation_ts_only.py"

    # -- Base dataset --
    run_step "Simulate: constant" \
        python "${SIM_SCRIPT}" \
            --num_processes "${SIM_WORKERS}" --num_samples 10000 \
            --data_dir "${DATA_DIR}/base_dataset" --scenario constant

    run_step "Simulate: sawtooth" \
        python "${SIM_SCRIPT}" \
            --num_processes "${SIM_WORKERS}" --num_samples 1000 \
            --data_dir "${DATA_DIR}/ssd" --scenario sawtooth

    run_step "Simulate: island" \
        python "${SIM_SCRIPT}" \
            --num_processes "${SIM_WORKERS}" --num_samples 1000 \
            --data_dir "${DATA_DIR}/idd" --scenario island

    # -- LLM parameter sweeps --
    run_step "Simulate: llm_ne_sawtooth" \
        python "${SIM_SCRIPT}" \
            --num_processes "${SIM_WORKERS}" --num_samples 125 \
            --data_dir "${DATA_DIR}/llm" --scenario llm_ne_sawtooth

    run_step "Simulate: llm_hard_sweeps" \
        python "${SIM_SCRIPT}" \
            --num_processes "${SIM_WORKERS}" --num_samples 50 \
            --data_dir "${DATA_DIR}/llm" --scenario llm_hard_sweeps

    run_step "Simulate: llm_island_3pop" \
        python "${SIM_SCRIPT}" \
            --num_processes "${SIM_WORKERS}" --num_samples 50 \
            --data_dir "${DATA_DIR}/llm" --scenario llm_island_3pop

    run_step "Simulate: llm_ne_constant" \
        python "${SIM_SCRIPT}" \
            --num_processes "${SIM_WORKERS}" --num_samples 500 \
            --data_dir "${DATA_DIR}/llm" --scenario llm_ne_constant

    # -- stdpopsim mammals --
    for scenario in \
        stdpopsim_homsap stdpopsim_homsap_map \
        stdpopsim_bostau \
        stdpopsim_canfam stdpopsim_canfam_map \
        stdpopsim_pantro \
        stdpopsim_papanu stdpopsim_papanu_map \
        stdpopsim_ponabe stdpopsim_ponabe_map; do

        local nsamples=1000
        run_step "Simulate: ${scenario}" \
            python "${SIM_SCRIPT}" \
                --num_processes "${SIM_WORKERS}" --num_samples "${nsamples}" \
                --data_dir "${DATA_DIR}/stdpopsim/v0.2/${scenario}" \
                --scenario "${scenario}"
    done

    # -- stdpopsim other species (varying sample counts) --
    declare -A other_species=(
        [stdpopsim_aedaeg]=300
        [stdpopsim_anapla]=25
        [stdpopsim_anocar]=5
        [stdpopsim_anogam]=100
        [stdpopsim_aratha]=500
        [stdpopsim_aratha_map]=500
        [stdpopsim_caeele]=1000
        [stdpopsim_caeele_map]=1000
        [stdpopsim_dromel]=5
        [stdpopsim_drosec]=300
        [stdpopsim_gasacu]=1000
        [stdpopsim_helann]=300
        [stdpopsim_helmel]=5
        [stdpopsim_apimel]=5
        [stdpopsim_musmus]=1000
    )
    for scenario in "${!other_species[@]}"; do
        run_step "Simulate: ${scenario}" \
            python "${SIM_SCRIPT}" \
                --num_processes "${SIM_WORKERS}" --num_samples "${other_species[$scenario]}" \
                --data_dir "${DATA_DIR}/stdpopsim/v0.2/${scenario}" \
                --scenario "${scenario}"
    done

    STAGE_TIMES+=("Simulate: $(( SECONDS - t0 ))s")
}

# ===================================================================
#  STAGE 2 — PREPROCESS
#  Converts tree sequences into PairDataset layout (train/test splits,
#  multiple window sizes, sample counts, and bitmask variants).
# ===================================================================
do_preprocess() {
    local t0=$SECONDS
    log "============================================"
    log "STAGE 2: PREPROCESS"
    log "============================================"

    # Build ts_large_pop symlink tree: only high-Ne stdpopsim species
    # (used by w200 preprocessing — low-Ne species lack variation at 200bp)
    local LARGE_POP_DIR="${DATA_DIR}/ts_large_pop"
    mkdir -p "${LARGE_POP_DIR}/stdpopsim/v0.2"
    local large_pop_species=(
        stdpopsim_aedaeg stdpopsim_anapla stdpopsim_anocar stdpopsim_anogam
        stdpopsim_caeele stdpopsim_caeele_map stdpopsim_dromel stdpopsim_drosec
        stdpopsim_gasacu stdpopsim_helann stdpopsim_helmel
        stdpopsim_papanu stdpopsim_papanu_map
    )
    for sp in "${large_pop_species[@]}"; do
        local src="${DATA_DIR}/stdpopsim/v0.2/${sp}"
        local dst="${LARGE_POP_DIR}/stdpopsim/v0.2/${sp}"
        if [ -d "${src}" ] && [ ! -e "${dst}" ]; then
            ln -s "${src}" "${dst}"
        fi
    done
    log "ts_large_pop: ${#large_pop_species[@]} species symlinked"

    # 0. processed_narrow (w2000, 50 samples, 200 pairs) — constant scenario only, used by narrow
    run_step "Preprocess: processed_narrow" \
        python -m cxt.preprocess \
            --base_dir "${DATA_DIR}/base_dataset" \
            --out_subdir processed_narrow \
            --window_size 2000 \
            --num_pairs 200 \
            --train_ratio 0.9 \
            --global_seed 12345 \
            --num_workers "${PREPROCESS_WORKERS}" \
            --skip_existing

    # 1. processed (w2000, 50 samples, 200 pairs) — used by broad
    run_step "Preprocess: processed" \
        python -m cxt.preprocess \
            --base_dir "${DATA_DIR}" \
            --out_subdir processed \
            --window_size 2000 \
            --num_pairs 200 \
            --train_ratio 0.9 \
            --global_seed 12345 \
            --num_workers "${PREPROCESS_WORKERS}" \
            --skip_existing

    # 2. processed_n10 (w2000, 10 samples, 20 pairs) — used by broad+adapter
    run_step "Preprocess: processed_n10" \
        python -m cxt.preprocess \
            --base_dir "${DATA_DIR}" \
            --out_subdir processed_n10 \
            --window_size 2000 \
            --num_pairs 20 \
            --simplify_first_n_samples 10 \
            --train_ratio 0.9 \
            --global_seed 12345 \
            --num_workers "${PREPROCESS_WORKERS}"

    # 3. processed_small_window (w200, 50 samples, 200 pairs) — used by broad_w200
    #    Uses ts_large_pop (high-Ne species only)
    run_step "Preprocess: processed_small_window" \
        python -m cxt.preprocess \
            --base_dir "${LARGE_POP_DIR}" \
            --out_subdir processed_small_window \
            --window_size 200 \
            --sequence_length 100000 \
            --num_pairs 200 \
            --train_ratio 0.9 \
            --global_seed 12345 \
            --num_workers "${PREPROCESS_WORKERS}" \
            --skip_existing

    # 4. processed_small_window_missing_data (w200, 50 samples, bitmask) — used by w200_wmissing
    #    Uses ts_large_pop (high-Ne species only)
    run_step "Preprocess: processed_small_window_missing_data" \
        python -m cxt.preprocess \
            --base_dir "${LARGE_POP_DIR}" \
            --out_subdir processed_small_window_missing_data \
            --window_size 200 \
            --sequence_length 100000 \
            --num_pairs 200 \
            --train_ratio 0.9 \
            --global_seed 12345 \
            --num_workers "${PREPROCESS_WORKERS}" \
            --skip_existing \
            --bitmask "${BITMASK}"

    # 5. processed_small_window_missing_data_n10 (w200, 10 samples, bitmask) — used by w200_wmissing_adapter
    #    Uses ts_large_pop (high-Ne species only)
    run_step "Preprocess: processed_small_window_missing_data_n10" \
        python -m cxt.preprocess \
            --base_dir "${LARGE_POP_DIR}" \
            --out_subdir processed_small_window_missing_data_n10 \
            --window_size 200 \
            --sequence_length 100000 \
            --num_pairs 20 \
            --simplify_first_n_samples 10 \
            --train_ratio 0.9 \
            --global_seed 12345 \
            --num_workers "${PREPROCESS_WORKERS}" \
            --skip_existing \
            --bitmask "${BITMASK}"

    STAGE_TIMES+=("Preprocess: $(( SECONDS - t0 ))s")
}

# ===================================================================
#  STAGE 3 — TRAIN
#  Trains all 7 model variants in dependency order.
#  Checkpoints are installed into CXT_CHECKPOINT_CACHE after each run
#  so that cxt.load_model() finds them during figure generation.
# ===================================================================

install_ckpt() {
    # install_ckpt <model_type> <registry_filename>
    # Finds the latest checkpoint in CKPT_DIR and copies it to the cache
    local model_type="$1"
    local registry_name="$2"
    local latest
    latest=$(find "${CKPT_DIR}" -name "*.ckpt" -path "*/checkpoints/*" -newer "${CKPT_DIR}/.train_marker_${model_type}" 2>/dev/null | sort | tail -1 || true)
    if [ -z "$latest" ]; then
        log "  WARNING: no checkpoint found for ${model_type}"
        return 1
    fi
    local dest="${CKPT_CACHE}/${model_type}"
    mkdir -p "${dest}"
    cp "${latest}" "${dest}/${registry_name}"
    log "  Installed: ${dest}/${registry_name}"
}

do_train() {
    local t0=$SECONDS
    log "============================================"
    log "STAGE 3: TRAIN"
    log "============================================"

    PROCESSED="${DATA_DIR}/processed"
    PROCESSED_NARROW="${DATA_DIR}/base_dataset/processed_narrow"
    PROCESSED_N10="${DATA_DIR}/processed_n10"
    PROCESSED_SW="${DATA_DIR}/ts_large_pop/processed_small_window"
    PROCESSED_SWM="${DATA_DIR}/ts_large_pop/processed_small_window_missing_data"
    PROCESSED_SWM_N10="${DATA_DIR}/ts_large_pop/processed_small_window_missing_data_n10"

    mkdir -p "${CKPT_DIR}"

    # ---- From-scratch models ----

    # 1. narrow (6 layers, 6 epochs) — constant scenario only
    if [ -f "${CKPT_CACHE}/narrow/narrow_epoch=5-step=4692.ckpt" ]; then
        log "  Skipping narrow (checkpoint already exists)"
    else
        touch "${CKPT_DIR}/.train_marker_narrow"
        run_step "Train: narrow" \
            python -m cxt.train \
                --model narrow \
                --dataset-path "${PROCESSED_NARROW}" \
                --gpus ${GPUS} \
                --epochs 6 \
                --workers "${TRAIN_WORKERS}" \
                --log-dir "${BASE_DIR}"
        install_ckpt "narrow" "narrow_epoch=5-step=4692.ckpt"
    fi

    # 2. broad (10 layers, 2 epochs) — main backbone
    BROAD_CKPT="${CKPT_CACHE}/broad/broad_epoch=1-step=5280.ckpt"
    if [ -f "${BROAD_CKPT}" ]; then
        log "  Skipping broad (checkpoint already exists)"
    else
        touch "${CKPT_DIR}/.train_marker_broad"
        run_step "Train: broad" \
            python -m cxt.train \
                --model broad \
                --dataset-path "${PROCESSED}" \
                --gpus ${GPUS} \
                --epochs 2 \
                --workers "${TRAIN_WORKERS}" \
                --log-dir "${BASE_DIR}"
        install_ckpt "broad" "broad_epoch=1-step=5280.ckpt"
    fi

    # ---- Fine-tuned from broad ----

    if [ -f "${BROAD_CKPT}" ]; then
        # 4. broad_w200
        touch "${CKPT_DIR}/.train_marker_broad_w200"
        run_step "Train: broad_w200" \
            python -m cxt.train \
                --model broad_w200 \
                --dataset-path "${PROCESSED_SW}" \
                --gpus ${GPUS} \
                --epochs 2 \
                --lr 3e-5 \
                --checkpoint "${BROAD_CKPT}" \
                --workers "${TRAIN_WORKERS}" \
                --log-dir "${BASE_DIR}"
        install_ckpt "broad_w200" "broad_w200_epoch=1-step=944.ckpt"
        BROAD_W200_CKPT="${CKPT_CACHE}/broad_w200/broad_w200_epoch=1-step=944.ckpt"

        # 5. broad+adapter
        touch "${CKPT_DIR}/.train_marker_broad+adapter"
        run_step "Train: broad+adapter" \
            python -m cxt.train \
                --model broad \
                --adapter \
                --adapter-samples 10 \
                --dataset-path "${PROCESSED_N10}" \
                --gpus ${GPUS} \
                --epochs 3 \
                --checkpoint "${BROAD_CKPT}" \
                --workers "${TRAIN_WORKERS}" \
                --log-dir "${BASE_DIR}"
        install_ckpt "broad+adapter" "broad_adapter_epoch=2-step=792.ckpt"
    else
        log "WARNING: broad checkpoint not found, skipping downstream models"
    fi

    # ---- Fine-tuned from broad_w200 ----

    if [ -f "${BROAD_W200_CKPT:-}" ]; then
        # 6. w200_wmissing
        touch "${CKPT_DIR}/.train_marker_w200_wmissing"
        run_step "Train: w200_wmissing" \
            python -m cxt.train \
                --model w200_wmissing \
                --dataset-path "${PROCESSED_SWM}" \
                --gpus ${GPUS} \
                --epochs 2 \
                --lr 3e-5 \
                --checkpoint "${BROAD_W200_CKPT}" \
                --workers "${TRAIN_WORKERS}" \
                --log-dir "${BASE_DIR}"
        install_ckpt "w200_wmissing" "w200_wmissing_epoch=1-step=944.ckpt"
        W200_CKPT="${CKPT_CACHE}/w200_wmissing/w200_wmissing_epoch=1-step=944.ckpt"
    fi

    # ---- w200_wmissing_adapter: resume from broad+adapter on w200 missingness data ----
    # Two-stage training: broad+adapter learned the 10→50 mapping on w2000 data,
    # now we resume on w200+bitmask data at low lr to adapt for missingness.

    BROAD_ADAPTER_CKPT="${CKPT_CACHE}/broad+adapter/broad_adapter_epoch=2-step=792.ckpt"
    if [ -f "${BROAD_ADAPTER_CKPT}" ]; then
        # 7. w200_wmissing_adapter
        touch "${CKPT_DIR}/.train_marker_w200_wmissing_adapter"
        run_step "Train: w200_wmissing_adapter" \
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
        install_ckpt "w200_wmissing_adapter" "w200_wmissing_adapter_epoch=9-step=480.ckpt"
    fi

    STAGE_TIMES+=("Train: $(( SECONDS - t0 ))s")
}

# ===================================================================
#  STAGE 4 — FIGURES
#  Generates all main and supplementary figures using the freshly
#  trained checkpoints (via CXT_CHECKPOINT_CACHE).
# ===================================================================
do_figures() {
    local t0=$SECONDS
    log "============================================"
    log "STAGE 4: FIGURES"
    log "============================================"

    mkdir -p "${FIG_OUT_MAIN}" "${FIG_OUT_SUPP}" "${FIG_CACHE_MAIN}" "${FIG_CACHE_SUPP}"

    # Set CUDA_VISIBLE_DEVICES so figure scripts using default device lists work
    export CUDA_VISIBLE_DEVICES="${GPUS// /,}"

    local dev_args="--devices cuda:0 cuda:1"

    # ---- Main figures ----

    run_step "Fig 1: Model schematic" \
        python -m figures.main.fig1_model_schematic \
            --output-dir "${FIG_OUT_MAIN}" --cache-dir "${FIG_CACHE_MAIN}/fig1" \
            ${dev_args}

    run_step "Fig 2: Benchmark comparison" \
        python -m figures.main.fig2_benchmark_comparison \
            --output-dir "${FIG_OUT_MAIN}" --cache-dir "${FIG_CACHE_MAIN}/fig2" \
            ${dev_args}

    run_step "Fig 3: stdpopsim v2 coalescence" \
        python -m figures.main.fig3_stdpopsim_v2_coalescence \
            --output-dir "${FIG_OUT_MAIN}" --cache-dir "${FIG_CACHE_MAIN}/fig3" \
            ${dev_args}

    run_step "Fig 4: stdpopsim v3 OOD" \
        python -m figures.main.fig4_stdpopsim_v3_ood \
            --output-dir "${FIG_OUT_MAIN}" --cache-dir "${FIG_CACHE_MAIN}/fig4" \
            ${dev_args}

    run_step "Fig 5: Demography inference" \
        python -m figures.main.fig5_demography_inference \
            --output-dir "${FIG_OUT_MAIN}" --cache-dir "${FIG_CACHE_MAIN}/fig5" \
            ${dev_args}

    run_step "Fig 6: Human 1000 Genomes" \
        python -m figures.main.fig6_human_1kg \
            --output-dir "${FIG_OUT_MAIN}" --cache-dir "${FIG_CACHE_MAIN}/fig6" \
            ${dev_args}

    # fig7 before fig8 and figS9 (they read fig7 caches)
    run_step "Fig 7: Mosquito RDL" \
        python -m figures.main.fig7_mosquito_rdl \
            --output-dir "${FIG_OUT_MAIN}" --cache-dir "${FIG_CACHE_MAIN}/fig7" \
            ${dev_args}

    run_step "Fig 8: Inversion coalescence" \
        python -m figures.main.fig8_inversion_coalescence \
            --output-dir "${FIG_OUT_MAIN}" --cache-dir "${FIG_CACHE_MAIN}/fig7"

    # ---- Supplementary figures ----

    run_step "Fig S4: Sample size adapter" \
        python -m figures.supplementary.figS4_sample_size_adapter \
            --output-dir "${FIG_OUT_SUPP}" --cache-dir "${FIG_CACHE_SUPP}/figS4" \
            ${dev_args}

    run_step "Fig S5: Window resolution" \
        python -m figures.supplementary.figS5_window_resolution \
            --output-dir "${FIG_OUT_SUPP}" --cache-dir "${FIG_CACHE_SUPP}/figS5" \
            ${dev_args}

    run_step "Fig S6: Runtime benchmark" \
        python -m figures.supplementary.figS6_runtime_benchmark \
            --output-dir "${FIG_OUT_SUPP}" --cache-dir "${FIG_CACHE_SUPP}/figS6"

    run_step "Fig S9: Mosquito comparison" \
        python -m figures.supplementary.figS9_mosquito_comparison \
            --output-dir "${FIG_OUT_SUPP}" --cache-dir "${FIG_CACHE_SUPP}/figS9"

    run_step "Fig S10: Cross coalescence" \
        python -m figures.supplementary.figS10_cross_coalescence \
            --output-dir "${FIG_OUT_SUPP}" --cache-dir "${FIG_CACHE_SUPP}/figS10" \
            ${dev_args}

    run_step "Fig S11: Interpolation grid" \
        python -m figures.supplementary.figS11_interpolation_grid \
            --output-dir "${FIG_OUT_SUPP}" --cache-dir "${FIG_CACHE_SUPP}/figS11" \
            ${dev_args}

    STAGE_TIMES+=("Figures: $(( SECONDS - t0 ))s")
}

# ===================================================================
#  Dispatch
# ===================================================================
for stage in "${STAGES[@]}"; do
    case "$stage" in
        simulate)   do_simulate   ;;
        preprocess) do_preprocess ;;
        train)      do_train      ;;
        figures)    do_figures    ;;
        *)
            log "Unknown stage: $stage"
            log "Valid stages: simulate, preprocess, train, figures"
            exit 1
            ;;
    esac
done

# ===================================================================
#  Summary
# ===================================================================
echo ""
log "============================================"
log "FRESH RUN — SUMMARY"
log "============================================"
log "Base directory: ${BASE_DIR}"
log "Total wall time: $(( SECONDS - SCRIPT_START ))s"
for t in "${STAGE_TIMES[@]}"; do
    log "  $t"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    log ""
    log "FAILED steps (${#FAILED[@]}):"
    for f in "${FAILED[@]}"; do
        log "  - $f"
    done
    log ""
    log "Log: $LOGFILE"
    exit 1
else
    log ""
    log "All steps completed successfully."
    log "Log: $LOGFILE"
fi
