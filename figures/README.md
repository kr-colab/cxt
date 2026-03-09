# Paper Figures

Reproduction scripts for all figures in
*Coalescence and Translation: A Language Model for Population Genetics*
(Korfmann et al., bioRxiv 2025.06.24.661337v2).

## Directory structure

```
figures/
├── paths.py                    Data path definitions (configurable via env vars)
├── utils.py                    Shared plotting, simulation, and SMC++ utilities
├── run_all_figures.sh          Run all figures (quick, uses existing caches)
├── main/                       Main manuscript figures (1–8)
│   ├── fig1_model_schematic.py
│   ├── fig2_benchmark_comparison.py
│   ├── fig3_stdpopsim_v2_coalescence.py
│   ├── fig4_stdpopsim_v3_ood.py
│   ├── fig5_demography_inference.py
│   ├── fig6_human_1kg.py
│   ├── fig7_mosquito_rdl.py
│   └── fig8_inversion_coalescence.py
├── supplementary/              Supplementary figures
│   ├── figS4_sample_size_adapter.py
│   ├── figS5_window_resolution.py
│   ├── figS6_runtime_benchmark.py
│   ├── figS9_mosquito_comparison.py
│   ├── figS10_cross_coalescence.py
│   └── figS11_interpolation_grid.py
└── output/                     Generated figure files
    ├── main/
    └── supplementary/
```

## Usage

### Run all figures

From the repo root:

```bash
./figures/run_all_figures.sh
```

### Fresh end-to-end run

To generate figures from freshly trained models in an isolated directory:

```bash
./scripts/run_fresh.sh figures
```

This uses `CXT_CHECKPOINT_CACHE` to load checkpoints from the fresh run
directory instead of the global cache.

### Run individual scripts

```bash
python -m figures.main.fig1_model_schematic --output-dir figures/output/main
```

Most scripts accept `--output-dir`, `--cache-dir`, and `--devices`
arguments. Cache directories store intermediate results (simulations,
TMRCA predictions) so that expensive GPU computations are not repeated.

Data paths are resolved from `figures/paths.py` (defaults match the
shared data paths). Override via env: `AG1000G_DATA_DIR`,
`AG1000G_ACCESSIBILITY`, `HG1KG_TSZ_DIR`, `BENCHMARK_JSONL`.

## Dependencies

- `cxt` (this package, uses `cxt.load_model`, `cxt.translate`, `cxt.correction`, `cxt.utils`)
- `msprime`, `tskit`, `stdpopsim`
- `numpy`, `matplotlib`, `scipy`
- `torch`, `pytorch-lightning`
- For SMC++ comparisons: `singularity` or `apptainer` with `docker://terhorst/smcpp:latest`

## Checkpoint loading

Figure scripts call `cxt.load_model()` which by default downloads
pretrained checkpoints from GitHub to `~/.cache/cxt/checkpoints/`.

To use locally trained checkpoints instead, set:

```bash
export CXT_CHECKPOINT_CACHE=/path/to/your/checkpoints
```

The `scripts/run_fresh.sh` script sets this automatically.

## Note on calibration experiments

Posterior calibration (Figure S8) and model calibration (Figures S2/S3) experiments
live in `experiments/` and are not duplicated here. They require `ray` for parallel
execution and produce cache files consumed by their respective `make_supp_figure.py`
/ `make_supp_figs.py` plotting scripts.
