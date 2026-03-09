# Paper Figures

Reproduction scripts for all figures in
*Coalescence and Translation: A Language Model for Population Genetics*
(Korfmann et al., bioRxiv 2025.06.24.661337v2).

## Directory structure

```
figures/
├── utils.py                    Shared plotting, simulation, and SMC++ utilities
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

Data paths are resolved from `figures/paths.py` (defaults match revision notebooks).
Override via env: `AG1000G_DATA_DIR`, `HG1KG_TSZ_DIR`, `BENCHMARK_JSONL`

### Run individual scripts

```bash
python -m figures.main.fig1_model_schematic --output-dir figures/output/main
```

Most scripts accept `--output-dir` and `--cache-dir` arguments. Cache directories
store intermediate simulation results (e.g. tree sequences, SMC++ outputs) so that
expensive computations need not be repeated.

## Dependencies

- `cxt` (this package)
- `msprime`, `tskit`, `stdpopsim`
- `numpy`, `matplotlib`, `scipy`
- `torch`, `pytorch-lightning`
- For SMC++ comparisons: `singularity` or `apptainer` with `docker://terhorst/smcpp:latest`

## Note on calibration experiments

Posterior calibration (Figure S8) and model calibration (Figures S2/S3) experiments
live in `experiments/` and are not duplicated here. They require `ray` for parallel
execution and produce cache files consumed by their respective `make_supp_figure.py`
/ `make_supp_figs.py` plotting scripts.
