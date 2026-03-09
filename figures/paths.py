"""Canonical data paths matching revision/ notebooks. Override via env vars if needed."""

import os

# Repo root (parent of figures/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ag1000G tree sequences and masks (revision/figure_mosquito/*.ipynb)
AG1000G_DATA_DIR = os.environ.get(
    "AG1000G_DATA_DIR",
    "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2",
)

# Ag1000G accessibility bitmask
AG1000G_ACCESSIBILITY = os.environ.get(
    "AG1000G_ACCESSIBILITY",
    "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz",
)

# Human 1000 Genomes tsz trees (revision/figure6/experiment.ipynb)
HG1KG_TSZ_DIR = os.environ.get(
    "HG1KG_TSZ_DIR",
    "/sietch_colab/data_share/hg1kg/tsinfer-trees/working",
)

# Benchmark JSONL from revision/figure_benchmark/experiment.ipynb
BENCHMARK_JSONL = os.environ.get(
    "BENCHMARK_JSONL",
    os.path.join(_REPO_ROOT, "revision/figure_benchmark/cxt_benchmark_runtime_blocks.jsonl"),
)

BENCHMARK_VALIDATION_JSONL = os.environ.get(
    "BENCHMARK_VALIDATION_JSONL",
    os.path.join(_REPO_ROOT, "revision/figure_benchmark/cxt_benchmark_runtime_blocks_regions_validation.jsonl"),
)

# SMC++ timing log from revision/figure_mosquito
SMCPP_TIMING_JSONL = os.environ.get(
    "SMCPP_TIMING_JSONL",
    os.path.join(_REPO_ROOT, "revision/figure_mosquito/tmp_smcpp_local_0/timing_log.jsonl"),
)

# SINGER benchmark timing data
SINGER_TIMING_PATH = os.environ.get(
    "SINGER_TIMING_PATH",
    "/sietch_colab/data_share/cxt/singer-benchmarks/mosquitos-runtime/singer_timing.txt",
)

