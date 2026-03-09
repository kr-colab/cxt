"""Canonical data paths matching revision/ notebooks. Override via env vars if needed."""

import os

# Repo root (parent of figures/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ag1000G tree sequences and masks (revision/figure_mosquito/*.ipynb)
AG1000G_DATA_DIR = os.environ.get(
    "AG1000G_DATA_DIR",
    "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees",
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
