"""Canonical data paths for paper figures. Override via env vars if needed.

All external data consolidated in ~/cxt_paper_archive/ for long-term
reproducibility. Set CXT_PAPER_ARCHIVE to relocate the archive.
"""

import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_ARCHIVE = os.environ.get(
    "CXT_PAPER_ARCHIVE",
    os.path.join(os.path.expanduser("~"), "cxt_paper_archive"),
)

AG1000G_DATA_DIR = os.environ.get(
    "AG1000G_DATA_DIR",
    os.path.join(_ARCHIVE, "ag1000g"),
)

AG1000G_ACCESSIBILITY = os.environ.get(
    "AG1000G_ACCESSIBILITY",
    os.path.join(_ARCHIVE, "ag1000g", "agp3.is_accessible.txt.npz"),
)

HG1KG_TSZ_DIR = os.environ.get(
    "HG1KG_TSZ_DIR",
    os.path.join(_ARCHIVE, "hg1kg"),
)

BENCHMARK_JSONL = os.environ.get(
    "BENCHMARK_JSONL",
    os.path.join(_ARCHIVE, "benchmarks", "cxt_benchmark_runtime_blocks.jsonl"),
)

BENCHMARK_VALIDATION_JSONL = os.environ.get(
    "BENCHMARK_VALIDATION_JSONL",
    os.path.join(_ARCHIVE, "benchmarks", "cxt_benchmark_runtime_blocks_regions_validation.jsonl"),
)

SMCPP_TIMING_JSONL = os.environ.get(
    "SMCPP_TIMING_JSONL",
    os.path.join(_ARCHIVE, "benchmarks", "smcpp_timing_log.jsonl"),
)

SINGER_TIMING_PATH = os.environ.get(
    "SINGER_TIMING_PATH",
    os.path.join(_ARCHIVE, "singer-benchmarks", "mosquitos-runtime", "singer_timing.txt"),
)

SINGER_BASE = os.environ.get(
    "SINGER_BASE",
    os.path.join(_ARCHIVE, "singer-benchmarks"),
)

REVISION_MOSQUITO_CACHE = os.environ.get(
    "REVISION_MOSQUITO_CACHE",
    os.path.join(_ARCHIVE, "revision_mosquito_cache"),
)

