# Runtime Benchmarks for Fig S6

Scripts to reproduce the CXT, SMC++, and SINGER runtime data used in Figure S6.

## Outputs

| Script | Output | Consumed by |
|--------|--------|-------------|
| `benchmark_cxt_runtime.py` | `cxt_benchmark_runtime_blocks_regions_validation.jsonl` | figS6 |
| `benchmark_smcpp_runtime.py` | `timing_log.jsonl` | figS6 |
| `benchmark_singer_runtime.py` | `singer_timing.txt` | figS6 |

## Requirements

- **CXT**: `cxt`, GPU(s), tree sequences at `--data-dir`
- **SMC++**: `singularity` or `apptainer`, `bgzip`, `tabix`, SMC++ SIF image
- **SINGER**: Implement `run_singer_on_tree_sequence()` in the script (template only)

## Data

Pre-simulated tree sequences (50 diploids = 100 haploids per region) at:
```
/sietch_colab/data_share/cxt/mosquito/benchmarks/
  benchmark_samples10_region0.trees
  benchmark_samples10_region1.trees
  ...
  benchmark_samples50_region0.trees
  ...
```

## Usage

```bash
# CXT
python -m scripts.benchmarks.benchmark_cxt_runtime --results-path revision/figure_benchmark/cxt_benchmark_runtime_blocks_regions_validation.jsonl

# SMC++
python -m scripts.benchmarks.benchmark_smcpp_runtime --output-dir revision/figure_mosquito/tmp_smcpp_local_0

# SINGER (template – implement run_singer_on_tree_sequence first)
python -m scripts.benchmarks.benchmark_singer_runtime --output /path/to/singer_timing.txt
```

## Plot Fig S6

After all three benchmarks have produced data:

```bash
python -m figures.supplementary.figS6_runtime_benchmark
```
