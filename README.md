# cxt(kit): Coalescence x Translation (toolkit)

A transformer-based method for inferring pairwise coalescent times (TMRCA) from
genotype data using a language-modelling approach.

<p align="center">
    <img src="docs/source/figures/logo_3d_2.png" alt="Logo" width="200">
</p>

[Full documentation on ReadTheDocs](https://cxt.readthedocs.io)

## Overview

**cxt** treats TMRCA inference as a sequence-to-sequence translation task. For
each pair of haplotypes it computes a multi-scale site-frequency spectrum (SFS)
in sliding windows, feeds it through a token-free transformer decoder, and
outputs a discretized log-TMRCA profile across the genome.

Key features:

- **No tree-sequence inference** -- works directly on genotype matrices, VCF
  files, or `tskit` tree sequences.
- **Stochastic sampling** -- multiple replicate predictions give uncertainty
  estimates.
- **Bias correction** -- optional Bayesian diversity-based correction.
- **Multiple model variants** -- narrow, broad, broad\_w200, residual,
  w200\_wmissing, and adapter-based models for different sample sizes.
- **Multi-GPU inference** -- shards pairs across GPUs automatically.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+ and PyTorch 2.0+. GPU recommended for inference.

## Quick start

```python
import cxt

# Load a pretrained model (downloads checkpoint on first use)
model = cxt.load_model("broad", device="cuda")

# Translate a tree sequence
import msprime
ts = msprime.sim_ancestry(25, population_size=1e4, sequence_length=1e6,
                          recombination_rate=1e-8, random_seed=42)
ts = msprime.mutate(ts, rate=1e-8, random_seed=42)

tmrca, index_map = cxt.translate(
    ts, model,
    blocks=[(0, 1_000_000)],
    pivot_pairs=[(0, 1), (2, 3)],
    devices=["cuda:0"],
    n_reps=15,
)
# tmrca shape: (n_items, n_reps, n_windows)  -- log-TMRCA values
```

### From a VCF

```python
tmrca, index_map = cxt.translate(
    "path/to/file.vcf", model,
    blocks=[(0, 1_000_000)],
    pivot_pairs=[(0, 1)],
)
```

### From a genotype matrix

```python
import numpy as np
gm = np.load("genotypes.npy")     # (n_haplotypes, n_sites)
pos = np.load("positions.npy")    # (n_sites,) in bp

tmrca, index_map = cxt.translate(
    (gm, pos), model,
    blocks=[(0, 1_000_000)],
    pivot_pairs=[(0, 1)],
)
```

## Model variants

| Name | Preset | Layers | Description |
|------|--------|--------|-------------|
| `narrow` | `PRESETS["narrow"]` | 6 | Smaller model, faster inference |
| `broad` | `PRESETS["broad"]` | 10 | Main model, best accuracy |
| `broad_w200` | `PRESETS["broad_w200"]` | 10 | 200bp windows (fine-scale) |
| `residual` | `PRESETS["residual"]` | 10 | Predicts log-residuals from mean |
| `w200_wmissing` | `PRESETS["w200_wmissing"]` | 10 | 200bp windows, handles missingness |
| `broad+adapter` | adapter on `broad` | 10 | 10-sample adapter on broad backbone |
| `w200_wmissing_adapter` | adapter on `w200_wmissing` | 10 | 10-sample adapter with missingness |

## Training

All model variants are trained with a single unified script:

```bash
# Broad model from scratch
python -m cxt.train --model broad --dataset-path /path/to/data --gpus 0 1 2 --epochs 2

# Fine-tune broad_w200 from broad checkpoint
python -m cxt.train --model broad_w200 --checkpoint /path/to/broad.ckpt --lr 3e-5

# Adapter training (10-sample) on frozen broad backbone
python -m cxt.train --model broad --adapter --adapter-samples 10 \
    --checkpoint /path/to/broad.ckpt --dataset-path /path/to/n10_data

# w200 with missingness support
python -m cxt.train --model w200_wmissing --checkpoint /path/to/broad_w200.ckpt --lr 3e-5

# Direct checkpoints to a specific directory
python -m cxt.train --model broad --dataset-path /path/to/data --log-dir /path/to/output
```

## Data pipeline

```
simulate → preprocess → train → figures
```

1. **Simulate**: Generate tree sequences with `cxt/simulation_ts_only.py`

   ```bash
   python cxt/simulation_ts_only.py --scenario constant --data_dir /path/to/raw \
       --num_samples 10000 --num_processes 80
   ```

   Tree sequences (`.trees` files) are saved per scenario, enabling
   downstream preprocessing.

2. **Preprocess**: Convert tree sequences to training pairs with `cxt.preprocess`

   ```bash
   python -m cxt.preprocess --base_dir /path/to/raw --out_subdir processed \
       --window_size 2000 --num_pairs 200 --num_workers 80 --skip_existing
   ```

3. **Train**: Run the unified training script above

### Full reproduction

A single script bootstraps a `uv` virtualenv and runs the entire pipeline
(simulate → preprocess → train → figures) in an isolated directory:

```bash
./scripts/run_fresh.sh
```

All outputs go to `/sietch_colab/data_share/cxt_scratch/` by default.
Override with `BASE_DIR`:

```bash
BASE_DIR=/scratch/myuser/cxt_run ./scripts/run_fresh.sh
```

Run individual stages: `./scripts/run_fresh.sh simulate`, `preprocess`,
`train`, or `figures`. Multiple stages can be combined:
`./scripts/run_fresh.sh train figures`.

## Package structure

```
cxt/
├── __init__.py              # Public API: load_model, translate, PRESETS
├── config.py                # ModelConfig, PRESETS, AdapterConfig, TrainingConfig
├── model.py                 # TokenFreeDecoder (transformer architecture)
├── modules.py               # Attention, MLP, LayerNorm, MutationsToLatentSpace
├── checkpoint.py            # Model loading, checkpoint download/cache
├── translate.py             # Unified inference: translate(), generate(), multi-GPU
├── sfs.py                   # SFS computation, source building, filtering
├── correction.py            # Diversity-based bias correction (deterministic + stochastic)
├── train.py                 # Unified training script (Lightning)
├── dataset.py               # PairDataset for training
├── simulation_ts_only.py    # Tree-sequence simulation CLI (msprime + stdpopsim)
├── preprocess.py            # TS → training data conversion
└── utils.py                 # Grids, helpers, legacy simulation functions
```

## Configuration

Model configuration is handled through a single `ModelConfig` dataclass with
presets for each variant:

```python
from cxt.config import ModelConfig, PRESETS

# Get a preset
config = PRESETS["broad"]

# Customize for inference
config = config.for_inference(batch_size=1, device="cuda")

# Customize for training
config = config.for_training(batch_size=128, device="cuda")
```

Key config flags:
- `mask_singletons`: Whether to zero out singleton SFS entries (True for most
  models, False for `w200_wmissing`)
- `use_kv_cache`: Allocate KV cache for autoregressive decoding (set
  automatically by `for_inference()`)

## License

See [LICENSE](LICENSE) for details.
