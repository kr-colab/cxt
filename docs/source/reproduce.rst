Full Reproducibility
====================

``scripts/run_fresh.sh`` bootstraps a ``uv`` virtualenv, then simulates
new data, trains all models from scratch, and generates all figures in a
single isolated directory. It installs ``uv`` automatically if missing.


Quick start (fresh run)
-----------------------

Run everything (simulations → preprocessing → training → figures) in a
single isolated directory:

.. code-block:: bash

   ./scripts/run_fresh.sh

By default all outputs go to ``/sietch_colab/data_share/cxt_scratch/``.
Override with the ``BASE_DIR`` environment variable:

.. code-block:: bash

   BASE_DIR=/scratch/myuser/cxt_run ./scripts/run_fresh.sh

The script uses GPUs 1 and 2 and 80 CPU workers by default. These are
configured at the top of the script. A ``uv`` virtualenv is created at
``BASE_DIR/.venv`` and reused on subsequent runs. To recreate it:

.. code-block:: bash

   rm -rf /sietch_colab/data_share/cxt_scratch/.venv
   ./scripts/run_fresh.sh

Run individual stages:

.. code-block:: bash

   ./scripts/run_fresh.sh simulate       # only simulations
   ./scripts/run_fresh.sh preprocess      # only preprocessing
   ./scripts/run_fresh.sh train           # only training
   ./scripts/run_fresh.sh figures         # only figures
   ./scripts/run_fresh.sh train figures   # multiple stages


Pipeline overview
-----------------

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────────────┐
   │                                                                      │
   │  STAGE 1: SIMULATE        python -m cxt.simulate                    │
   │  ─────────────────                                                   │
   │  35 scenarios (constant, sawtooth, island, LLM sweeps,              │
   │  10 stdpopsim mammals, 13 stdpopsim other species)                  │
   │  → .trees files in DATA_DIR/                                        │
   │                                                                      │
   │  STAGE 2: PREPROCESS      python -m cxt.preprocess                  │
   │  ──────────────────                                                  │
   │  5 preprocessed datasets:                                            │
   │    processed                 (w2000, n50, 200 pairs)                │
   │    processed_n10             (w2000, n10, 20 pairs)                 │
   │    processed_small_window    (w200, n50, 200 pairs)                 │
   │    processed_small_window_missing_data        (w200, n50, bitmask)  │
   │    processed_small_window_missing_data_n10    (w200, n10, bitmask)  │
   │  → X.npy, y.npy, pairs.npy per simulation                          │
   │                                                                      │
   │  STAGE 3: TRAIN           python -m cxt.train                       │
   │  ──────────────                                                      │
   │  7 checkpoints in dependency order:                                  │
   │    narrow           ← processed                                     │
   │    broad            ← processed                                     │
   │    residual         ← processed                                     │
   │    broad_w200       ← processed_small_window + broad ckpt           │
   │    w200_wmissing    ← processed_sw_missing + broad_w200 ckpt        │
   │    broad+adapter    ← processed_n10 + broad ckpt                    │
   │    w200_wmissing_adapter ← processed_sw_missing_n10 + w200 ckpt     │
   │                                                                      │
   │  STAGE 4: FIGURES         python -m figures.main.*                   │
   │  ────────────────                                                    │
   │  8 main figures (Fig 1-8) + 6 supplementary (S4, S5, S6, S9-S11)   │
   │  → figures/output/main/ and figures/output/supplementary/            │
   │                                                                      │
   └──────────────────────────────────────────────────────────────────────┘


Configuration
--------------------------------

``run_fresh.sh`` places everything under a single ``BASE_DIR``:

.. code-block:: text

   BASE_DIR/
   ├── data/                  # simulated .trees + preprocessed datasets
   ├── lightning_logs/         # PyTorch Lightning training logs
   ├── checkpoints/           # installed model checkpoints (for cxt.load_model)
   └── figures/output/        # generated figure PNGs
       ├── main/
       └── supplementary/

Hardware and path settings are at the top of the script:

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Variable
     - Description
     - Default
   * - ``BASE_DIR``
     - Root for all outputs
     - ``/sietch_colab/data_share/cxt_scratch``
   * - ``GPUS``
     - GPU indices for training and figures
     - ``1 2``
   * - ``SIM_WORKERS``
     - Parallel workers for simulation
     - ``80``
   * - ``PREPROCESS_WORKERS``
     - Parallel workers for preprocessing
     - ``80``
   * - ``TRAIN_WORKERS``
     - DataLoader workers for training
     - ``16``

The script sets ``CXT_CHECKPOINT_CACHE=$BASE_DIR/checkpoints`` so that
``cxt.load_model()`` uses the freshly trained checkpoints instead of the
global cache at ``~/.cache/cxt/checkpoints/``. It also sets
``CUDA_VISIBLE_DEVICES`` so that figure scripts see the correct GPUs.

Figure-specific external data (also configurable via env):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Default
   * - ``AG1000G_DATA_DIR``
     - ``/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2``
   * - ``AG1000G_ACCESSIBILITY``
     - ``/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz``
   * - ``HG1KG_TSZ_DIR``
     - ``/sietch_colab/data_share/hg1kg/tsinfer-trees/working``


Example: custom paths
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   BASE_DIR=/scratch/myuser/cxt_run ./scripts/run_fresh.sh


Simulation details
------------------

Stage 1 runs 35 simulation batches across four categories:

**Base dataset** (3 scenarios):

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Scenario
     - Samples
     - Description
   * - ``constant``
     - 10,000
     - Constant :math:`N_e = 20{,}000`
   * - ``sawtooth``
     - 1,000
     - Oscillating :math:`N_e` (Schiffels & Durbin zigzag)
   * - ``island``
     - 1,000
     - 3-population island model with migration

**LLM-style sweeps** (4 scenarios):

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Scenario
     - Samples
     - Description
   * - ``llm_ne_sawtooth``
     - 125
     - 3 magnitudes × 3 :math:`N_e` × 2 :math:`\mu` × 2 :math:`r`
   * - ``llm_hard_sweeps``
     - 50
     - 3 :math:`N_e` × 2 :math:`\mu` × 2 :math:`r` × 3 sel. coefficients
   * - ``llm_island_3pop``
     - 50
     - 2 migration × 3 :math:`N_e` × 2 :math:`\mu` × 2 :math:`r`
   * - ``llm_ne_constant``
     - 500
     - 3 :math:`N_e` × 2 :math:`\mu` × 2 :math:`r`

**stdpopsim mammals** (10 scenarios, 1,000 samples each):
``homsap``, ``homsap_map``, ``bostau``, ``canfam``, ``canfam_map``,
``pantro``, ``papanu``, ``papanu_map``, ``ponabe``, ``ponabe_map``

**stdpopsim other species** (18 scenarios, 5--1,000 samples):
``aedaeg``, ``anapla``, ``anocar``, ``anogam``, ``aratha``, ``aratha_map``,
``caeele``, ``caeele_map``, ``dromel``, ``drosec``, ``gasacu``, ``helann``,
``helmel``, ``apimel``, ``musmus``


Preprocessing details
---------------------

Stage 2 creates five preprocessed datasets. See :doc:`preprocessing` for
the full schema. The datasets differ in window size, sample count, and
whether a missingness bitmask is encoded:

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 10 10

   * - Dataset
     - Window
     - Pairs
     - Samples
     - Bitmask
   * - ``processed``
     - 2,000 bp
     - 200
     - 50
     - No
   * - ``processed_n10``
     - 2,000 bp
     - 20
     - 10
     - No
   * - ``processed_small_window``
     - 200 bp
     - 200
     - 50
     - No
   * - ``processed_small_window_missing_data``
     - 200 bp
     - 200
     - 50
     - Yes
   * - ``processed_small_window_missing_data_n10``
     - 200 bp
     - 20
     - 10
     - Yes


Training details
----------------

Stage 3 trains all 7 model checkpoints respecting the dependency chain.
``run_fresh.sh`` installs each checkpoint into ``BASE_DIR/checkpoints/``
after training so that figure generation uses the freshly trained models.

.. code-block:: text

   narrow  (from scratch, 6 epochs)
   broad   (from scratch, 2 epochs)   ← backbone for downstream
   residual (from scratch, 2 epochs)

   broad_w200          ← broad + processed_small_window, lr=3e-5
     └── w200_wmissing ← broad_w200 + processed_sw_missing, lr=3e-5
           └── w200_wmissing_adapter ← adapter, 10 epochs, lr=3e-5

   broad+adapter       ← broad + processed_n10, 3 epochs

See :doc:`training` for hyperparameter tables.


Figure details
--------------

Stage 4 generates all paper figures. Each figure script caches its
intermediate results (simulated tree sequences, TMRCA predictions) so
re-runs skip expensive computation.

**Main figures:**

.. list-table::
   :header-rows: 1
   :widths: 10 25 25 40

   * - Fig
     - Script
     - Models
     - Description
   * - 1
     - ``fig1_model_schematic``
     - narrow
     - Model architecture and batch inference demo
   * - 2
     - ``fig2_benchmark_comparison``
     - narrow, broad
     - True vs predicted TMRCA (cxt, SMC++, SINGER)
   * - 3
     - ``fig3_stdpopsim_v2_coalescence``
     - broad
     - KDE comparison across 16 stdpopsim v0.2 species
   * - 4
     - ``fig4_stdpopsim_v3_ood``
     - broad
     - Out-of-distribution evaluation on 6 v0.3 species
   * - 5
     - ``fig5_demography_inference``
     - broad
     - IICR for H. sapiens, B. taurus, A. thaliana
   * - 6
     - ``fig6_human_1kg``
     - broad
     - TMRCA landscapes for GBR (chr2, chr6, LCT, HLA)
   * - 7
     - ``fig7_mosquito_rdl``
     - w200_wmissing
     - A. gambiae RDL region across 5 populations
   * - 8
     - ``fig8_inversion_coalescence``
     - (uses Fig 7 cache)
     - In(2L)a inversion coalescence patterns

**Supplementary figures:**

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Fig
     - Script
     - Description
   * - S4
     - ``figS4_sample_size_adapter``
     - Sample-size adapter (n=5 vs n=25)
   * - S5
     - ``figS5_window_resolution``
     - Window size effect (w=2000, 200, 20 bp) + residual model
   * - S6
     - ``figS6_runtime_benchmark``
     - Runtime comparison: cxt vs SMC++ vs SINGER
   * - S9
     - ``figS9_mosquito_comparison``
     - RDL region: cxt vs Singer+Polegon vs SMC++
   * - S10
     - ``figS10_cross_coalescence``
     - Cross-population coalescence (OutOfAfrica_2T12)
   * - S11
     - ``figS11_interpolation_grid``
     - Mutation × recombination grid evaluation

Outputs land in ``figures/output/main/`` and ``figures/output/supplementary/``.


External data dependencies
--------------------------

Some figures require external genomic datasets that are **not** generated by
the simulation stage:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Figures
     - Variable
     - Data
   * - 6
     - ``HG1KG_TSZ_DIR``
     - Human 1000 Genomes tsinfer trees + masks
   * - 7, S9
     - ``AG1000G_DATA_DIR``
     - Ag1000G chr2L dated trees
   * - 7, S9
     - ``AG1000G_ACCESSIBILITY``
     - Ag1000G per-site accessibility bitmask
   * - S6
     - (paths.py)
     - Benchmark timing logs (JSONL)
   * - S9
     - (paths.py)
     - SINGER and SMC++ revision caches

These paths are resolved via ``figures/paths.py`` and can be overridden
with environment variables.


Logs
----

``run_fresh.sh`` writes a timestamped log file
(``run_fresh_YYYYMMDD_HHMMSS.log``) under ``BASE_DIR``. The log captures
stdout/stderr from all stages and ends with a summary of wall times and
any failures.
