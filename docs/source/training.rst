Training
========

All cxt model variants are trained using a single unified script
(``python -m cxt.train``) built on PyTorch Lightning. This page documents the
exact commands that reproduce every checkpoint shipped with the package.


Checkpoint dependency graph
---------------------------

The six released checkpoints form a dependency chain. Models lower in
the graph are fine-tuned from checkpoints higher up:

.. code-block:: text

   narrow  (from scratch on "processed_narrow" — constant only)
   broad  (from scratch on "processed")
     ├── broad+adapter  (adapter on "processed_n10", frozen broad backbone)
     │     └── w200_wmissing_adapter  (--resume-adapter on "processed_small_window_missing_data_n10")
     └── broad_w200  (fine-tuned on "processed_small_window")
           └── w200_wmissing  (fine-tuned on "processed_small_window_missing_data")


Training data summary
---------------------

Which model trains on which dataset and whether it uses fine-tuning:

.. list-table::
   :header-rows: 1
   :widths: 18 18 25 18 10 10 12

   * - Model
     - Fine-tuning
     - Dataset
     - Source scenarios
     - Window
     - Samples
     - Pairs
   * - ``narrow``
     - No (from scratch)
     - ``processed_narrow``
     - constant only
     - w2000
     - 50
     - 200
   * - ``broad``
     - No (from scratch)
     - ``processed``
     - all
     - w2000
     - 50
     - 200
   * - ``broad_w200``
     - Yes ← ``broad``
     - ``processed_small_window``
     - 13 high-Ne stdpopsim
     - w200
     - 50
     - 200
   * - ``broad+adapter``
     - Yes ← ``broad``
     - ``processed_n10``
     - all
     - w2000
     - 10
     - 20
   * - ``w200_wmissing``
     - Yes ← ``broad_w200``
     - ``processed_small_window_missing_data``
     - 13 high-Ne stdpopsim
     - w200
     - 50
     - 200 + bitmask
   * - ``w200_wmissing_adapter``
     - Yes ← ``broad+adapter`` (``--resume-adapter``)
     - ``processed_small_window_missing_data_n10``
     - 13 high-Ne stdpopsim
     - w200
     - 10
     - 20 + bitmask


CLI reference
-------------

.. code-block:: text

   python -m cxt.train \
       --model <preset_name> \
       --dataset-path <preprocessed_dir> \
       --gpus <gpu_ids> \
       --epochs <n> \
       [--lr <learning_rate>] \
       [--batch-size <bs>] \
       [--grad-accum <steps>] \
       [--checkpoint <path_or_model_type>] \
       [--adapter] \
       [--adapter-samples <n>] \
       [--adapter-bottleneck <dim>] \
       [--adapter-dropout <float>] \
       [--resume-adapter <ckpt_path>] \
       [--log-dir <directory>]

Key arguments:

- ``--model``: preset name (``narrow``, ``broad``, ``broad_w200``,
  ``residual``, ``w200_wmissing``)
- ``--dataset-path``: path to a preprocessed dataset (must contain
  ``train/`` and ``test/`` subdirectories)
- ``--gpus``: GPU indices (e.g. ``0 1``)
- ``--checkpoint``: warm-start from checkpoint (for fine-tuning)
- ``--adapter``: enable adapter training with frozen backbone
- ``--adapter-samples``: adapter input dimension (sample count)
- ``--resume-adapter``: resume adapter training from a full
  ``LitAdapterDecoder`` checkpoint (loads both backbone and adapter
  weights, not just the backbone). Used for two-stage adapter training
  (e.g. training ``w200_wmissing_adapter`` from ``broad+adapter``).
- ``--log-dir``: root directory for ``lightning_logs/`` (default: current
  working directory). Use this to redirect checkpoints and TensorBoard
  logs to an isolated directory.


Reproducing all checkpoints
---------------------------

Below are the exact commands for each checkpoint. Replace paths with your
actual directories.


1. ``narrow``
^^^^^^^^^^^^^

Trained from scratch on the constant-\ :math:`N_e` dataset only. 6 layers,
6 epochs.

**Checkpoint:** ``narrow_epoch=5-step=4692.ckpt``

.. code-block:: bash

   python -m cxt.train \
       --model narrow \
       --dataset-path /path/to/base_dataset/processed_narrow \
       --gpus 0 1 \
       --epochs 6


2. ``broad``
^^^^^^^^^^^^

Trained from scratch. 10 layers, 2 epochs. This is the main model and
serves as the backbone for all downstream fine-tuning.

**Checkpoint:** ``broad_epoch=1-step=5280.ckpt``

.. code-block:: bash

   python -m cxt.train \
       --model broad \
       --dataset-path /path/to/processed \
       --gpus 0 1 \
       --epochs 2


3. ``broad_w200``
^^^^^^^^^^^^^^^^^

Fine-tuned from the ``broad`` checkpoint on 200 bp window data. Uses a
reduced learning rate.

**Checkpoint:** ``broad_w200_epoch=1-step=944.ckpt``

.. code-block:: bash

   python -m cxt.train \
       --model broad_w200 \
       --dataset-path /path/to/processed_small_window \
       --gpus 0 1 \
       --epochs 2 \
       --lr 3e-5 \
       --checkpoint /path/to/broad_epoch=1-step=5280.ckpt


4. ``w200_wmissing``
^^^^^^^^^^^^^^^^^^^^

Fine-tuned from the ``broad_w200`` checkpoint on data with encoded
missingness. The ``w200_wmissing`` preset automatically sets
``mask_singletons=False``.

**Checkpoint:** ``w200_wmissing_epoch=1-step=944.ckpt``

.. code-block:: bash

   python -m cxt.train \
       --model w200_wmissing \
       --dataset-path /path/to/processed_small_window_missing_data \
       --gpus 0 1 \
       --epochs 2 \
       --lr 3e-5 \
       --checkpoint /path/to/broad_w200_epoch=1-step=944.ckpt


5. ``broad+adapter``
^^^^^^^^^^^^^^^^^^^^

Lightweight adapter on top of a frozen ``broad`` backbone. Trained on
10-sample data. The adapter learns to map from 10 input samples to the
50-sample feature space expected by the backbone.

**Checkpoint:** ``broad_adapter_epoch=2-step=792.ckpt``

.. code-block:: bash

   python -m cxt.train \
       --model broad \
       --adapter \
       --adapter-samples 10 \
       --dataset-path /path/to/processed_n10 \
       --gpus 0 1 \
       --epochs 3 \
       --checkpoint /path/to/broad_epoch=1-step=5280.ckpt


6. ``w200_wmissing_adapter``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two-stage adapter training: the ``broad+adapter`` checkpoint already
learned the 10→50 sample mapping on w2000 data. The second stage resumes
that adapter on w200 + bitmask data at low learning rate to adapt for
missingness. The ``--resume-adapter`` flag loads the full
``LitAdapterDecoder`` checkpoint (backbone + adapter weights).

**Checkpoint:** ``w200_wmissing_adapter_epoch=9-step=480.ckpt``

.. code-block:: bash

   python -m cxt.train \
       --model w200_wmissing \
       --adapter \
       --adapter-samples 10 \
       --resume-adapter /path/to/broad_adapter_epoch=2-step=792.ckpt \
       --dataset-path /path/to/processed_small_window_missing_data_n10 \
       --gpus 0 1 \
       --epochs 10 \
       --lr 3e-5


End-to-end summary
------------------

For reference, here is the full pipeline from simulation to checkpoints:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │ 1. SIMULATE  (python cxt/simulation_ts_only.py)            │
   │    → base_dataset, ssd, idd, llm/*, stdpopsim/*            │
   │                                                             │
   │ 2. PREPROCESS  (python -m cxt.preprocess)                  │
   │    → processed_narrow    (w2000, 50 samples, constant only)│
   │    → processed           (w2000, 50 samples, 200 pairs)    │
   │    → processed_n10       (w2000, 10 samples, 20 pairs)     │
   │    → processed_small_window           (w200, 50 samples)   │
   │    → processed_small_window_missing_data       (+ bitmask) │
   │    → processed_small_window_missing_data_n10   (+ n10)     │
   │                                                             │
   │ 3. TRAIN  (python -m cxt.train)                            │
   │    narrow           ← processed_narrow (constant only)     │
   │    broad            ← processed                            │
   │    broad_w200       ← processed_small_window + broad ckpt  │
   │    broad+adapter    ← processed_n10 + broad ckpt           │
   │    w200_wmissing    ← processed_sw_missing + broad_w200    │
   │    w200_wmissing_adapter ← processed_sw_missing_n10        │
   │                           + broad+adapter (--resume-adapter)│
   └─────────────────────────────────────────────────────────────┘


Training hyperparameters
------------------------

Default training hyperparameters (from :class:`cxt.config.TrainingConfig`):

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Parameter
     - Default
   * - Learning rate
     - 3e-4
   * - Min learning rate
     - 3e-5
   * - Warmup iterations
     - 10
   * - LR decay iterations
     - 150,000
   * - Batch size
     - 128
   * - Gradient accumulation
     - 6
   * - Weight decay
     - 0.1
   * - Optimizer betas
     - (0.9, 0.95)
   * - Precision
     - bf16-mixed
   * - Strategy
     - DDP


Practical notes
---------------

- **Checkpoints** are saved by PyTorch Lightning under ``lightning_logs/``
  in the directory specified by ``--log-dir`` (or the current working
  directory if not set). The ``--checkpoint`` argument performs warm-start
  initialization for fine-tuning.

- **Checkpoint cache**: ``cxt.load_model()`` looks for pretrained
  checkpoints in ``~/.cache/cxt/checkpoints/`` by default. Set the
  ``CXT_CHECKPOINT_CACHE`` environment variable to redirect this (useful
  for isolated reproduction runs that should not touch the global cache).

- **Singleton masking** is controlled by the model preset.
  ``w200_wmissing`` has ``mask_singletons=False`` automatically.

- **KV cache** is disabled during training and enabled automatically at
  inference via :func:`cxt.load_model`.

- **Multi-GPU** training uses DDP. ``--gpus 0 1`` selects GPUs by index.

- **Adapter training** freezes the backbone and only updates the adapter
  module plus selected layer-norm and last-N transformer blocks
  (controlled by ``unfreeze_strategy="ln_lastN"``).

- **Two-stage adapter training** (used for ``w200_wmissing_adapter``):
  first train an adapter on w2000 data (``broad+adapter``), then resume
  with ``--resume-adapter`` on w200 + bitmask data. This transfers the
  learned sample-size mapping while adapting for missingness and smaller
  windows.
