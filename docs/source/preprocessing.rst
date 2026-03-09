Preprocessing
=============

After simulation, tree sequences must be preprocessed into the ``(X, y)``
format expected by the training pipeline. The preprocessing module
(``python -m cxt.preprocess``) extracts multi-scale windowed SFS features
and discretized log-TMRCA targets for each haplotype pair.


What preprocessing does
-----------------------

For each tree sequence and each sampled pair of haplotypes:

1. **Feature extraction** (``X``): Compute the XOR and XNOR site-frequency
   spectra at four window scales (2×, 8×, 32×, 64× the base window), yielding
   a tensor of shape ``(2, 4, n_windows, n_samples)``.

2. **Target extraction** (``y``): Compute the true pairwise TMRCA per window
   via span-weighted averaging from the simplified two-sample tree, then
   apply a log transform.

3. **Data splitting**: Files are deterministically assigned to ``train/``
   or ``test/`` splits using a grouped hash that ensures all pairs from the
   same simulation scenario stay in the same split.

Output structure:

.. code-block:: text

   <out_dir>/
   ├── train/
   │   └── <scenario>/<file_id>/
   │       ├── X.npy      # (n_pairs, 2, 4, n_windows, n_samples) float16
   │       ├── y.npy      # (n_pairs, n_windows) float16
   │       ├── pairs.npy  # (n_pairs, 2) int
   │       └── meta.json
   └── test/
       └── ...


CLI reference
-------------

.. code-block:: text

   python -m cxt.preprocess \
       --base_dir <dir_with_tree_sequences> \
       --out_subdir <output_name> \
       --window_size 2000 \
       --num_pairs 200 \
       --train_ratio 0.9 \
       --global_seed 12345 \
       --num_workers 75 \
       [--skip_existing] \
       [--simplify_first_n_samples 50] \
       [--bitmask /path/to/bitmask.npz]


Paper datasets
--------------

The following six preprocessed datasets are needed to reproduce all
checkpoints used in the paper. Each corresponds to a different training
regime.

.. note::

   ``DATA_DIR`` refers to the directory containing simulated tree sequences
   (see :doc:`simulation`). ``DATA_DIR_LP`` is the same directory but may
   contain additional large-population simulations used for fine-tuning the
   w200 variants. In ``run_fresh.sh``, a ``ts_large_pop`` symlink tree is
   created that links only the high-Ne stdpopsim species.


0. ``processed_narrow`` -- constant-only baseline (w2000)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used to train: **narrow**

Contains only the constant-:math:`N_e` scenario. This is a smaller dataset
used for the 6-layer narrow model.

.. code-block:: bash

   python -m cxt.preprocess \
       --base_dir ${DATA_DIR}/base_dataset \
       --out_subdir processed_narrow \
       --window_size 2000 \
       --num_pairs 200 \
       --train_ratio 0.9 \
       --global_seed 12345 \
       --num_workers 75 \
       --skip_existing


1. ``processed`` -- full base dataset (w2000)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used to train: **broad**

.. code-block:: bash

   python -m cxt.preprocess \
       --base_dir ${DATA_DIR} \
       --out_subdir processed \
       --window_size 2000 \
       --num_pairs 200 \
       --train_ratio 0.9 \
       --global_seed 12345 \
       --num_workers 75 \
       --skip_existing


2. ``processed_n10`` -- adapter dataset (10 samples)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used to train: **broad+adapter**

Simplifies each tree sequence to 10 haploid samples before feature
extraction. Uses fewer pairs (20) since the combinatorial space is smaller.

.. code-block:: bash

   python -m cxt.preprocess \
       --base_dir ${DATA_DIR} \
       --out_subdir processed_n10 \
       --window_size 2000 \
       --num_pairs 20 \
       --simplify_first_n_samples 10 \
       --train_ratio 0.9 \
       --global_seed 12345 \
       --num_workers 75


3. ``processed_small_window`` -- w200 dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used to train: **broad_w200**

Uses 200 bp windows and 100 kb sequence length for fine-scale resolution
in large-\ :math:`N_e` species.

.. code-block:: bash

   python -m cxt.preprocess \
       --base_dir ${DATA_DIR_LP} \
       --out_subdir processed_small_window \
       --window_size 200 \
       --sequence_length 100000 \
       --num_pairs 200 \
       --train_ratio 0.9 \
       --global_seed 12345 \
       --num_workers 75 \
       --skip_existing


4. ``processed_small_window_missing_data`` -- w200 + missingness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used to train: **w200_wmissing**

Same as above but incorporates an accessibility bitmask (from e.g.
Ag1000G) to encode per-window missingness into the source tensor.

.. code-block:: bash

   python -m cxt.preprocess \
       --base_dir ${DATA_DIR_LP} \
       --out_subdir processed_small_window_missing_data \
       --window_size 200 \
       --sequence_length 100000 \
       --num_pairs 200 \
       --train_ratio 0.9 \
       --global_seed 12345 \
       --num_workers 75 \
       --skip_existing \
       --bitmask /path/to/bitmask.npz


5. ``processed_small_window_missing_data_n10`` -- w200 + missingness + adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Used to train: **w200_wmissing_adapter**

Combines small-window missingness with 10-sample simplification.

.. code-block:: bash

   python -m cxt.preprocess \
       --base_dir ${DATA_DIR_LP} \
       --out_subdir processed_small_window_missing_data_n10 \
       --window_size 200 \
       --sequence_length 100000 \
       --num_pairs 20 \
       --simplify_first_n_samples 10 \
       --train_ratio 0.9 \
       --global_seed 12345 \
       --num_workers 75 \
       --skip_existing \
       --bitmask /path/to/bitmask.npz


Dataset summary
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 10 10 10 15

   * - Dataset
     - Window
     - Pairs
     - Samples
     - Missingness
   * - ``processed_narrow``
     - 2,000 bp
     - 200
     - 50
     - No (constant only)
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
