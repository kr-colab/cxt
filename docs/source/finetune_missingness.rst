Recipe: Fine-tuning for Missingness
====================================

This recipe walks through fine-tuning a cxt model to handle missing or
inaccessible genomic sites. The result is a model that encodes per-window
missingness directly into its source tensor, rather than treating absent
sites as carrying no information.


When to fine-tune for missingness
---------------------------------

The pretrained ``broad`` and ``broad_w200`` models assume every base in each
window is observable. When applied to empirical data with substantial
inaccessible regions (low-coverage masking, repetitive-element filters,
accessibility masks), they can produce biased TMRCA estimates in windows
where a large fraction of sites is missing.

Fine-tuning for missingness teaches the model what "no data" looks like at
each window scale, so that it can distinguish genuine absence of mutations
from unobserved sequence.

Typical use cases:

- Whole-genome sequencing with accessibility masks (e.g. Ag1000G, 1000
  Genomes strict-mask regions).
- Reduced-representation or capture data where only a subset of bases is
  sequenced.
- Any dataset where the callable fraction varies substantially across the
  genome.


Overview
--------

The pipeline has four stages:

1. **Obtain a bitmask** -- a per-base boolean array marking inaccessible
   positions.
2. **Simulate training data** -- or reuse existing tree sequences from the
   stdpopsim catalog.
3. **Preprocess with the bitmask** -- the ``--bitmask`` flag injects random
   missingness crops from your mask into each training example.
4. **Fine-tune** -- warm-start from the ``broad_w200`` checkpoint and train
   on the missingness-augmented dataset.

After fine-tuning, the model can be used for inference with the
``missingness_bitmask`` argument to :func:`cxt.translate`.


Step 1: Prepare a bitmask
--------------------------

A bitmask is a 1-D boolean NumPy array where ``True`` means *accessible*
and ``False`` means *inaccessible*. Its length should be at least as long as
the chromosome you plan to analyse. During preprocessing, a random crop of
length ``sequence_length`` is drawn from the bitmask for each training
example.

**From an Ag1000G-style accessibility file:**

.. code-block:: python

   import numpy as np

   acc = np.load("accessibility.npz")["access_2L"]   # True = accessible
   unaccessible = ~acc                                 # True = inaccessible

   np.savez_compressed("unaccessible_bitmask.npz", access_2L=acc)

**From a BED file of callable regions:**

.. code-block:: python

   import numpy as np

   chrom_length = 49_364_325  # e.g. Ag chr2L
   mask = np.zeros(chrom_length, dtype=bool)

   with open("callable_regions.bed") as fh:
       for line in fh:
           chrom, start, end = line.split()[:3]
           mask[int(start):int(end)] = True

   np.savez_compressed("my_bitmask.npz", access_2L=mask)

The key in the ``.npz`` file should match the key referenced in your
preprocessing command (by default ``access_2L``).


Step 2: Simulate or gather training data
------------------------------------------

Fine-tuning for missingness uses the same simulated tree sequences as the
``broad_w200`` model -- typically the 13 high-:math:`N_e` stdpopsim species.
If you have already simulated these (see :doc:`simulation`), point
``DATA_DIR_LP`` at the directory containing them.

If starting from scratch, generate a minimal set:

.. code-block:: bash

   DATA_DIR=/path/to/training_data
   SIM="python cxt/simulation_ts_only.py"

   # A. gambiae (100 simulations)
   ${SIM} --num_processes 50 --num_samples 100 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_anogam \
       --scenario stdpopsim_anogam

   # A. aegypti (300 simulations)
   ${SIM} --num_processes 50 --num_samples 300 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_aedaeg \
       --scenario stdpopsim_aedaeg

Add as many species as needed. The more diversity in the training set, the
more robust the resulting model. See :doc:`simulation` for the full list.


Step 3: Preprocess with the bitmask
-------------------------------------

Use ``python -m cxt.preprocess`` with the ``--bitmask`` flag. This does two
things for each training example:

- Draws a random crop from the bitmask and deletes the inaccessible
  intervals from the tree sequence before computing the SFS.
- Encodes per-window missingness fractions at each scale into the source
  tensor (channels ``X[0,:,:,0]`` and ``X[1,:,:,0]``).

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
       --bitmask /path/to/my_bitmask.npz

This produces a dataset with the same structure as any other preprocessed
directory (``train/`` and ``test/`` subdirectories with ``X.npy``,
``y.npy``, ``pairs.npy``, and ``meta.json``), except that ``X`` now
contains the missingness channels.


Step 4: Fine-tune from ``broad_w200``
--------------------------------------

Fine-tuning warm-starts from the ``broad_w200`` checkpoint and continues
training on the missingness-augmented data at a reduced learning rate. The
``w200_wmissing`` preset automatically sets ``mask_singletons=False``, which
is required for missingness-aware models.

.. code-block:: bash

   python -m cxt.train \
       --model w200_wmissing \
       --dataset-path /path/to/processed_small_window_missing_data \
       --gpus 0 1 \
       --epochs 2 \
       --lr 3e-5 \
       --checkpoint broad_w200

The ``--checkpoint`` argument accepts either a local path to a ``.ckpt``
file or the name of a pretrained model type (e.g. ``broad_w200``), in which
case the checkpoint is downloaded automatically.

Training produces a checkpoint under ``lightning_logs/`` (or the directory
set by ``--log-dir``).


Step 5: Run inference with the fine-tuned model
-------------------------------------------------

Load the checkpoint and pass ``missingness_bitmask`` to
:func:`cxt.translate`:

.. code-block:: python

   import numpy as np
   import cxt

   model = cxt.load_model(
       "/path/to/lightning_logs/version_X/checkpoints/epoch=1-step=944.ckpt",
       device="cuda",
   )

   bitmask = np.load("my_bitmask.npz")["access_2L"]
   unaccessible = ~bitmask

   tmrca, index_map = cxt.translate(
       "genotypes.vcf",
       model,
       pivot_pairs=[(0, 1), (2, 3)],
       blocks=[(0, 100_000)],
       devices=["cuda:0"],
       missingness_bitmask=unaccessible,
       mutation_rate=3.5e-9,
       B=128,
       build_workers=8,
   )

The ``missingness_bitmask`` should be a boolean array of shape
``(chromosome_length,)`` where ``True`` marks *inaccessible* positions.


Optional: Adapter for small sample sizes
-----------------------------------------

If your empirical dataset has fewer samples than the 50 haplotypes the model
was trained on, you can add an adapter on top of the missingness model.

**Preprocess adapter data** (10 samples, 20 pairs):

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
       --bitmask /path/to/my_bitmask.npz

**Train the adapter** (two-stage: resume from a ``broad+adapter`` checkpoint
that already learned the 10 → 50 sample mapping):

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

**Inference with the adapter:**

.. code-block:: python

   adapter_model = cxt.load_model(
       "/path/to/w200_wmissing_adapter.ckpt",
       device="cuda",
   )

   tmrca, index_map = cxt.translate(
       ts_small,
       adapter_model.backbone,
       pivot_pairs=[(0, 1)],
       blocks=[(0, 100_000)],
       devices=["cuda:0"],
       missingness_bitmask=unaccessible,
       mutation_rate=3.5e-9,
       adapter=adapter_model.adapter,
   )

See :doc:`mosquito` for a complete worked example using both the
missingness model and its adapter on the Ag1000G dataset.
