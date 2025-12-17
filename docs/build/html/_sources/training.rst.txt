Training
========

``cxt`` is trained using the Pytorch Lightning framework. The following code demonstrates how to train the model.

Training cxt models
===================

This page documents the command-line recipes used to train the pretrained
:math:`\mathbf{cxt}` model variants shipped with the toolkit.

All training runs are launched via ``train2.py`` (standard sample size) or
``train2_n10.py`` (sample-size transfer / adapter training), using multi-GPU
data-parallel training.

The main knobs are:

- ``--dataset_path``: path to a preprocessed training dataset directory
- ``--gpus``: list of GPU indices to use (e.g. ``0 1 2``)
- ``--num_epochs``: number of epochs
- ``--learning_rate``: (optional) override learning rate for fine-tuning
- ``--checkpoint_path``: (optional) initialize from a pretrained checkpoint
  (fine-tuning / continued training)

---

Narrow model
------------

The ``narrow`` model is trained from scratch on the default processed dataset
and typically run for more epochs than ``broad``.

.. code-block:: bash

    python train2.py \
      --dataset_path /sietch_colab/kkor/cxt/ts/processed \
      --gpus 0 1 2 \
      --num_epochs 6

---

Broad models
------------

Vanilla broad model
^^^^^^^^^^^^^^^^^^^

The vanilla ``broad`` model is trained from scratch on the default dataset and
converges quickly (short training schedule).

.. code-block:: bash

    python train2.py \
      --dataset_path /sietch_colab/kkor/cxt/ts/processed \
      --gpus 0 1 2 \
      --num_epochs 2

Broad_w200 (fine-tuning for large populations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``broad_w200`` model is obtained by fine-tuning the vanilla broad checkpoint
on a dataset with a smaller window size (``w200``) and large-population regimes.
This run uses a reduced learning rate and a pretrained initialization via
``--checkpoint_path``.

.. code-block:: bash

    python train2.py \
      --dataset_path /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window \
      --gpus 0 1 2 \
      --num_epochs 2 \
      --learning_rate 3e-5 \
      --checkpoint_path /home/kkor/cxt/cxt/lightning_logs/version_20/checkpoints/epoch=1-step=5280.ckpt

Broad_w200_missing (missingness-aware fine-tuning)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This variant fine-tunes from the same broad checkpoint, but on datasets that
include missingness patterns. **Important:** ensure singleton masking is
deactivated for this training regime (either by configuration or manually).

.. code-block:: bash

    # Make sure to deactivate singleton masking (if not done automatically)
    python train2.py \
      --dataset_path /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window_missing_data \
      --gpus 0 1 2 \
      --num_epochs 2 \
      --learning_rate 3e-5 \
      --checkpoint_path /home/kkor/cxt/cxt/lightning_logs/version_20/checkpoints/epoch=1-step=5280.ckpt

Broad_w200_missing (longer schedule)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same as above, but with a longer fine-tuning schedule to improve robustness and
reduce variance under heavy masking.

.. code-block:: bash

    # Same as previous but with more epochs
    python train2.py \
      --dataset_path /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window_missing_data \
      --gpus 0 1 2 \
      --num_epochs 10 \
      --learning_rate 3e-5 \
      --checkpoint_path /home/kkor/cxt/cxt/lightning_logs/version_20/checkpoints/epoch=1-step=5280.ckpt

---

Adapter / n10 training
----------------------

In addition to the base models, :math:`\mathbf{cxt}` supports sample-size
transfer using lightweight adapter modules. These are trained via
``train2_n10.py`` and typically initialize from a broad-family checkpoint.

Broad + adapter (n10)
^^^^^^^^^^^^^^^^^^^^^

This script is currently configured to automatically load the broad model
checkpoint (if not provided explicitly) and train an adapter for inference on
``n=10`` sample settings.

.. code-block:: bash

    python train2_n10.py \
      --dataset_path /sietch_colab/kkor/cxt/ts/processed_n10 \
      --gpus 0 1 2 \
      --num_epochs 3

Broad_w200_missing_n10 (all-in-one)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This run trains the missingness-aware small-window / n10 setup, initializing
from the ``broad+adapter`` checkpoint. Adjust ``--num_epochs`` depending on
whether you want a short run or a longer schedule.

.. code-block:: bash

    # uses broad+adapter checkpoint
    python train2_n10.py \
      --dataset_path /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window_missing_data_n10 \
      --gpus 0 1 2 \
      --num_epochs 2 \
      --learning_rate 3e-5 \
      --checkpoint_path /sietch_colab/data_share/cxt/models/broad+adapter/version_26/checkpoints/epoch=2-step=792.ckpt

    # longer schedule (same command, more epochs)
    python train2_n10.py \
      --dataset_path /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window_missing_data_n10 \
      --gpus 0 1 2 \
      --num_epochs 10 \
      --learning_rate 3e-5 \
      --checkpoint_path /sietch_colab/data_share/cxt/models/broad+adapter/version_26/checkpoints/epoch=2-step=792.ckpt

---

Practical notes
---------------

- **Checkpoints:** training uses PyTorch Lightning checkpoints; the
  ``--checkpoint_path`` argument performs warm-start initialization and is used
  for fine-tuning regimes (e.g., ``broad_w200`` and missingness-aware variants).

- **Datasets:** each ``--dataset_path`` directory is expected to contain the
  preprocessed windows/blocks produced by the dataset preprocessing pipeline.
  Use consistent preprocessing settings (window size, masking rules) between
  training and the intended downstream inference use-case.

- **Singleton masking:** for missingness-aware training, singleton masking
  should be disabled to avoid confounding real missingness with artificially
  dropped rare variants.

- **Multi-GPU:** ``--gpus 0 1 2`` selects GPUs by index. Ensure that the CUDA
  visible devices match the indices you pass, especially on shared machines.
