Installation
============

From source
-----------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone git@github.com:kevinkorfmann/cxt.git
   cd cxt
   pip install -e .

Checkpoints are downloaded automatically on first use via
:func:`cxt.load_model` and cached in ``~/.cache/cxt/checkpoints/``.
To redirect the cache (e.g. for isolated reproduction runs), set the
``CXT_CHECKPOINT_CACHE`` environment variable:

.. code-block:: bash

   export CXT_CHECKPOINT_CACHE=/path/to/my/checkpoints

If you prefer to have checkpoints available offline via Git LFS:

.. code-block:: bash

   conda install anaconda::git-lfs   # or: apt install git-lfs
   git lfs install
   git lfs pull

Requirements
------------

- Python 3.10+
- PyTorch 2.0+
- Lightning (for training)
- msprime, tskit, stdpopsim (for simulation)
- numpy, scipy, pandas, einops, tqdm

A CUDA-capable GPU is strongly recommended for inference.
Multi-GPU setups (2--4 GPUs) provide near-linear speedups.

Optional dependencies
---------------------

- ``tszip`` -- for compressed tree-sequence storage (human and mosquito examples)
- ``stdpopsim`` -- for species-specific simulation scenarios
- ``matplotlib`` -- for plotting examples
- ``requests`` -- for automatic checkpoint downloads

Verifying the installation
--------------------------

.. code-block:: python

   import cxt

   model = cxt.load_model("broad", device="cpu")
   print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
