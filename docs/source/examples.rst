Window-wise Decoding with cxt
=============================

This example demonstrates how :math:`\mathbf{cxt}` is used to decode
pairwise time-to-most-recent-common-ancestor (TMRCA) trajectories from
tree sequences using the high-level :func:`cxt.api2.translate` interface.

The pipeline is:

1. Load pretrained :math:`\mathbf{cxt}` model variants.
2. Simulate a recombining tree sequence using :mod:`msprime`.
3. Decode TMRCA trajectories on fixed genomic blocks for selected pivot pairs.
4. (Optional) Apply lightweight adapters to transfer models across sample sizes.

---

Model loading
-------------

:cxt: ships with multiple pretrained model variants that differ in window size,
architectural bias, and robustness to missing data. Models are loaded via
:func:`cxt.utils.setup_cxt_model`, which automatically resolves cached
checkpoints.

.. code-block:: python

    from cxt.utils import setup_cxt_model

    model_types = [
        "broad",
        "broad+adapter",
        "narrow",
        "broad_w200",
        "residual",
        "w200_wmissing",
        "w200_wmissing_adapter",
    ]

    models = {}
    for model_type in model_types:
        print(f"Loading {model_type}...")
        models[model_type] = setup_cxt_model(model_type=model_type)
        print(f"{model_type} loaded successfully!\n")

Typical output:

.. code-block:: text

    Loading broad...
    Using cached checkpoint: ~/.cache/cxt/checkpoints/broad/broad_epoch=1-step=5280.ckpt
    broad loaded successfully!

    Loading broad+adapter...
    Using cached checkpoint: ~/.cache/cxt/checkpoints/broad+adapter/broad_adapter_epoch=2-step=792.ckpt
    broad+adapter loaded successfully!

Cached checkpoints are reused automatically if available.

---

Simulating input data
---------------------

For demonstration purposes, we simulate a recombining tree sequence with
:mod:`msprime`.

.. code-block:: python

    import msprime

    ts = msprime.sim_ancestry(
        25,
        recombination_rate=1e-8,
        sequence_length=1e6,
        population_size=2e4,
        random_seed=42,
    )

    ts = msprime.mutate(
        ts,
        rate=1.29e-8,
        random_seed=42,
    )

The :func:`translate` API also accepts empirical tree sequences,
genotype matrices, or VCF-backed datasets.

---

Decoding pairwise TMRCA
-----------------------

The central inference step is performed using :func:`cxt.api2.translate`.
Here we decode the TMRCA trajectory for a single pivot pair across a
1 Mb genomic block.

.. code-block:: python

    from cxt.api2 import translate

    blocks = [(0, 1e6)]
    pivot_pairs = [(0, 1)]
    devices = ["cuda:0"]
    B = 128

    yhat_tmrca, index_map = translate(
        input_data=ts,
        data_type="ts",
        model=models["broad"],
        pivot_pairs=pivot_pairs,
        blocks=blocks,
        devices=devices,
        B_per_device=B,
        B=B,
        build_workers=8,
        mutation_rate=1.29e-8,
    )

**Outputs**

- ``yhat_tmrca``  
  Predicted log-TMRCA values indexed by pivot pair and genomic window.

- ``index_map``  
  Mapping from model-internal indices to genomic coordinates and pivot pairs.

Decoding scales linearly with the number of windows and is highly parallelizable
across GPUs.

---

Sample-size transfer using adapters
-----------------------------------

Some models include lightweight **adapter modules** that enable inference on
sample sizes different from those seen during training.

In the example below, we simplify the tree sequence to 10 samples and apply
a pretrained backbone together with its adapter.

.. code-block:: python

    ts_n10 = ts.simplify(samples=range(10))

    yhat_tmrca, index_map = translate(
        input_data=ts_n10,
        data_type="ts",
        model=models["broad+adapter"].backbone,
        pivot_pairs=pivot_pairs,
        blocks=blocks,
        devices=devices,
        B_per_device=B,
        B=B,
        build_workers=8,
        mutation_rate=1.29e-8,
        adapter=models["broad+adapter"].adapter,
    )

This allows **sample-size–robust decoding** without retraining the full model.

---

Summary
-------

This example illustrates the minimal workflow for decoding pairwise coalescence
times with :math:`\mathbf{cxt}`:

- Pretrained models are loaded via :func:`setup_cxt_model`
- Tree sequences are decoded window-wise using :func:`translate`
- Adapter modules enable flexible transfer across sample sizes
- The same interface applies to simulated and empirical data, as well as genotype matrices and VCFs.

