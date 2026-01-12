Quick start (example from paper using narrow model)
===================================================

.. note::

This example demonstrates the full workflow used to generate the
:math:`\mathbf{cxt}`-narrow constant-\ :math:`N_e` benchmark figure from the paper.
It covers simulation, inference, and comparison between inferred and true pairwise
coalescent times. In most cases the broad model is recommended for practical use;
see the examples page for a general decoding tutorial.

Simulation
----------

We begin by simulating a 1 Mb tree sequence under a constant-\ :math:`N_e` model:

.. code-block:: python

    import os
    import numpy as np

    from cxt.utils import simulate_parameterized_tree_sequence

    # Directory for intermediate results
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Simulate a single tree sequence (1 Mb)
    ts = simulate_parameterized_tree_sequence(seed=103370001)

Setting up the model and pivot pairs
------------------------------------

Next, we load the :math:`\mathbf{cxt}`-narrow model and construct all pairwise
sample combinations (“pivot pairs”) among 50 samples.

.. code-block:: python

    from cxt.utils import setup_cxt_model

    # Load narrow model
    model = setup_cxt_model(model_type="narrow")

    # Genomic block (1 Mb)
    blocks = [(0, 1_000_000)]

    # Generate all pairwise sample combinations among 50 samples
    num_samples = 50
    pivot_pairs = [
        (i, j)
        for i in range(num_samples)
        for j in range(i + 1, num_samples)
    ]


.. note::

    Depending on how genomic blocks and pivot pairs are specified, two common usage
    patterns naturally arise. In a *targeted* pattern, inference is concentrated on
    specific loci or narrow genomic intervals by evaluating many pivot pairs at those
    positions, increasing depth at regions of interest. In a complementary *long-range*
    pattern, inference is distributed across large, contiguous genomic regions using a
    more limited set of pivot pairs, enabling efficient genome-wide or chromosome-scale
    decoding.

    In the configuration shown above, inference budget is allocated primarily to
    pairwise depth: a single 1 Mb genomic block is decoded while all pairwise
    combinations among 50 samples are evaluated:

    .. code-block:: python

        # Targeted pattern: few blocks, many pivot pairs
        blocks = [(0, 1_000_000)]

        num_samples = 50
        pivot_pairs = [
            (i, j)
            for i in range(num_samples)
            for j in range(i + 1, num_samples)
        ]

    Hypothetically, a long-range configuration would instead distribute inference
    across many genomic blocks while restricting the number of pivot pairs evaluated
    per block, for example:

    .. code-block:: python

        # Long-range pattern: many blocks, fewer pivot pairs
        blocks = [
            (start, start + 1_000_000)
            for start in range(0, 50_000_000, 1_000_000)
        ]

        pivot_pairs = [
            (i, j)
            for i in range(10)
            for j in range(i + 1, 10)
        ]

    Both behaviors are supported by the same API and arise implicitly from user-level
    choices of regions and pivot pairs. The example shown here follows the targeted
    pattern, while the human and mosquito analyses presented later use the long-range
    pattern to decode extended genomic regions.


Inference with :func:`cxt.api2.translate`
-----------------------------------------

We directly run inference on the tree sequence using multiple GPUs.
A small mutation-rate calibration step aligns the inferred TMRCAs with biological scale.

.. code-block:: python

    from cxt.api2 import translate

    devices = ["cuda:0", "cuda:1", "cuda:2"]
    B = 256   # global batch size

    yhat_tmrca, index_map = translate(
        input_data=ts,
        data_type="ts",
        model=model,
        pivot_pairs=pivot_pairs,
        blocks=blocks,
        devices=devices,
        B_per_device=B,
        B=B,
        build_workers=8,
        mutation_rate=1.29e-8,   # optional calibration
    )

Here ``yhat_tmrca`` contains log-TMRCA predictions for each pivot pair and window.

Computing true TMRCAs
---------------------

For benchmarking, we compute “true” pairwise TMRCAs directly from the simulated tree
sequence. The helper :func:`cxt.preprocess.interpolate_tmrcas` computes the coalescent
time for a given pair and interpolates it into 2 kb windows.

.. code-block:: python

    from concurrent.futures import ProcessPoolExecutor
    from cxt.preprocess import interpolate_tmrcas

    WINDOW_BP = 2000
    BLOCK_LEN = 1_000_000

    def _true(args):
        ts, a, b = args
        return interpolate_tmrcas(ts, WINDOW_BP, BLOCK_LEN, a, b)

    def build_yhats_ytrues(ts, pivot_ids, yhat_tmrca, max_workers=None):
        # Convert model output from log space to generations
        yhat_means = np.exp(yhat_tmrca)

        # Compute true TMRCAs for each pivot pair
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            ytrues = list(ex.map(_true, [(ts, a, b) for a, b in pivot_ids]))

        # Mean predicted TMRCAs across replicates
        yhats = [yhat_means.mean(0)[i] for i in range(len(pivot_ids))]

        return yhats, ytrues

    yhats, ytrues = build_yhats_ytrues(ts, pivot_pairs, yhat_tmrca, max_workers=24)

Flatten for easy plotting:

.. code-block:: python

    yhats = np.array(yhats).flatten()
    ytrues = np.array(ytrues).flatten()

Discretizing to the training time grid
--------------------------------------

To match the exact time grid used during model training (as done for the paper),
we discretize the true TMRCAs in log space onto the internal ``TIMES`` grid.

.. code-block:: python

    from cxt.utils import TIMES

    def discretize(sequence, population_time):
        idx = np.searchsorted(population_time, sequence, side="right") - 1
        idx = np.clip(idx, 0, len(population_time) - 1)
        return idx

    # Discretize true TMRCAs
    ytrues_log = np.log(ytrues)
    ytrues_idx = discretize(ytrues_log, TIMES)
    ytrues = np.exp(TIMES[ytrues_idx])

Scatter plot (as in the paper)
------------------------------

To reproduce the constant-\ :math:`N_e` benchmark panel from the manuscript,
we generate a scatter plot comparing inferred and true TMRCAs.

.. code-block:: python

    from local_utils import plot_tmrca_scatter

    ax_cxt_constant = plot_tmrca_scatter(
        yhats,
        ytrues,
        "cxt_constant.png",
        tool=r"$\mathbf{cxt}$-narrow: Constant $N_e$",
    )

The saved file ``cxt_constant.png`` corresponds directly to the
:math:`\mathbf{cxt}`-narrow constant-\ :math:`N_e` panel shown in the paper.
