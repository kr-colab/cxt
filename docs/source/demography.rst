Demography-based benchmark and coalescence rates
================================================

This example shows how to benchmark :math:`\mathbf{cxt}` on a realistic human
demographic model (Zigzag_1S14 from ``stdpopsim``) and how to convert the
resulting TMRCA distribution into coalescence-rate curves. The end result is a
coalescence-rate panel similar in spirit to the one used in the paper.

The steps are:

1. Simulate a 10 Mb human tree sequence under Zigzag_1S14.
2. Compute *discretized* true TMRCAs for a subset of pairs.
3. Run :func:`cxt.api2.translate` on the same pairs and blocks.
4. Estimate coalescence rates from both true and inferred TMRCAs.
5. Compare them to the theoretical coalescence-rate trajectory implied by the
   demographic model.

Setup and imports
-----------------

.. code-block:: python

    import os
    import json

    import numpy as np
    import tskit
    import torch
    import stdpopsim
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from cxt.api2 import translate
    from cxt.utils import setup_cxt_model, coalescence_rates
    from cxt.preprocess import interpolate_tmrcas

    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Use all available GPUs
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    # Load the cxt model (broad variant in this example)
    model = setup_cxt_model(model_type="broad")

Simulating HomSap Zigzag\_1S14 with stdpopsim
---------------------------------------------

We simulate 25 diploid individuals (``num_pairs``) under the human Zigzag\_1S14
demography for 10 Mb of chromosome 1. The resulting tree sequence is cached to
avoid re-simulation.

.. code-block:: python

    num_pairs = 25
    window_size = 2e3           # 2 kb windows
    seed = int(10e6)
    sequence_length = 10e6      # 10 Mb

    species = stdpopsim.get_species("HomSap")
    demogr = species.get_demographic_model("Zigzag_1S14")
    contig = species.get_contig("chr1", right=sequence_length)

    # First population name (e.g. "pop_0")
    population_name = demogr.populations[0].name
    sample = {population_name: num_pairs}

    engine = stdpopsim.get_engine("msprime")

    if "homsap.ts" not in os.listdir(cache_dir):
        ts = engine.simulate(
            contig=contig,
            samples=sample,
            demographic_model=demogr,
            seed=seed,
        ).trim()
        ts.dump(os.path.join(cache_dir, "homsap.ts"))
    else:
        ts = tskit.load(os.path.join(cache_dir, "homsap.ts"))

Computing true discretized TMRCAs
---------------------------------

We next compute the *true* discretized TMRCAs in 2 kb windows for all
pairwise combinations among the 25 individuals. These are cached as
``homsap_true_tmrcas.npy``:

.. code-block:: python

    if "homsap_true_tmrcas.npy" in os.listdir(cache_dir):
        true_tmrcas = np.load(os.path.join(cache_dir, "homsap_true_tmrcas.npy"))
    else:
        pivot_ids = []
        true_tmrcas = []

        for i in tqdm(range(num_pairs)):
            for j in range(i + 1, num_pairs):
                pivot_A, pivot_B = i, j
                pivot_ids.append((pivot_A, pivot_B))

                tmrca_ij = list(
                    interpolate_tmrcas(
                        ts,
                        window_size,
                        sequence_length,
                        pivot_A,
                        pivot_B,
                    )
                )
                true_tmrcas.append(tmrca_ij)

        true_tmrcas = np.array(true_tmrcas)
        np.save(
            os.path.join(cache_dir, "homsap_true_tmrcas.npy"),
            true_tmrcas,
        )

Defining blocks and pivot pairs for cxt
---------------------------------------

We analyze the 10 Mb region in 10 non-overlapping 1 Mb blocks and use the same
25-individual sample to define pivot pairs:

.. code-block:: python

    # Ten 1 Mb blocks covering 10 Mb
    blocks = []
    num_blocks = 10
    for x in np.linspace(0, num_blocks * 1e6 - 1e6, num_blocks):
        blocks.append((int(x), int(x + 1e6)))

    pivot_pairs = []
    for i in tqdm(range(num_pairs)):
        for j in range(i + 1, num_pairs):
            pivot_A, pivot_B = i, j
            pivot_pairs.append((pivot_A, pivot_B))

Running cxt on the simulated tree sequence
------------------------------------------

We now run :func:`cxt.api2.translate` on the tree sequence to obtain log-TMRCA
predictions in generations. The result is cached as ``tmrca_homsap.npz`` and
``index_map_homsap.npz``:

.. code-block:: python

    if (
        "tmrca_homsap.npz" in os.listdir(cache_dir)
        and "index_map_homsap.npz" in os.listdir(cache_dir)
    ):
        tmrca = np.load(os.path.join(cache_dir, "tmrca_homsap.npz"))["tmrca"]
        index_map = np.load(os.path.join(cache_dir, "index_map_homsap.npz"))["index_map"]
    else:
        tmrca, index_map = translate(
            input_data=ts,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            B_per_device=128,
            B=128,
            devices=devices,
            build_workers=32,
            use_fast_process_per_gpu=True,
            mutation_rate=1.29e-8,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "tmrca_homsap.npz"),
            tmrca=tmrca,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "index_map_homsap.npz"),
            index_map=index_map,
        )

Estimating coalescence rates
----------------------------

We now convert the windowed TMRCAs into coalescence-rate curves. We use:

* ``true_tmrcas``: discretized TMRCAs computed directly from the tree sequence.
* ``tmrca``: :math:`\mathbf{cxt}` predictions (log time in generations).
* The demographic model itself to obtain an “oracle” coalescence-rate trajectory
  :math:`\lambda(t)` over a fine time grid.

First, define time windows and compute the theoretical coalescence-rate
trajectory from the demography:

.. code-block:: python

    num_time_windows = 40
    max_log_time = np.floor(np.log10(ts.max_time))

    # Time windows for the empirical coalescence-rate estimates
    time_windows = np.logspace(2, max_log_time, num_time_windows + 1)
    time_windows[0] = 0.0

    # Fine grid for the model-based coalescence-rate trajectory
    fine_time_grid = np.logspace(2, max_log_time, 1000)

    coalrate_ck, _ = demogr.model.debug().coalescence_rate_trajectory(
        lineages={population_name: 2},
        steps=fine_time_grid,
    )

Next, flatten predictions and ground truth and pass them through
:func:`cxt.utils.coalescence_rates`:

.. code-block:: python

    # cxt outputs log time in generations
    tmrca_flat = np.exp(tmrca.flatten())

    # true_tmrcas is already in generations (discretized per window)
    true_tmrcas_flat = true_tmrcas.flatten()

    # Estimated coalescence rates from cxt and from the "true" TMRCAs
    yhat_coalrate = coalescence_rates(tmrca_flat, time_windows)
    ytrue_coalrate = coalescence_rates(true_tmrcas_flat, time_windows)

Plotting coalescence-rate curves
--------------------------------

Finally, we plot the inferred and true coalescence-rate curves together with
the theoretical trajectory implied by the demographic model. With suitable
styling, this yields a panel similar to the coalescence-rate figure in the
paper.

.. code-block:: python

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.step(
        time_windows[:-1],
        yhat_coalrate,
        where="post",
        label=r"$\mathbf{cxt}$",
        color="dodgerblue",
    )
    ax.step(
        time_windows[:-1],
        ytrue_coalrate,
        where="post",
        label="inference-limit (discrete)",
        color="firebrick",
    )
    ax.plot(
        fine_time_grid,
        coalrate_ck,
        "-",
        color="black",
        label="demographic expectation",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (generations)")
    ax.set_ylabel("Coalescence rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        r"$\mathit{H. sapiens}$ Zigzag\_1S14",
        loc="left",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("cxt_homsap_zigzag_coalescence_rates.png", dpi=200)

This produces a coalescence-rate figure in which the :math:`\mathbf{cxt}` curve
can be directly compared to both the discrete “inference-limit” curve from the
true simulated TMRCAs and the continuous expectation implied by the Zigzag\_1S14
demographic model.
