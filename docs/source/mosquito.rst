Mosquito 2L inversion and RDL sweep
===================================

This example shows how :math:`\mathbf{cxt}` is applied to real *Anopheles gambiae*
data on chromosome 2L to produce the mosquito figures in the paper:

* Genome-wide TMRCA patterns across the 2La inversion for multiple populations.
* A zoomed-in view of the RDL insecticide-resistance region with missingness tracks.

The pipeline is:

1. Load and subset Ag1000G 2L tree sequences per population and inversion genotype.
2. Load the accessibility bitmask and derive a missingness track.
3. Run :func:`cxt.api2.translate` with a missingness-aware model variant.
4. Aggregate TMRCAs per genome and per population.
5. Plot genome-wide 2L inversion TMRCA panels.
6. Plot a focused RDL-region panel with aligned RDL coordinates and missingness.

Paths, imports, and cache
-------------------------

.. code-block:: python

    import os
    import json

    import tszip
    import tskit
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from tqdm import tqdm

    import stdpopsim
    import torch

    from cxt.api2 import translate
    from cxt.utils import setup_cxt_model
    from cxt.preprocess import interpolate_tmrcas

    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Ag1000G 2L dated tree sequence
    data_path = (
        "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/"
        "tsinfer_data_v2/gamb.2L.gff.dated.ne.trees"
    )

    # Accessibility bitmask (Singer / Ag3)
    accessible_path = (
        "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/"
        "agp3.is_accessible.txt.npz"
    )

Loading and subsetting tree sequences per population
----------------------------------------------------

We subset the full 2L tree sequence to heterozygous 2La individuals from each
population and cache one tree sequence per population:

.. code-block:: python

    def load_population_ts(pop_name, n=25):
        """
        Load and simplify the 2L tree sequence to `n` heterozygous 2La
        individuals from the specified country, caching to disk.
        """
        ts_path = os.path.join(cache_dir, f"ts_{pop_name.lower().replace(' ', '_')}.trees")
        if ts_path in [os.path.join(cache_dir, f) for f in os.listdir(cache_dir)]:
            return tskit.load(ts_path)

        ts = tszip.load(data_path)
        ids = []
        for ind in ts.individuals():
            meta = json.loads(ind.metadata.decode("ascii"))
            country = meta["country"]
            inv_geno = int(meta["2La"])
            if country == pop_name and inv_geno == 1:
                ids.append(ind.nodes)

        ts_pop = ts.simplify(samples=np.concatenate(ids[:n]))
        ts_pop.dump(ts_path)
        return ts_pop

    # Example populations used in the figure
    ts_burkina_faso = load_population_ts("Burkina Faso", n=25)
    ts_cameroon     = load_population_ts("Cameroon",     n=25)
    ts_mali         = load_population_ts("Mali",         n=25)
    ts_uganda       = load_population_ts("Uganda",       n=25)

Loading the accessibility mask and missingness bitmask
------------------------------------------------------

The Ag1000G “accessibility” bitmask is used to provide a missingness track that
is passed to :func:`translate` and later plotted beneath the TMRCA panels:

.. code-block:: python

    bitmask = np.load(accessible_path)["access_2L"]   # True = accessible
    unaccessible_bitmask = ~bitmask                   # True = missing

Setting up the cxt model and pivots
-----------------------------------

For the mosquito analysis we use a missingness-aware model variant
(``w200_wmissing``) and treat each individual as a pivot pair defined by its
two nodes in the tree sequence:

.. code-block:: python

    # Missingness-aware model for mosquitoes
    model = setup_cxt_model(model_type="w200_wmissing")

    # Use the same pivot construction across populations
    pivot_pairs = [tuple(ind.nodes) for ind in ts_burkina_faso.individuals()]

    # Default mutation rate for Ag3 mosquitoes
    mutation_rate = 3.5e-9

    # GPU configuration
    devices = ["cuda:0", "cuda:1", "cuda:2"]

Genome-wide blocks along 2L
---------------------------

For the genome-wide inversion panel we tile almost the entire 2L arm in 100 kb
blocks:

.. code-block:: python

    blocks = [(i * 0.1e6, (i + 1) * 0.1e6) for i in range(490)]

Running cxt for Burkina Faso
----------------------------

We first run :func:`translate` on Burkina Faso with missingness, caching the
results:

.. code-block:: python

    if "tmrca_burkina_faso_wmissing_2L.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "tmrca_burkina_faso_wmissing_2L.npz"))
        tmrca_burkina_faso = data["tmrca"]
        index_map = data["index_map"]
    else:
        tmrca_burkina_faso, index_map = translate(
            input_data=ts_burkina_faso,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            missingness_bitmask=unaccessible_bitmask,
            devices=devices,
            B_per_device=512,
            build_workers=36,
            mutation_rate=mutation_rate,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "tmrca_burkina_faso_wmissing_2L.npz"),
            tmrca=tmrca_burkina_faso,
            index_map=index_map,
        )

Running cxt for Cameroon
------------------------

For Cameroon we compute both a focused RDL-region run and a full 2L run. The
figure uses the full 2L run; the RDL run is useful for local consistency checks.

.. code-block:: python

    ts_cameroon = load_population_ts("Cameroon", n=25)
    devices = ["cuda:1", "cuda:2"]

    # Focused RDL-region blocks (optional)
    rdl_blocks = [
        (25_200_000, 25_300_000),
        (25_300_000, 25_400_000),
        (25_400_000, 25_500_000),
    ]

    if "tmrca_cameroon_wmissing_2L_rdl.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "tmrca_cameroon_wmissing_2L_rdl.npz"))
        tmrca_cameroon_rdl = data["tmrca"]
        index_map_rdl = data["index_map"]
    else:
        tmrca_cameroon_rdl, index_map_rdl = translate(
            input_data=ts_cameroon,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=rdl_blocks,
            missingness_bitmask=unaccessible_bitmask,
            devices=devices,
            B_per_device=512,
            build_workers=36,
            mutation_rate=mutation_rate,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "tmrca_cameroon_wmissing_2L_rdl.npz"),
            tmrca=tmrca_cameroon_rdl,
            index_map=index_map_rdl,
        )

    # Full 2L run for Cameroon (used for genome-wide inversion panel)
    blocks = [(i * 0.1e6, (i + 1) * 0.1e6) for i in range(490)]

    if "tmrca_cameroon_wmissing_2L.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "tmrca_cameroon_wmissing_2L.npz"))
        tmrca_cameroon = data["tmrca"]
        index_map = data["index_map"]
    else:
        tmrca_cameroon, index_map = translate(
            input_data=ts_cameroon,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            missingness_bitmask=unaccessible_bitmask,
            devices=devices,
            B_per_device=512,
            build_workers=36,
            mutation_rate=mutation_rate,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "tmrca_cameroon_wmissing_2L.npz"),
            tmrca=tmrca_cameroon,
            index_map=index_map,
        )

Running cxt for Mali and Uganda
-------------------------------

We repeat the full 2L run for Mali and Uganda using the same ``blocks`` and
``pivot_pairs``:

.. code-block:: python

    ts_mali = load_population_ts("Mali", n=25)

    if "tmrca_mali_wmissing_2L.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "tmrca_mali_wmissing_2L.npz"))
        tmrca_mali = data["tmrca"]
        index_map = data["index_map"]
    else:
        tmrca_mali, index_map = translate(
            input_data=ts_mali,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            missingness_bitmask=unaccessible_bitmask,
            devices=devices,
            B_per_device=512,
            build_workers=36,
            mutation_rate=mutation_rate,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "tmrca_mali_wmissing_2L.npz"),
            tmrca=tmrca_mali,
            index_map=index_map,
        )

    ts_uganda = load_population_ts("Uganda", n=25)

    if "tmrca_uganda_wmissing_2L.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "tmrca_uganda_wmissing_2L.npz"))
        tmrca_uganda = data["tmrca"]
        index_map = data["index_map"]
    else:
        tmrca_uganda, index_map = translate(
            input_data=ts_uganda,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            missingness_bitmask=unaccessible_bitmask,
            devices=devices,
            B_per_device=512,
            build_workers=36,
            mutation_rate=mutation_rate,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "tmrca_uganda_wmissing_2L.npz"),
            tmrca=tmrca_uganda,
            index_map=index_map,
        )

Aggregating TMRCAs into genome-wide windows
-------------------------------------------

We aggregate per-country TMRCAs over blocks and pivot pairs into genome-wide
2 kb windows and log-transform them. We use a helper that collapses replicate
and pivot dimensions onto a `(replicates, windows)` array:

.. code-block:: python

    def make_tmrca_genome(tmrca_country, pivot_pairs, index_map):
        tmrca_genomes = []
        for i in range(len(pivot_pairs)):
            mask = index_map[:, 1] == i
            tmrca_genome = tmrca_country[:, mask, :].mean(0).flatten()
            tmrca_genomes.append(tmrca_genome)
        tmrca_genomes = np.array(tmrca_genomes)
        return np.exp(tmrca_genomes)  # (replicates, windows) in generations

    tmrca_uganda        = make_tmrca_genome(tmrca_uganda,        pivot_pairs, index_map)
    tmrca_cameroon      = make_tmrca_genome(tmrca_cameroon,      pivot_pairs, index_map)
    tmrca_mali          = make_tmrca_genome(tmrca_mali,          pivot_pairs, index_map)
    tmrca_burkina_faso  = make_tmrca_genome(tmrca_burkina_faso,  pivot_pairs, index_map)

    # Log-transform for plotting
    tmrca_uganda        = np.log(tmrca_uganda)
    tmrca_cameroon      = np.log(tmrca_cameroon)
    tmrca_mali          = np.log(tmrca_mali)
    tmrca_burkina_faso  = np.log(tmrca_burkina_faso)

Loading Ghana with adapter model
--------------------------------

Ghana is treated separately using a model with an adapter. We also downsample
to 10 individuals and pass the adapter explicitly into :func:`translate`:

.. code-block:: python

    # Ghana tree sequence (heterozygous 2La)
    if "ts_ghana.trees" not in os.listdir(cache_dir):
        ts = tszip.load(data_path)
        ids = []
        for ind in ts.individuals():
            meta = json.loads(ind.metadata.decode("ascii"))
            country = meta["country"]
            inv_geno = int(meta["2La"])
            if country == "Ghana" and inv_geno == 1:
                ids.append(ind.nodes)
        ts_ghana = ts.simplify(samples=np.concatenate(ids[:25]))
        ts_ghana.dump(os.path.join(cache_dir, "ts_ghana.trees"))
    else:
        ts_ghana = tskit.load(os.path.join(cache_dir, "ts_ghana.trees"))

    # Ghana model: adapter variant, do not mask singletons
    model_ghana = setup_cxt_model(model_type="w200_wmissing_adapter")
    model_ghana.backbone.transformer.bt2ls.config.mask_singletons = False

    # Downsample Ghana to 10 samples
    ts_ghana = ts_ghana.simplify(samples=np.arange(0, 10))
    pivot_pairs_ghana = [tuple(ind.nodes) for ind in ts_ghana.individuals()]

    mutation_rate = 3.5e-9
    devices = ["cuda:1"]

    if "tmrca_ghana_wmissing.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "tmrca_ghana_wmissing.npz"))
        tmrca_ghana = data["tmrca"]
        index_map_ghana = data["index_map"]
    else:
        tmrca_ghana, index_map_ghana = translate(
            input_data=ts_ghana,
            data_type="ts",
            model=model_ghana.backbone,
            pivot_pairs=pivot_pairs_ghana,
            blocks=blocks,
            missingness_bitmask=unaccessible_bitmask,
            devices=devices,
            B_per_device=512,
            build_workers=36,
            mutation_rate=mutation_rate,
            adapter=model_ghana.adapter,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "tmrca_ghana_wmissing.npz"),
            tmrca=tmrca_ghana,
            index_map=index_map_ghana,
        )

    tmrca_ghana = make_tmrca_genome(tmrca_ghana, pivot_pairs_ghana, index_map_ghana)
    tmrca_ghana = np.log(tmrca_ghana)

Genome-wide 2L inversion panel
------------------------------

We first produce a genome-wide 2L inversion panel (2 × 2 layout) with a shared
missingness track at the bottom of each panel.

Helper functions for missingness and panel plotting:

.. code-block:: python

    fmt_mb = FuncFormatter(lambda v, _: f"{v/1e6:.1f} Mb")

    def moving_average(frac, smooth_bp=50_000, step_bp=2000):
        k = max(1, int(round(smooth_bp / step_bp)))
        kernel = np.ones(k, dtype=np.float32) / k
        return np.convolve(frac.astype(np.float32), kernel, mode="same")

    def missing_track_from_bitmask(unaccessible_bitmask, blocks, window_bp=200):
        start, end = blocks[0][0], blocks[-1][1]
        region = np.asarray(unaccessible_bitmask[start:end], dtype=np.bool_)
        n_bins = int(np.ceil((end - start) / window_bp))
        pad = n_bins * window_bp - region.size
        if pad > 0:
            region = np.pad(region, (0, pad), constant_values=False)
        miss_frac = region.reshape(n_bins, window_bp).mean(axis=1).astype(np.float32)
        return miss_frac

    def _draw_missing_track(ax, x, missing_frac, smooth_bp=50_000, step_bp=200,
                            height=0.16, pad=0.26, xlabel=None):
        """Draw compact missingness track below main axis; x-label shown here (if provided)."""
        y_raw = np.clip(missing_frac, 0, 1)
        y_smooth = np.clip(moving_average(y_raw, smooth_bp, step_bp), 0, 1)

        tr = ax.inset_axes([0.0, -pad, 1.0, height], transform=ax.transAxes)
        tr.fill_between(x, 0, y_smooth, alpha=0.45, color="lightsteelblue", zorder=1)
        tr.plot(x, y_smooth, lw=1.5, color="steelblue", alpha=0.95, zorder=2)

        tr.set_xlim(x.min(), x.max())
        tr.set_ylim(0, 1)
        tr.xaxis.set_major_formatter(fmt_mb)
        for s in ("top", "right", "left"):
            tr.spines[s].set_visible(False)
        tr.spines["bottom"].set_linewidth(0.5)
        tr.spines["bottom"].set_alpha(0.5)
        tr.set_yticks([])
        tr.set_facecolor("none")
        tr.grid(False)

        if xlabel is None:
            tr.tick_params(axis="x", which="both", labelbottom=False)
        else:
            tr.set_xlabel(xlabel, labelpad=10, fontsize=9)

        for line in tr.lines:
            line.set_clip_on(False)
        for coll in tr.collections:
            coll.set_clip_on(False)

    def plot_panel(ax, tmrca_genome, title, missing_frac=None,
                   smooth_bp=5_000, step_bp=200, xlabel=None):
        mean_tmrca = tmrca_genome.mean(axis=0)
        std_tmrca  = tmrca_genome.std(axis=0)
        x = np.arange(tmrca_genome.shape[1]) * step_bp

        # 2La inversion and RDL coordinates
        inversion_start, inversion_end = 21e6, 43.4e6
        rdl_start, rdl_end = 25_363_652, 25_434_556
        inversion_mask = (x >= inversion_start) & (x <= inversion_end)

        # TMRCA replicates
        n = tmrca_genome.shape[0]
        for i in range(n):
            alpha = 0.30 + 0.40 * (i / n)
            lw = 0.6 + 0.4 * (i % 5 == 0)
            ax.plot(x, tmrca_genome[i], lw=lw, color="lightgray", alpha=alpha*0.8, zorder=0)
            ax.plot(x[inversion_mask], tmrca_genome[i][inversion_mask],
                    lw=lw, color="steelblue", alpha=alpha, zorder=0)

        eps   = 1e-12
        lower = np.clip(mean_tmrca - std_tmrca, eps, None)
        upper = np.clip(mean_tmrca + std_tmrca, eps, None)

        ax.fill_between(x, lower, upper, color="lightgray", alpha=0.2, zorder=1)
        ax.plot(x, lower, ls="--", lw=0.8, color="silver", alpha=0.7, zorder=2)
        ax.plot(x, upper, ls="--", lw=0.8, color="silver", alpha=0.7, zorder=2)
        ax.plot(x, mean_tmrca, lw=1.0, color="darkgray", alpha=0.8, zorder=3)

        ax.fill_between(x[inversion_mask], lower[inversion_mask], upper[inversion_mask],
                        color="deepskyblue", alpha=0.35, zorder=1)
        ax.plot(x[inversion_mask], lower[inversion_mask],
                ls="--", lw=0.8, color="dodgerblue", alpha=0.9, zorder=2)
        ax.plot(x[inversion_mask], upper[inversion_mask],
                ls="--", lw=0.8, color="dodgerblue", alpha=0.9, zorder=2)
        ax.plot(x[inversion_mask], mean_tmrca[inversion_mask],
                lw=1.0, color="dodgerblue", zorder=3, label="2La inversion")

        ax.set_yscale("log")
        ax.set_xlim(x.min(), x.max())
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_formatter(fmt_mb)
        ax.set_ylim(1e4, 5e6)
        ax.legend(fontsize=8, loc="best")
        ax.tick_params(axis="x", which="both", labelbottom=False)

        if missing_frac is not None:
            _draw_missing_track(ax, x, missing_frac,
                                smooth_bp=50_000, step_bp=step_bp, xlabel=xlabel)

Compute missingness and plot the inversion panel:

.. code-block:: python

    blocks = [(int(i * 0.1e6), int((i + 1) * 0.1e6)) for i in range(490)]
    missing_2L = missing_track_from_bitmask(unaccessible_bitmask, blocks, window_bp=200)

    start = 0
    end   = blocks[-1][1]
    start_block = start // 200
    end_block   = end // 200

    datasets = [
        ("Mali",         np.exp(tmrca_mali)[:, start_block:end_block],         missing_2L),
        ("Burkina Faso", np.exp(tmrca_burkina_faso)[:, start_block:end_block], missing_2L),
        ("Cameroon",     np.exp(tmrca_cameroon)[:, start_block:end_block],     missing_2L),
        ("Uganda",       np.exp(tmrca_uganda)[:, start_block:end_block],       missing_2L),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for i, (ax, (name, genome, miss)) in enumerate(zip(axes.ravel(), datasets)):
        xlabel = "Position on 2L (bp)" if i >= 2 else None
        plot_panel(ax, genome, name, missing_frac=miss,
                   smooth_bp=5_000, step_bp=200, xlabel=xlabel)

    for ax in axes[:, 0]:
        ax.set_ylabel("TMRCA (generations)")

    plt.tight_layout()
    plt.show()

RDL zoom panel with aligned coordinates
---------------------------------------

To focus on the RDL insecticide-resistance gene, we restrict to the region
[25.1 Mb, 25.6 Mb] on 2L, align all populations to the same genomic coordinates,
and overlay a missingness track only in the bottom panel.

Helper functions and RDL settings:

.. code-block:: python

    REGION_START = 25_100_000
    REGION_END   = 25_600_000
    STEP_BP      = 200

    rdl_start, rdl_end = 25_363_652, 25_434_556
    fmt_mb = FuncFormatter(lambda v, _: f"{v/1e6:.1f} Mb")

    def moving_average(frac, smooth_bp=5_000, step_bp=200):
        k = max(1, int(round(smooth_bp / step_bp)))
        kernel = np.ones(k, dtype=np.float32) / k
        return np.convolve(frac.astype(np.float32), kernel, mode="same")

    def missing_track_from_bitmask(unaccessible_bitmask, blocks, window_bp=200):
        start, end = blocks[0][0], blocks[-1][1]
        region = np.asarray(unaccessible_bitmask[start:end], dtype=np.bool_)
        n_bins = int(np.ceil((end - start) / window_bp))
        pad = n_bins * window_bp - region.size
        if pad > 0:
            region = np.pad(region, (0, pad), constant_values=False)
        miss_frac = region.reshape(n_bins, window_bp).mean(axis=1).astype(np.float32)
        return miss_frac

    def _draw_missing_track(ax, x, missing_frac, smooth_bp=5_000, step_bp=200,
                            height=0.16, pad=0.26, xlabel=None):
        y_raw = np.clip(missing_frac, 0, 1)
        y_smooth = np.clip(moving_average(y_raw, smooth_bp, step_bp), 0, 1)

        tr = ax.inset_axes([0.0, -pad, 1.0, height], transform=ax.transAxes)
        tr.fill_between(x, 0, y_smooth, alpha=0.35, color="lightsteelblue", zorder=1)
        tr.plot(x, y_smooth, lw=1.1, color="steelblue", alpha=0.95, zorder=2)
        tr.plot(x, y_raw,   lw=0.5, color="steelblue", alpha=0.30, zorder=2)

        tr.set_xlim(x.min(), x.max())
        tr.set_ylim(0, 1)
        tr.xaxis.set_major_formatter(fmt_mb)
        for s in ("top", "right", "left"):
            tr.spines[s].set_visible(False)
        tr.spines["bottom"].set_linewidth(0.5)
        tr.spines["bottom"].set_alpha(0.5)
        tr.set_yticks([])
        tr.set_facecolor("none")
        tr.grid(False)

        if xlabel is None:
            tr.tick_params(axis="x", which="both", labelbottom=False)
        else:
            tr.set_xlabel(xlabel, labelpad=10, fontsize=9)

        for line in tr.lines:
            line.set_clip_on(False)
        for coll in tr.collections:
            coll.set_clip_on(False)

    def plot_panel(ax, tmrca_genome, title, base_x,
                   missing_frac=None, smooth_bp=5_000, step_bp=200, xlabel=None):
        """
        tmrca_genome: (n_reps, n_bins) slice corresponding to [REGION_START, REGION_END]
        base_x:       (n_bins,) array of bp coordinates (true genomic positions)
        """
        mean_tmrca = tmrca_genome.mean(axis=0)
        std_tmrca  = tmrca_genome.std(axis=0)

        n = tmrca_genome.shape[0]
        for i in range(n):
            alpha = 0.30 + 0.40 * (i / n)
            lw    = 0.6 + 0.4 * (i % 5 == 0)
            ax.plot(base_x, tmrca_genome[i], lw=lw,
                    color="steelblue", alpha=alpha, zorder=0)

        eps   = 1e-12
        lower = np.clip(mean_tmrca - std_tmrca, eps, None)
        upper = np.clip(mean_tmrca + std_tmrca, eps, None)

        ax.fill_between(base_x, lower, upper,
                        color="deepskyblue", alpha=0.35, zorder=1)
        ax.plot(base_x, lower, ls="--", lw=0.8,
                color="dodgerblue", alpha=0.9, zorder=2)
        ax.plot(base_x, upper, ls="--", lw=0.8,
                color="dodgerblue", alpha=0.9, zorder=2)
        ax.plot(base_x, mean_tmrca, lw=1.0, color="dodgerblue", zorder=3)

        # RDL shading + arrow
        ax.axvspan(rdl_start, rdl_end, color="crimson", alpha=0.10, zorder=0)
        mid = (rdl_start + rdl_end) / 2

        ymin, ymax = np.nanmin(mean_tmrca), np.nanmax(mean_tmrca)
        y_arrow = np.exp((np.log(ymin) + np.log(ymax)) / 3)

        ax.annotate(
            "",
            xy=(rdl_end, y_arrow),
            xytext=(rdl_start, y_arrow),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
            zorder=6,
        )
        ax.text(
            mid,
            y_arrow * 1.05,
            "RDL",
            fontsize=9,
            ha="center",
            va="bottom",
            zorder=6,
        )

        ax.set_yscale("log")
        ax.set_xlim(REGION_START, REGION_END)
        ax.set_ylim(0.5e2, 5e6)
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        ax.set_title(title, fontsize=11, loc="left")
        ax.xaxis.set_major_formatter(fmt_mb)
        ax.tick_params(axis="x", which="both", labelbottom=False)

        if missing_frac is not None:
            _draw_missing_track(
                ax, base_x, missing_frac,
                smooth_bp=smooth_bp,
                step_bp=step_bp,
                xlabel=xlabel,
            )

Build the region slice and plot the RDL zoom panel:

.. code-block:: python

    blocks_rdl = [
        (25_100_000, 25_200_000),
        (25_200_000, 25_300_000),
        (25_300_000, 25_400_000),
        (25_400_000, 25_500_000),
        (25_500_000, 25_600_000),
    ]
    missing_2L_rdl = missing_track_from_bitmask(unaccessible_bitmask, blocks_rdl, window_bp=STEP_BP)

    start_block = REGION_START // STEP_BP
    n_bins_region = (REGION_END - REGION_START) // STEP_BP
    base_x_region = np.arange(n_bins_region) * STEP_BP + REGION_START

    datasets = [
        ("Mali",         np.exp(tmrca_mali)),
        ("Burkina Faso", np.exp(tmrca_burkina_faso)),
        ("Cameroon",     np.exp(tmrca_cameroon)),
        ("Ghana",        np.exp(tmrca_ghana)),
        ("Uganda",       np.exp(tmrca_uganda)),
    ]

    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 9), sharex=True, sharey=True)
    if n_rows == 1:
        axes = [axes]

    for i, (ax, (name, genome_full)) in enumerate(zip(axes, datasets)):
        pop_start_block = start_block
        pop_end_block   = pop_start_block + n_bins_region

        n_bins_chr = genome_full.shape[1]
        if pop_end_block > n_bins_chr:
            pop_end_block = n_bins_chr
            pop_start_block = pop_end_block - n_bins_region

        tmrca_slice = genome_full[:, pop_start_block:pop_end_block]

        if i == n_rows - 1:
            xlabel = "Position on 2L (Mb)"
            miss   = missing_2L_rdl
        else:
            xlabel = None
            miss   = None

        plot_panel(
            ax,
            tmrca_slice,
            name,
            base_x=base_x_region,
            missing_frac=miss,
            smooth_bp=5_000,
            step_bp=STEP_BP,
            xlabel=xlabel,
        )

    for ax in axes:
        ax.set_ylabel("TMRCA (generations)")

    plt.tight_layout()
    plt.show()

With appropriate global figure layout and styling, these calls reproduce the
mosquito inversion and RDL panels used in the manuscript, including the
population-wise TMRCA patterns and the accessibility-driven missingness track
in the bottom panel.
