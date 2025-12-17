Human 1000 Genomes: LCT and HLA examples
========================================

This example shows how :math:`\mathbf{cxt}` is applied to GBR samples from the
1000 Genomes dataset to produce the LCT (chr2) and HLA (chr6) panels used in
the manuscript. The pipeline is:

1. Load tsinfer-based tree sequences for chr2p/chr2q and chr6p/chr6q.
2. Run :func:`cxt.api2.translate` in 1 Mb blocks on pivot pairs.
3. Estimate per-window missingness from bcftools masks and apply a
   post-hoc diversity–based correction.
4. Stitch p and q arms into whole-chromosome TMRCA tracks.
5. Plot chromosome-wide curves and zoom in on LCT and HLA with gene annotations.

Note that, unlike the mosquito analysis, we **do not** use a missingness-aware
model variant here. Instead, we apply a *post-hoc stochastic diversity bias
correction* per block, which is sufficient for the more moderate human
missingness patterns in these regions.

Model and devices
-----------------

.. code-block:: python

    import torch
    from cxt.api2 import translate
    from cxt.utils import setup_cxt_model

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    model = setup_cxt_model(model_type="broad")

    import os
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)

Loading GBR tree sequences (chr2 and chr6)
------------------------------------------

We load precomputed tsinfer-based trees, simplify to 50 haploid samples, and
cache them in ``./cache`` using ``tszip``:

.. code-block:: python

    import tszip
    import tskit
    import numpy as np

    # ---- Chr6 p arm ----
    if "gbr_chr6p.ts" not in os.listdir(cache_dir):
        path = "/sietch_colab/data_share/hg1kg/tsinfer-trees/working/"
        chrom = "chr6p"
        ts_chr6p = tszip.load(f"{path}/{chrom}.tsz")
        ts_chr6p = ts_chr6p.simplify(samples=np.arange(50))
        tszip.compress(ts_chr6p, os.path.join(cache_dir, "gbr_chr6p.tsz"))
    else:
        ts_chr6p = tszip.load(os.path.join(cache_dir, "gbr_chr6p.tsz"))

    print(f"num_samples: {ts_chr6p.num_samples}; sequence_length: {ts_chr6p.sequence_length}")

    # ---- Chr6 q arm ----
    if "gbr_chr6q.ts" not in os.listdir(cache_dir):
        path = "/sietch_colab/data_share/hg1kg/tsinfer-trees/working/"
        chrom = "chr6q"
        ts_chr6q = tszip.load(f"{path}/{chrom}.tsz")
        ts_chr6q = ts_chr6q.simplify(samples=np.arange(50))
        tszip.compress(ts_chr6q, os.path.join(cache_dir, "gbr_chr6q.tsz"))
    else:
        ts_chr6q = tszip.load(os.path.join(cache_dir, "gbr_chr6q.tsz"))

    print(f"num_samples: {ts_chr6q.num_samples}; sequence_length: {ts_chr6q.sequence_length}")

    # ---- Chr2 p arm ----
    if "gbr_chr2p.ts" not in os.listdir(cache_dir):
        path = "/sietch_colab/data_share/hg1kg/tsinfer-trees/working/"
        chrom = "chr2p"
        ts_chr2p = tszip.load(f"{path}/{chrom}.tsz")
        ts_chr2p = ts_chr2p.simplify(samples=np.arange(50))
        tszip.compress(ts_chr2p, os.path.join(cache_dir, "gbr_chr2p.tsz"))
    else:
        ts_chr2p = tszip.load(os.path.join(cache_dir, "gbr_chr2p.tsz"))

    print(f"num_samples: {ts_chr2p.num_samples}; sequence_length: {ts_chr2p.sequence_length}")

    # ---- Chr2 q arm ----
    if "gbr_chr2q.ts" not in os.listdir(cache_dir):
        path = "/sietch_colab/data_share/hg1kg/tsinfer-trees/working/"
        chrom = "chr2q"
        ts_chr2q = tszip.load(f"{path}/{chrom}.tsz")
        ts_chr2q = ts_chr2q.simplify(samples=np.arange(50))
        tszip.compress(ts_chr2q, os.path.join(cache_dir, "gbr_chr2q.tsz"))
    else:
        ts_chr2q = tszip.load(os.path.join(cache_dir, "gbr_chr2q.tsz"))

    print(f"num_samples: {ts_chr2q.num_samples}; sequence_length: {ts_chr2q.sequence_length}")

Running cxt on 1 Mb blocks
--------------------------

We tile each arm into 1 Mb blocks and construct pivot pairs as consecutive
indices (0–1, 2–3, …, 48–49). We run :func:`translate` separately for each arm,
without specifying a mutation rate; a correction is applied later.

.. code-block:: python

    import numpy as np

    # ---- helper to build blocks + pivots for a given ts ----
    def build_blocks_and_pivots(ts, block_size=1e6, n_samples=50):
        print(f"num_samples: {ts.num_samples}; sequence_length: {ts.sequence_length}")
        num_blocks = int(ts.sequence_length // block_size)
        blocks = []
        for i in np.linspace(0, num_blocks * block_size - block_size, num_blocks):
            blocks.append((int(i), int(i + block_size)))
        pivot_pairs = [(i, i + 1) for i in range(0, n_samples, 2)]
        return blocks, pivot_pairs

    # ---- chr6p ----
    blocks, pivot_pairs = build_blocks_and_pivots(ts_chr6p, block_size=1e6, n_samples=50)

    mutation_rate = None  # correction applied later with missing data in mind

    if "gbr_chr6p.npz" in os.listdir(cache_dir):
        tmrca_chr6p = np.load(os.path.join(cache_dir, "gbr_chr6p.npz"))["tmrca"]
        index_map_chr6p = np.load(os.path.join(cache_dir, "gbr_chr6p.npz"))["index_map"]
    else:
        tmrca_chr6p, index_map_chr6p = translate(
            input_data=ts_chr6p,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            B_per_device=128,
            B=128,
            devices=devices,
            build_workers=32,
            use_fast_process_per_gpu=True,
            mutation_rate=None,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "gbr_chr6p.npz"),
            tmrca=tmrca_chr6p,
            index_map=index_map_chr6p,
        )

    # ---- chr6q (reuse blocks + pivots from chr6p length if desired) ----
    if "gbr_chr6q.npz" in os.listdir(cache_dir):
        tmrca_chr6q = np.load(os.path.join(cache_dir, "gbr_chr6q.npz"))["tmrca"]
        index_map_chr6q = np.load(os.path.join(cache_dir, "gbr_chr6q.npz"))["index_map"]
    else:
        tmrca_chr6q, index_map_chr6q = translate(
            input_data=ts_chr6q,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            B_per_device=128,
            B=128,
            devices=devices,
            build_workers=32,
            use_fast_process_per_gpu=True,
            mutation_rate=None,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "gbr_chr6q.npz"),
            tmrca=tmrca_chr6q,
            index_map=index_map_chr6q,
        )

    # ---- chr2p ----
    blocks, pivot_pairs = build_blocks_and_pivots(ts_chr2p, block_size=1e6, n_samples=50)

    if "gbr_chr2p.npz" in os.listdir(cache_dir):
        tmrca_chr2p = np.load(os.path.join(cache_dir, "gbr_chr2p.npz"))["tmrca"]
        index_map_chr2p = np.load(os.path.join(cache_dir, "gbr_chr2p.npz"))["index_map"]
    else:
        tmrca_chr2p, index_map_chr2p = translate(
            input_data=ts_chr2p,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            B_per_device=128,
            B=128,
            devices=devices,
            build_workers=32,
            use_fast_process_per_gpu=True,
            mutation_rate=None,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "gbr_chr2p.npz"),
            tmrca=tmrca_chr2p,
            index_map=index_map_chr2p,
        )

    # ---- chr2q ----
    if "gbr_chr2q.npz" in os.listdir(cache_dir):
        tmrca_chr2q = np.load(os.path.join(cache_dir, "gbr_chr2q.npz"))["tmrca"]
        index_map_chr2q = np.load(os.path.join(cache_dir, "gbr_chr2q.npz"))["index_map"]
    else:
        tmrca_chr2q, index_map_chr2q = translate(
            input_data=ts_chr2q,
            data_type="ts",
            model=model,
            pivot_pairs=pivot_pairs,
            blocks=blocks,
            B_per_device=128,
            B=128,
            devices=devices,
            build_workers=32,
            use_fast_process_per_gpu=True,
            mutation_rate=None,
        )
        np.savez_compressed(
            os.path.join(cache_dir, "gbr_chr2q.npz"),
            tmrca=tmrca_chr2q,
            index_map=index_map_chr2q,
        )

Missingness grids and post-hoc diversity correction
---------------------------------------------------

Instead of training a model with explicit missingness, we estimate the fraction
of available sites per 1 Mb window from bcftools masks and use a
post-hoc stochastic diversity-based correction. This downweights blocks with
heavy masking by scaling the effective mutation rate.

.. code-block:: python

    from pathlib import Path
    import pandas as pd

    def get_missingness_grid(chrom: str, window_size: int = 1e6) -> np.ndarray:
        """
        Returns a missingness grid (fraction missing per window) for the given chromosome.

        Args:
            chrom: Chromosome name, e.g., "chr1"
            window_size: Window size in base pairs (default: 1e6)

        Returns:
            missingness: NumPy array of shape (num_windows,) with values in [0, 1]
        """
        mask_base = Path("/sietch_colab/data_share/hg1kg/tsinfer-trees/working")
        mask = pd.read_pickle(mask_base / f"{chrom}.mask.pkl")
        pos = mask["position"]
        bcgm_mask = mask["bcgm_masked"].astype(int)

        edges = np.arange(0, pos.max() + window_size, window_size)
        idx = np.digitize(pos, edges) - 1
        totals = np.bincount(idx, minlength=len(edges) - 1)
        miss = np.bincount(idx, weights=bcgm_mask, minlength=len(edges) - 1)

        frac_missing = np.divide(
            miss,
            totals,
            out=np.zeros_like(miss, float),
            where=totals > 0,
        )
        return frac_missing

    chr2_available = 1 - get_missingness_grid("chr2")
    chr6_available = 1 - get_missingness_grid("chr6")
    chr2_available *= 100  # percent available
    chr6_available *= 100

We then apply :func:`cxt.utils.stochastic_diversity_bias_correction` per block
using the appropriate availability and tree sequence. The helper below returns
a mean-corrected TMRCA vector per block, which we then stitch into
chromosome-wide tracks:

.. code-block:: python

    from concurrent.futures import ProcessPoolExecutor as Pool
    from tqdm import tqdm
    from cxt.utils import stochastic_diversity_bias_correction

    # one block -> mean-corrected vector
    def _one_block(i: int):
        sample_index = (INDEX_MAP[:, 0] == i)
        tmrca_country_pivot = TMRCA_COUNTRY[:, sample_index]
        block = blocks[i]
        rng = np.random.default_rng(20_000_001 + i)  # per-block deterministic seed
        try:
            ts_block = TS_COUNTRY.keep_intervals([block], simplify=True)
            out = stochastic_diversity_bias_correction(
                tree_sequence=ts_block,
                mutation_rate=mutation_rate / (AVAILBILITY[i] / 100),
                predictions=tmrca_country_pivot,
                pivot_pairs=np.array(pivot_pairs),
                rng=rng,
            )
            return out.mean(0)
        except Exception:
            return np.zeros([25, 500])

    workers = 64
    mutation_rate = 1.29e-8

    # ---- chr2p ----
    num_blocks = 242
    TMRCA_COUNTRY = tmrca_chr2p
    TS_COUNTRY = ts_chr2p
    INDEX_MAP = index_map_chr2p
    AVAILBILITY = chr2_available

    if "genome_chr2p.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "genome_chr2p.npz"))
        genome_chr2p = data["genome"]
    else:
        with Pool(max_workers=workers) as ex:
            tmrca_genome = list(tqdm(ex.map(_one_block, range(num_blocks)), total=num_blocks))
        tmrca_genome = np.stack(tmrca_genome, axis=0)
        genome_chr2p = np.array(tmrca_genome).transpose(1, 0, 2).reshape(25, -1)
        np.savez_compressed(os.path.join(cache_dir, "genome_chr2p.npz"), genome=genome_chr2p)

    # ---- chr2q ----
    TMRCA_COUNTRY = tmrca_chr2q
    TS_COUNTRY = ts_chr2q
    INDEX_MAP = index_map_chr2q

    if "genome_chr2q.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "genome_chr2q.npz"))
        genome_chr2q = data["genome"]
    else:
        with Pool(max_workers=workers) as ex:
            tmrca_genome = list(tqdm(ex.map(_one_block, range(num_blocks)), total=num_blocks))
        tmrca_genome = np.stack(tmrca_genome, axis=0)
        genome_chr2q = np.array(tmrca_genome).transpose(1, 0, 2).reshape(25, -1)
        np.savez_compressed(os.path.join(cache_dir, "genome_chr2q.npz"), genome=genome_chr2q)

    genome_chr2 = genome_chr2p + genome_chr2q

    # ---- chr6p ----
    num_blocks = 170
    TMRCA_COUNTRY = tmrca_chr6p
    TS_COUNTRY = ts_chr6p
    INDEX_MAP = index_map_chr6p
    AVAILBILITY = chr6_available

    if "genome_chr6p.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "genome_chr6p.npz"))
        genome_chr6p = data["genome"]
    else:
        with Pool(max_workers=workers) as ex:
            tmrca_genome = list(tqdm(ex.map(_one_block, range(num_blocks)), total=num_blocks))
        tmrca_genome = np.stack(tmrca_genome, axis=0)
        genome_chr6p = np.array(tmrca_genome).transpose(1, 0, 2).reshape(25, -1)
        np.savez_compressed(os.path.join(cache_dir, "genome_chr6p.npz"), genome=genome_chr6p)

    # ---- chr6q ----
    TMRCA_COUNTRY = tmrca_chr6q
    TS_COUNTRY = ts_chr6q
    INDEX_MAP = index_map_chr6q

    if "genome_chr6q.npz" in os.listdir(cache_dir):
        data = np.load(os.path.join(cache_dir, "genome_chr6q.npz"))
        genome_chr6q = data["genome"]
    else:
        with Pool(max_workers=workers) as ex:
            tmrca_genome = list(tqdm(ex.map(_one_block, range(num_blocks)), total=num_blocks))
        tmrca_genome = np.stack(tmrca_genome, axis=0)
        genome_chr6q = np.array(tmrca_genome).transpose(1, 0, 2).reshape(25, -1)
        np.savez_compressed(os.path.join(cache_dir, "genome_chr6q.npz"), genome=genome_chr6q)

    genome_chr6 = genome_chr6p + genome_chr6q

Chromosome-wide overview of LCT and HLA
---------------------------------------

We can first plot a simple chromosome-wide mean TMRCA curve for chr2 and chr6,
highlighting the approximate LCT and HLA regions (GRCh38 coordinates).

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    generation_time = 28

    # 2 kb per window
    x_chr2 = np.arange(genome_chr2.shape[1]) * 2000
    x_chr6 = np.arange(genome_chr6.shape[1]) * 2000

    data = [
        (x_chr2, np.exp(genome_chr2.mean(0)) * generation_time, "Chromosome 2"),
        (x_chr6, np.exp(genome_chr6.mean(0)) * generation_time, "Chromosome 6"),
    ]

    # LCT ~135.3–136.4 Mb on chr2
    lct_start, lct_end = 135_300_000, 136_400_000
    # HLA (MHC region) ~29.6–33.3 Mb on chr6
    hla_start, hla_end = 29_600_000, 33_300_000

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharex=False, sharey=True)

    for ax, (x, y, label) in zip(axes, data):
        ax.plot(x, y, linewidth=0.8, color="dodgerblue", alpha=0.9)
        ax.set_yscale("log")
        ax.set_title(label, loc="left", fontsize=12)
        ax.grid(alpha=0.3, which="both", linestyle="--")

    axes[0].axvspan(lct_start, lct_end, color="crimson", alpha=0.3, label="LCT")
    axes[1].axvspan(hla_start, hla_end, color="crimson", alpha=0.3, label="HLA")

    axes[0].text(lct_start, 3e6, "LCT", color="crimson", fontsize=10, va="bottom")
    axes[1].text(hla_start, 3e6, "HLA", color="crimson", fontsize=10, va="bottom")

    fig.text(0.5, 0.03, "Position (bp)", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "Mean TMRCA (years)", va="center",
             rotation="vertical", fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.ylim(1e3, 1e7)
    plt.show()

Zoomed LCT and HLA panels with gene annotations
-----------------------------------------------

For the figure in the manuscript, we zoom into the LCT and HLA regions, show
replicates, robust IQR bands, mean and median trajectories, and overlay
selected genes as crimson bars.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    generation_time = 28
    BIN_BP = 2000  # 2 kb per window

    # --- genes (GRCh38 bp) ---
    hla_genes = [
        ("HLA-V",29759130,29765588,"+"),("HLA-H",29855342,29858857,"+"),
        ("HLA-F-AS1",29675299,29796273,"+"),("HLA-DRB6",32519632,32526623,"+"),
        ("HLA-DPA1",33032346,33041426,"+"),("HLA-L",30227330,30234728,"+"),
        ("HLA-DOB",32780540,32784779,"+"),("HLA-DQB2",32723875,32731309,"+"),
        ("HLA-DQB1",32627244,32634434,"+"),("HLA-DQB1-AS1",32627657,32628506,"+"),
        ("HLA-DPB1",33043767,33057473,"+"),("HLA-DMB",32902413,32908805,"+"),
        ("HLA-B",31321652,31324956,"+"),("HLA-DRA",32407664,32412823,"+"),
        ("HLA-A",29910309,29913647,"+"),("HLA-E",30457286,30461971,"+"),
        ("HLA-C",31236526,31239869,"+"),("HLA-DRB5",32485130,32498064,"+"),
        ("HLA-DQA2",32709168,32714975,"+"),("HLA-DMA",32916395,32920874,"+"),
        ("HLA-G",29795602,29798798,"+"),("HLA-DRB1",32546552,32557625,"+"),
        ("HLA-DQA1",32605183,32611461,"+"),("HLA-F",29691211,29695073,"+"),
        ("HLA-DOA",32971959,32977368,"+"),
    ]
    lct_genes = [
        ("MAP3K19",134964490,135047447,"-"),("RAB3GAP1",135052291,135176396,"+"),
        ("SNORA40B",135136627,135136755,"+"),("ZRANB3",135196968,135531218,"-"),
        ("R3HDM1",135531483,135725269,"+"),("MIR128-1",135665396,135665478,"+"),
        ("UBXN4",135741854,135785056,"+"),("LCT",135787849,135837184,"-"),
        ("LCT-AS1",135820190,135823087,"+"),("MCM6",135839625,135876443,"-"),
        ("DARS1",135905880,135985684,"-"),("DARS1-AS1",135985175,136007542,"+"),
        ("CXCR4",136114348,136118149,"-"),
    ]

    def stack_gene_labels(genes, min_sep_mb=0.6):
        """Greedy horizontal stacking to reduce overlaps."""
        sorted_genes = sorted(genes, key=lambda g: (g[1] + g[2]) / 2)
        levels, coords = [], []
        for name, start, end, strand in sorted_genes:
            mid = (start + end) / 2 / 1e6
            lvl = 0
            while (
                lvl < len(levels)
                and any(abs(mid - m) < min_sep_mb for m in levels[lvl])
            ):
                lvl += 1
            if lvl == len(levels):
                levels.append([])
            levels[lvl].append(mid)
            coords.append((mid, lvl, name, start / 1e6, end / 1e6))
        return coords

    def plot_zoom(ax, genome_log, chr_label, zoom_lo_mb, zoom_hi_mb, genes,
                  base_y=3e4, spacing=1.8, dashed_lines=None):
        """
        genome_log: (R, W) in ln(generations). Converts to years internally.
        dashed_lines: list of (y_value, label) to draw as dashed references (per panel).
        """
        R, W = genome_log.shape
        start_idx = max(0, int(np.floor((zoom_lo_mb * 1e6) / BIN_BP)))
        end_idx   = min(W, int(np.ceil((zoom_hi_mb * 1e6) / BIN_BP)))
        idx = np.arange(start_idx, end_idx)
        x_mb = (idx * BIN_BP) / 1e6

        # data → years; be NaN-safe
        Y = np.exp(np.asarray(genome_log[:, start_idx:end_idx], float)) * generation_time
        Y[~np.isfinite(Y)] = np.nan

        # replicates: slightly stronger
        for r in range(min(R, 10)):
            ax.plot(
                x_mb,
                Y[r],
                lw=0.6,
                color="dodgerblue",
                alpha=0.25,
                zorder=1,
            )

        # robust band: IQR
        q25 = np.nanpercentile(Y, 25, axis=0)
        q75 = np.nanpercentile(Y, 75, axis=0)
        mu  = np.nanmean(Y, axis=0)
        med = np.nanmedian(Y, axis=0)

        ax.fill_between(
            x_mb,
            q25,
            q75,
            alpha=0.18,
            color="dodgerblue",
            zorder=2,
            label="IQR (25–75%)",
        )

        # mean & median
        ax.plot(
            x_mb,
            mu,
            lw=2.0,
            color="dodgerblue",
            zorder=3,
            label="Mean",
        )
        ax.plot(
            x_mb,
            med,
            lw=1.6,
            color="dodgerblue",
            ls="--",
            zorder=3,
            label="Median",
        )

        # genes (crimson bars + labels)
        coords = stack_gene_labels(genes, min_sep_mb=0.6)
        for mid, lvl, name, g_lo, g_hi in coords:
            if g_hi < zoom_lo_mb or g_lo > zoom_hi_mb:
                continue
            y = base_y * (spacing ** lvl)
            a = max(g_lo, zoom_lo_mb)
            b = min(g_hi, zoom_hi_mb)
            ax.hlines(
                y,
                a,
                b,
                colors="crimson",
                lw=1.6,
                alpha=0.95,
                zorder=5,
            )
            ax.text(
                mid,
                y,
                name,
                ha="center",
                va="bottom",
                fontsize=7,
                color="crimson",
                zorder=6,
            )

        # optional per-panel dashed refs
        if dashed_lines:
            for yval, lab in dashed_lines:
                ax.axhline(yval, color="black", ls="--", lw=0.7)
                ax.text(
                    x_mb[0] + 0.05 * (x_mb[-1] - x_mb[0]),
                    yval * 0.92,
                    lab,
                    fontsize=7,
                    va="bottom",
                )

        ax.set_title(chr_label, loc="left", fontsize=12)
        ax.set_xlim(zoom_lo_mb, zoom_hi_mb)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both", ls="--")

    # zoom ranges (Mb)
    zoom_lct = (133.0, 138.0)
    zoom_hla = (29.0, 34.0)

    # per-panel dashed refs (optional, can be tuned)
    dashed_chr2 = [(1e4, "10 kyr"), (2e4, "20 kyr")]
    dashed_chr6 = [(1e7, "10 Myr"), (2e7, "20 Myr"), (3e7, "30 Myr")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)

    plot_zoom(
        axes[0],
        genome_chr2,
        "Chr2 LCT locus",
        *zoom_lct,
        lct_genes,
        base_y=3e4,
        spacing=1.9,
        dashed_lines=dashed_chr2,
    )
    plot_zoom(
        axes[1],
        genome_chr6,
        "Chr6 HLA region",
        *zoom_hla,
        hla_genes,
        base_y=3e4,
        spacing=1.9,
        dashed_lines=dashed_chr6,
    )

    fig.text(0.5, 0.03, "Genomic position (Mb)", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "TMRCA (years)", va="center",
             rotation="vertical", fontsize=12)

    axes[1].legend(loc="lower right", fontsize=9, ncol=3)

    plt.tight_layout(rect=[0.06, 0.05, 1, 0.98])
    plt.ylim(1e3, 3e8)
    plt.show()

With this pipeline, :math:`\mathbf{cxt}` produces whole-chromosome TMRCA tracks
for GBR samples, and the zoomed LCT and HLA panels closely match those
presented in the paper. Here, the impact of missing data is handled via a
**post-hoc diversity-based correction** rather than through a specialized
missingness-aware model, which is sufficient for these moderately masked human
regions.
