"""
Figure 7: Ag1000G A. gambiae Rdl region TMRCA across five African populations.

Zoomed view of the RDL insecticide-resistance region (25.1–25.6 Mb on chr2L) with
five populations stacked, replicate curves, mean±SD band, RDL gene highlight,
and accessibility-driven missingness track in the bottom panel.

Infers genome-wide chr2L TMRCA then plots the RDL slice. Requires Ag1000G tree
sequences and accessibility masks.

Faithful reproduction of revision/figure_mosquito/experiment_integrated_missing.ipynb
(Cells 40–41) and docs/source/mosquito.rst RDL zoom panel.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
import tskit
from matplotlib.ticker import FuncFormatter

from figures.paths import AG1000G_DATA_DIR, AG1000G_ACCESSIBILITY
import matplotlib.pyplot as plt

import cxt


POPULATIONS = {
    "BurkinaFaso": "Burkina Faso",
    "Mali": "Mali",
    "Cameroon": "Cameroon",
    "Ghana": "Ghana",
    "Uganda": "Uganda",
}

POP_ORDER = ["Mali", "BurkinaFaso", "Cameroon", "Uganda", "Ghana"]

N_BLOCKS = 490
MUTATION_RATE = 3.5e-9
BLOCK_SIZE = 0.1e6
STEP_BP = 200

# RDL zoom region (25.1–25.6 Mb)
REGION_START = 25_100_000
REGION_END = 25_600_000
RDL_START = 25_363_652
RDL_END = 25_434_556

fmt_mb = FuncFormatter(lambda v, _: f"{v/1e6:.1f}")


def _extract_population_ts(full_ts, country_name, cache_dir, pop_key,
                           n_individuals=25):
    """Filter 2La=1 heterozygotes for a given country."""
    ts_path = os.path.join(cache_dir, f"ts_{pop_key.lower()}.trees")
    if os.path.exists(ts_path):
        return tskit.load(ts_path)

    ids = []
    for ind in full_ts.individuals():
        meta = json.loads(ind.metadata.decode("ascii"))
        if meta["country"] == country_name and int(meta["2La"]) == 1:
            ids.append(ind.nodes)
    if len(ids) < n_individuals:
        print(f"  Warning: Only {len(ids)} inv-het individuals for {country_name} "
              f"(requested {n_individuals}), using all available")
    n_use = min(len(ids), n_individuals)
    ts_pop = full_ts.simplify(samples=np.concatenate(ids[:n_use]))
    ts_pop.dump(ts_path)
    return ts_pop


def _infer_genome(ts_pop, model, missingness_bitmask, devices,
                  cache_dir, pop_key, adapter=None):
    """Run cxt inference across chr2L for one population."""
    tmrca_path = os.path.join(cache_dir, f"tmrca_{pop_key.lower()}_wmissing_2L.npz")
    if os.path.exists(tmrca_path):
        data = np.load(tmrca_path)
        return data["tmrca"], data["index_map"]

    blocks = [(int(i * BLOCK_SIZE), int((i + 1) * BLOCK_SIZE))
              for i in range(N_BLOCKS)]
    pivot_pairs = [tuple(ind.nodes) for ind in ts_pop.individuals()]

    tmrca, index_map = cxt.translate(
        ts_pop, model,
        pivot_pairs=pivot_pairs,
        blocks=blocks,
        devices=devices,
        B_per_device=512, B=512,
        build_workers=36,
        mutation_rate=MUTATION_RATE,
        missingness_bitmask=missingness_bitmask,
        adapter=adapter,
    )
    np.savez_compressed(tmrca_path, tmrca=tmrca, index_map=index_map)
    return tmrca, index_map


def _make_genome(tmrca, index_map, n_pairs):
    """Assemble per-pair genome-wide TMRCA arrays.

    Matches revision notebook Cell 20 ``make_tmrca_genome`` exactly.
    tmrca shape: (n_reps, n_items, 500) from cxt.translate.
    Returns (n_pairs, n_windows_total) in LINEAR space (generations).
    """
    genomes = []
    for i in range(n_pairs):
        mask = index_map[:, 1] == i
        genomes.append(tmrca[:, mask, :].mean(0).flatten())
    return np.exp(np.array(genomes))


def _as_2d(x):
    x = np.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def moving_average(frac, smooth_bp=5_000, step_bp=200):
    k = max(1, int(round(smooth_bp / step_bp)))
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(frac.astype(np.float32), kernel, mode="same")


def missing_track_from_bitmask(unaccessible_bitmask, step_bp=200):
    """Per-bin missing fraction for RDL region."""
    region = np.asarray(unaccessible_bitmask[REGION_START:REGION_END], dtype=np.bool_)
    n_bins = (REGION_END - REGION_START) // step_bp
    pad = n_bins * step_bp - region.size
    if pad > 0:
        region = np.pad(region, (0, pad), constant_values=False)
    miss_frac = region[:n_bins * step_bp].reshape(n_bins, step_bp).mean(
        axis=1
    ).astype(np.float32)
    return miss_frac


def _draw_missing_track(ax, x, missing_frac, smooth_bp=5_000, step_bp=200,
                        height=0.16, pad=0.26, xlabel=None):
    """Draw compact missingness track below main axis."""
    y_raw = np.clip(missing_frac, 0, 1)
    y_smooth = np.clip(moving_average(y_raw, smooth_bp, step_bp), 0, 1)

    tr = ax.inset_axes([0.0, -pad, 1.0, height], transform=ax.transAxes)
    tr.fill_between(x, 0, y_smooth, alpha=0.35, color="lightsteelblue", zorder=1)
    tr.plot(x, y_smooth, lw=1.1, color="steelblue", alpha=0.95, zorder=2)
    tr.plot(x, y_raw, lw=0.5, color="steelblue", alpha=0.30, zorder=2)

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


def _extract_rdl_slice(genome, step_bp):
    """Extract RDL region [REGION_START, REGION_END] from genome (n_reps, n_bins)."""
    start_block = REGION_START // step_bp
    n_bins_region = (REGION_END - REGION_START) // step_bp
    end_block = start_block + n_bins_region
    n_bins_chr = genome.shape[1]
    if end_block > n_bins_chr:
        end_block = n_bins_chr
        start_block = end_block - n_bins_region
    return genome[:, start_block:end_block]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig7")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--data-dir", default=AG1000G_DATA_DIR)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip inference; load genome caches and plot only")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.plot_only:
        genome_data = {}
        for pop_key in POP_ORDER:
            path = os.path.join(args.cache_dir, f"genome_{pop_key.lower()}.npz")
            if not os.path.exists(path):
                print(f"Cache not found: {path}. Run without --plot-only first.")
                return
            genome_data[pop_key] = np.load(path)["genome"]
        bitmask_path = AG1000G_ACCESSIBILITY
        unaccessible_bitmask = ~np.load(bitmask_path)["access_2L"]
    else:
        tree_path = os.path.join(args.data_dir, "gamb.2L.gff.dated.ne.trees")
        print(f"Loading full chr2L tree from {tree_path} ...")
        full_ts = tskit.load(tree_path)

        bitmask_path = AG1000G_ACCESSIBILITY
        print(f"Loading accessibility mask from {bitmask_path} ...")
        unaccessible_bitmask = ~np.load(bitmask_path)["access_2L"]

        base_model = cxt.load_model("w200_wmissing", device="cpu")
        adapter_model = cxt.load_model("w200_wmissing_adapter", device="cpu")

        genome_data = {}
        for pop_key, country_name in POPULATIONS.items():
            print(f"\n{'=' * 60}\n Processing {pop_key} ({country_name})\n{'=' * 60}")

            ts_pop = _extract_population_ts(
                full_ts, country_name, args.cache_dir, pop_key)
            print(f"  {ts_pop.num_samples} samples, {ts_pop.num_sites} sites")

            if pop_key == "Ghana":
                ts_pop = ts_pop.simplify(samples=np.arange(0, 10))
                print(f"  Ghana adapter mode: simplified to {ts_pop.num_samples} samples")
                tmrca, index_map = _infer_genome(
                    ts_pop, adapter_model.backbone, unaccessible_bitmask,
                    args.devices, args.cache_dir, pop_key,
                    adapter=adapter_model.adapter,
                )
            else:
                tmrca, index_map = _infer_genome(
                    ts_pop, base_model, unaccessible_bitmask,
                    args.devices, args.cache_dir, pop_key,
                )

            pivot_pairs = [tuple(ind.nodes) for ind in ts_pop.individuals()]
            genome = _make_genome(tmrca, index_map, len(pivot_pairs))

            genome_path = os.path.join(args.cache_dir, f"genome_{pop_key.lower()}.npz")
            np.savez_compressed(genome_path, genome=genome)
            genome_data[pop_key] = genome
            print(f"  Genome shape: {genome.shape}")

            del tmrca, index_map
            gc.collect()
            torch.cuda.empty_cache()

    # --- RDL slice + missingness ---
    missing_frac = missing_track_from_bitmask(unaccessible_bitmask, STEP_BP)
    n_bins_region = (REGION_END - REGION_START) // STEP_BP
    base_x = np.arange(n_bins_region) * STEP_BP + REGION_START

    # --- Figure: RDL zoom, 5 pops stacked, missingness in bottom panel ---
    n_pops = len(POP_ORDER)
    fig, axes = plt.subplots(n_pops, 1, figsize=(6, 7), sharex=True, sharey=True)
    if n_pops == 1:
        axes = [axes]

    for i, pop_key in enumerate(POP_ORDER):
        ax = axes[i]
        genome = _as_2d(genome_data[pop_key])
        tmrca_slice = _extract_rdl_slice(genome, STEP_BP)
        name = POPULATIONS[pop_key]

        mean_t = np.nanmean(tmrca_slice, axis=0)
        std_t = np.nanstd(tmrca_slice, axis=0)
        n = tmrca_slice.shape[0]

        for j in range(n):
            alpha = 0.30 + 0.40 * (j / max(1, n - 1)) if n > 1 else 0.7
            lw = 0.6 + 0.4 * (j % 5 == 0)
            ax.plot(base_x, tmrca_slice[j], lw=lw, color="steelblue",
                    alpha=alpha, zorder=0)

        eps = 1e-12
        lower = np.clip(mean_t - std_t, eps, None)
        upper = np.clip(mean_t + std_t, eps, None)
        ax.fill_between(base_x, lower, upper, color="deepskyblue",
                        alpha=0.35, zorder=1)
        ax.plot(base_x, lower, ls="--", lw=0.8, color="dodgerblue",
                alpha=0.9, zorder=2)
        ax.plot(base_x, upper, ls="--", lw=0.8, color="dodgerblue",
                alpha=0.9, zorder=2)
        ax.plot(base_x, mean_t, lw=1.0, color="dodgerblue", zorder=3)

        # RDL shading + arrow
        ax.axvspan(RDL_START, RDL_END, color="crimson", alpha=0.10, zorder=0)
        mid = (RDL_START + RDL_END) / 2
        valid = mean_t[np.isfinite(mean_t) & (mean_t > 0)]
        y_arrow = np.exp((np.log(valid.min()) + np.log(valid.max())) / 3) if valid.size else 1e3
        ax.annotate("", xy=(RDL_END, y_arrow), xytext=(RDL_START, y_arrow),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"), zorder=6)
        ax.text(mid, y_arrow * 1.05, "RDL", fontsize=9, ha="center", va="bottom",
                zorder=6)

        ax.set_yscale("log")
        ax.set_xlim(REGION_START, REGION_END)
        ax.set_ylim(0.5e2, 5e6)
        ax.grid(True, alpha=0.3, which="both", linestyle="--")
        ax.set_title(name, fontsize=11, loc="left")
        ax.set_ylabel("TMRCA (gen.)")
        ax.xaxis.set_major_formatter(fmt_mb)
        ax.tick_params(axis="x", which="both", labelbottom=False)

        is_bottom = (i == n_pops - 1)
        if missing_frac is not None:
            _draw_missing_track(
                ax, base_x, missing_frac,
                smooth_bp=5_000, step_bp=STEP_BP,
                xlabel="Position on 2L (Mb)" if is_bottom else None,
            )
        elif is_bottom:
            ax.set_xlabel("Position on 2L (Mb)")

    plt.tight_layout()
    out = os.path.join(args.output_dir, "figure7_mosquito_rdl.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
