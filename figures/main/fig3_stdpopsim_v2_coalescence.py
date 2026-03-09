"""
Figure 3: Marginal coalescence distributions across stdpopsim v0.2 species.

For each species/demography, compares true vs predicted TMRCA KDEs,
reporting MSE and KL divergence. Produces two panels: one for simulations
without genetic maps, one with genetic maps.
"""

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde
import pandas as pd

import cxt
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import TIMES
from figures.utils import STDPOPSIM_V2_PARAMS, simulate_segment


def _interp_worker(args):
    ts, a, b, interval_start = args
    return interpolate_tmrcas(ts, 2000, 1e6, a, b, interval_start=interval_start)


def build_yhats_ytrues(ts, pivot_pairs, yhat_tmrca, interval_start=0, max_workers=None):
    yhat_tmrca = np.exp(yhat_tmrca)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ytrues = list(ex.map(
            _interp_worker,
            [(ts, a, b, interval_start) for a, b in pivot_pairs],
        ))
    return np.log(yhat_tmrca), np.log(ytrues)


def discretize(sequence):
    indices = np.searchsorted(TIMES, sequence, side="right") - 1
    return np.clip(indices, 0, len(TIMES) - 1)


def kde_pdf(samples, grid, bw_method=None):
    if np.all(samples == samples[0]):
        loc = float(samples[0])
        pdf = np.exp(-0.5 * ((grid - loc) / (1e-6 + 0.01 * (np.ptp(grid) or 1.0))) ** 2)
        pdf /= np.trapezoid(pdf, grid)
        return pdf
    kde = gaussian_kde(samples, bw_method=bw_method)
    pdf = kde(grid)
    area = np.trapezoid(pdf, grid)
    return pdf / area if area > 0 else pdf


def kl_divergence(p_grid, q_grid, x_grid, eps=1e-12):
    p, q = p_grid + eps, q_grid + eps
    p /= np.trapezoid(p, x_grid)
    q /= np.trapezoid(q, x_grid)
    return float(np.trapezoid(p * (np.log(p) - np.log(q)), x_grid))


def robust_grid(a, b, n=512, q_lo=0.005, q_hi=0.995):
    lo = np.nanmin([np.quantile(a, q_lo), np.quantile(b, q_lo)])
    hi = np.nanmax([np.quantile(a, q_hi), np.quantile(b, q_hi)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = min(np.nanmin(a), np.nanmin(b)), max(np.nanmax(a), np.nanmax(b))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, n)


_ln10 = np.log(10.0)


def set_loge_power10_ticks(ax, xmin, xmax):
    kmin = int(np.ceil(xmin / _ln10))
    kmax = int(np.floor(xmax / _ln10))
    ticks = [k * _ln10 for k in range(kmin, kmax + 1)]
    if ticks:
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: r"$10^{%d}$" % round(np.log10(np.e) * x)
        ))


def _metadata_name(m):
    return (
        f"{m['species_name']} {m['demography']} with map {m.get('genetic_map')}"
        .replace(" ", "_").replace("/", "_") + ".trees"
    )


def plot_kdes(indices, tmrca_results, metadata_df, cols, output_path):
    n = len(indices)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 1.8 * rows), squeeze=False)

    for k, index in enumerate(indices):
        yhat, ytrue = tmrca_results[index]
        yhat = np.asarray(yhat.mean(0)).flatten()
        ytrue = np.asarray(ytrue).flatten()

        ytrue = TIMES[discretize(ytrue)]

        mse_val = float(np.mean((yhat - ytrue) ** 2))
        x = robust_grid(yhat, ytrue, n=512)
        p_true = kde_pdf(ytrue, x)
        p_pred = kde_pdf(yhat, x)
        kl_val = np.log10(kl_divergence(p_true, p_pred, x))

        r, c = k // cols, k % cols
        ax = axes[r, c]
        ax.plot(x, p_true, color="black", label="True KDE")
        ax.plot(x, p_pred, color="dodgerblue", label="Pred KDE")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 16.2)
        ax.grid(alpha=0.3)
        set_loge_power10_ticks(ax, 0, 16.2)

        ax.set_title(
            f"{metadata_df.loc[k, 'species_name']}\n{metadata_df.loc[k, 'id']}",
            loc="left",
        )
        ax.text(0.98, 0.97,
                f"MSE = {mse_val:.3g}\nlog\u2081\u2080(KL) = {kl_val:.3g}",
                ha="right", va="top", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2))
        if k == 0:
            ax.legend(loc="best")
        if c == 0:
            ax.set_ylabel("Density")
        else:
            ax.tick_params(axis="y", labelleft=False)
        if r == rows - 1:
            ax.set_xlabel("TMRCA (generations)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig3")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = cxt.load_model("broad", device="cpu")
    pivot_pairs = [(i, j) for i in range(50) for j in range(i + 1, 50)]

    # --- Simulate or load stdpopsim v0.2 data ---
    meta_path = os.path.join(args.cache_dir, "stdpopsim_metadata.pkl")
    if os.path.exists(meta_path):
        import tskit
        with open(meta_path, "rb") as f:
            stdpopsim_metadata = pickle.load(f)
        stdpopsim_data = []
        for m in stdpopsim_metadata:
            name = _metadata_name(m)
            stdpopsim_data.append(tskit.load(os.path.join(args.cache_dir, name)))
    else:
        stdpopsim_data, stdpopsim_metadata = [], []
        for key, params in STDPOPSIM_V2_PARAMS.items():
            tss, meta = simulate_segment(
                seed=params["seed"],
                species_name=key,
                genetic_map_tuple=params.get("genetic_map_tuple"),
                left=params.get("left"),
                right=params.get("right"),
                num_samples=params["num_samples"],
                population_size=params.get("population_size"),
            )
            stdpopsim_data += tss
            stdpopsim_metadata += meta
            for i, ts in enumerate(tss):
                ts.dump(os.path.join(args.cache_dir, _metadata_name(meta[i])))
        with open(meta_path, "wb") as f:
            pickle.dump(stdpopsim_metadata, f)

    # --- Run inference or load cached results ---
    tmrca_results = []
    for i, ts in enumerate(stdpopsim_data):
        cache_name = _metadata_name(stdpopsim_metadata[i]).replace(".trees", "_tmrca.npz")
        cache_path = os.path.join(args.cache_dir, cache_name)

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            tmrca_results.append((data["yhats"], data["ytrues"]))
            continue

        species_key = stdpopsim_metadata[i]["species_name"]
        left = int(STDPOPSIM_V2_PARAMS[species_key].get("left", 0))
        right = int(STDPOPSIM_V2_PARAMS[species_key].get("right", left + int(1e6)))
        blocks = [(left, right)]

        mutation_rate = json.loads(ts.provenance(-1).record)["parameters"]["rate"]
        tmrca, _ = cxt.translate(
            ts, model, pivot_pairs=pivot_pairs,
            blocks=blocks, devices=args.devices,
            B_per_device=args.batch_size, B=args.batch_size,
            build_workers=24, mutation_rate=mutation_rate,
        )
        yhats, ytrues = build_yhats_ytrues(
            ts, pivot_pairs, tmrca, interval_start=left, max_workers=24,
        )
        yhats, ytrues = np.array(yhats), np.array(ytrues)
        np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)
        tmrca_results.append((yhats, ytrues))

    # --- Split by genetic map presence and plot ---
    metadata = pd.DataFrame(stdpopsim_metadata)
    mask_gm = ~metadata["genetic_map"].isnull()

    metadata_no_gm = metadata[~mask_gm].reset_index(drop=True)
    metadata_gm = metadata[mask_gm].reset_index(drop=True)

    no_gm_idx = mask_gm[~mask_gm].index
    gm_idx = mask_gm[mask_gm].index

    plot_kdes(no_gm_idx, tmrca_results, metadata_no_gm, cols=4,
              output_path=os.path.join(args.output_dir, "figure3_tmrca_kdes.png"))
    plot_kdes(gm_idx, tmrca_results, metadata_gm, cols=4,
              output_path=os.path.join(args.output_dir, "figure3_tmrca_kdes_map.png"))


if __name__ == "__main__":
    main()
