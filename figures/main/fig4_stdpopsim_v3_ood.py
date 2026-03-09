"""
Figure 4: Out-of-sample evaluation of the broad model on stdpopsim v0.3 species.

Compares true vs predicted marginal TMRCA distributions for species not seen
during training (MusMus, RatNor, GorGor, OrySat, SusScr, PhoSin).
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

import cxt
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import TIMES
from figures.utils import STDPOPSIM_V3_PARAMS, simulate_segment


def _interp_worker(args):
    ts, a, b = args
    return interpolate_tmrcas(ts, 2000, 1e6, a, b)


def build_yhats_ytrues(ts, pivot_pairs, yhat_tmrca, max_workers=None):
    yhat_tmrca = np.exp(yhat_tmrca)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        ytrues = list(ex.map(_interp_worker, [(ts, a, b) for a, b in pivot_pairs]))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig4")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = cxt.load_model("broad", device="cpu")
    pivot_pairs = [(i, j) for i in range(50) for j in range(i + 1, 50)]
    blocks = [(0, int(1e6))]

    # --- Simulate or load v0.3 species ---
    meta_path = os.path.join(args.cache_dir, "stdpopsim_v3_metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            metadata_all = pickle.load(f)
        data_all = []
        for m in metadata_all:
            data_all.append(__import__("tskit").load(os.path.join(args.cache_dir, _metadata_name(m))))
    else:
        data_all, metadata_all = [], []
        for key, params in STDPOPSIM_V3_PARAMS.items():
            tss, meta = simulate_segment(
                seed=params["seed"], species_name=key,
                left=params.get("left"), right=params.get("right"),
                length=params.get("length"),
                num_samples=params["num_samples"],
                population_size=params.get("population_size"),
            )
            data_all += tss
            metadata_all += meta
            for i, ts in enumerate(tss):
                ts.dump(os.path.join(args.cache_dir, _metadata_name(meta[i])))
        with open(meta_path, "wb") as f:
            pickle.dump(metadata_all, f)

    # --- Run inference ---
    tmrca_results = []
    for i, ts in enumerate(data_all):
        cache_name = _metadata_name(metadata_all[i]).replace(".trees", "_tmrca.npz")
        cache_path = os.path.join(args.cache_dir, cache_name)
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            tmrca_results.append((d["yhats"], d["ytrues"]))
            continue

        mutation_rate = json.loads(ts.provenance(-1).record)["parameters"]["rate"]
        tmrca, _ = cxt.translate(
            ts, model, pivot_pairs=pivot_pairs,
            blocks=blocks, devices=args.devices,
            B_per_device=args.batch_size, B=args.batch_size,
            build_workers=24, mutation_rate=mutation_rate,
        )
        yhats, ytrues = build_yhats_ytrues(ts, pivot_pairs, tmrca, max_workers=24)
        yhats, ytrues = np.array(yhats), np.array(ytrues)
        np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)
        tmrca_results.append((yhats, ytrues))

    # --- Plot KDEs ---
    cols = min(len(tmrca_results), 4)
    n = len(tmrca_results)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 1.8 * rows), squeeze=False)

    for k in range(n):
        yhat, ytrue = tmrca_results[k]
        yhat = np.asarray(yhat.mean(0)).flatten()
        ytrue = np.asarray(ytrue).flatten()
        mask = np.isfinite(yhat) & np.isfinite(ytrue)
        yhat, ytrue = yhat[mask], ytrue[mask]
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
            f"{metadata_all[k]['species_name']}\n{metadata_all[k].get('id', '')}",
            loc="left",
        )
        ax.text(0.98, 0.97, f"MSE = {mse_val:.3g}\nlog\u2081\u2080(KL) = {kl_val:.3g}",
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
    out = os.path.join(args.output_dir, "figure4_stdpopsim_v3.png")
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
