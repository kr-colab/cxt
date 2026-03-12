"""
Figure 4: Out-of-sample evaluation on stdpopsim v0.3 species.

Three-section figure comparing cxt, Singer+Polegon, and SMC++ marginal
TMRCA distributions against truth for species not seen during training.
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

V3_SPECIES = [
    "MusMus_DomesticusEurope_1F22",
    "MusMus_MusculusKorea_1F22",
    "MusMus_CastaneusIndia_1F22",
    "RatNor_PiecewiseConstant",
    "GorGor_GorillaGhost_5P23",
    "OrySat_BottleneckMigration_3C07",
    "SusScr_PiecewiseConstant",
    "PhoSin_Vaquita2Epoch_1R22",
]

SPECIES_TITLES = [
    ("MusMus", "DomesticusEurope_1F22"),
    ("MusMus", "MusculusKorea_1F22"),
    ("MusMus", "CastaneusIndia_1F22"),
    ("RatNor", "PiecewiseConstant"),
    ("GorGor", "GorillaGhost_5P23"),
    ("OrySat", "BottleneckMigration_3C07"),
    ("SusScr", "PiecewiseConstant"),
    ("PhoSin", "Vaquita2Epoch_1R22"),
]

_ln10 = np.log(10.0)
COLS = 4


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


def set_loge_power10_ticks(ax, xmin, xmax, step=2):
    kmin = int(np.ceil(xmin / _ln10))
    kmax = int(np.floor(xmax / _ln10))
    ticks = [k * _ln10 for k in range(kmin, kmax + 1, step)]
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


def _load_cxt_results(cache_dir, metadata_all):
    results = []
    for m in metadata_all:
        fname = _metadata_name(m).replace(".trees", "_tmrca.npz")
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            d = np.load(path)
            results.append((d["yhats"], d["ytrues"]))
        else:
            results.append(None)
    return results


def _load_singer_results(singer_cache_dir):
    singer_path = os.path.join(singer_cache_dir, "singer_v3_tmrcas.npz")
    if not os.path.exists(singer_path):
        return [None] * len(V3_SPECIES)
    d = np.load(singer_path, allow_pickle=True)
    tmrcas_singer = d["tmrcas_singer"]
    results = []
    for k, spec in enumerate(V3_SPECIES):
        singer_mean = tmrcas_singer[k].mean(0)
        true_path = os.path.join(singer_cache_dir, f"true_tmrcas_{spec}.npz")
        ytrues = np.load(true_path)["ytrues"] if os.path.exists(true_path) else None
        results.append((singer_mean, ytrues))
    return results


def _load_smcpp_results(smcpp_cache_dir):
    results = []
    for i in range(len(V3_SPECIES)):
        path = os.path.join(smcpp_cache_dir, f"smcpp_figure4_ts_{i}.npz")
        if os.path.exists(path):
            d = np.load(path)
            results.append((d["yhats"], d["ytrues"]))
        else:
            results.append(None)
    return results


def _compute_kde_stats(yhat_raw, ytrue_raw):
    """Flatten, mask, discretize and return (yhat, ytrue_d, mse, kl, x, p_true, p_pred)."""
    yhat = np.asarray(yhat_raw).flatten()
    ytrue = np.asarray(ytrue_raw).flatten()
    mask = np.isfinite(yhat) & np.isfinite(ytrue)
    yhat, ytrue = yhat[mask], ytrue[mask]
    if yhat.size < 10:
        return None
    ytrue_d = TIMES[discretize(ytrue)]
    mse = float(np.mean((yhat - ytrue_d) ** 2))
    x = robust_grid(yhat, ytrue_d, n=512)
    p_true = kde_pdf(ytrue_d, x)
    p_pred = kde_pdf(yhat, x)
    kl = np.log10(kl_divergence(p_true, p_pred, x))
    return yhat, ytrue_d, mse, kl, x, p_true, p_pred


def main():
    plt.rcParams.update({"font.size": plt.rcParams["font.size"] + 3})

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig4")
    parser.add_argument("--singer-cache-dir", default=None)
    parser.add_argument("--smcpp-cache-dir", default=None)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    singer_cache_dir = args.singer_cache_dir
    if singer_cache_dir is None:
        for c in ["figures/output/main/cache/singer",
                   "/sietch_colab/data_share/cxt_scratch/figures/output/main/cache/singer",
                   os.path.join(os.path.dirname(args.cache_dir), "singer")]:
            if os.path.isdir(c):
                singer_cache_dir = c
                break
    smcpp_cache_dir = args.smcpp_cache_dir
    if smcpp_cache_dir is None:
        for c in [os.path.expanduser("~/cxt_paper_archive/smcpp_comparison/fig4_smc++"),
                  os.path.join(os.path.dirname(args.cache_dir), "smcpp", "fig4_smc++")]:
            if os.path.isdir(c):
                smcpp_cache_dir = c
                break

    meta_path = os.path.join(args.cache_dir, "stdpopsim_v3_metadata.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "rb") as f:
        metadata_all = pickle.load(f)

    cxt_results = _load_cxt_results(args.cache_dir, metadata_all)
    singer_results = _load_singer_results(singer_cache_dir) if singer_cache_dir else [None] * 8
    smcpp_results = _load_smcpp_results(smcpp_cache_dir) if smcpp_cache_dir else [None] * 8

    has_singer = any(r is not None for r in singer_results)
    has_smcpp = any(r is not None for r in smcpp_results)

    methods = [("cxt", "steelblue", cxt_results)]
    if has_singer:
        methods.append(("Singer+Polegon", "steelblue", singer_results))
    if has_smcpp:
        methods.append(("SMC++", "steelblue", smcpp_results))

    n_species = len(metadata_all)
    species_rows = int(np.ceil(n_species / COLS))
    n_methods = len(methods)
    total_rows = species_rows * n_methods

    fig, axes = plt.subplots(
        total_rows, COLS,
        figsize=(5 * COLS, 1.8 * total_rows),
        squeeze=False,
    )

    for m_idx, (method_name, method_color, method_results) in enumerate(methods):
        row_offset = m_idx * species_rows
        is_last_method = (m_idx == n_methods - 1)

        for k in range(n_species):
            r_local, c = divmod(k, COLS)
            r = row_offset + r_local
            ax = axes[r, c]

            is_bottom = (r_local == species_rows - 1) and is_last_method

            result = method_results[k] if k < len(method_results) else None
            yhat_raw = None
            if result is not None:
                yhat_raw_orig, ytrue_raw = result
                if method_name == "cxt" and yhat_raw_orig.ndim == 3:
                    yhat_raw = yhat_raw_orig.mean(0)
                else:
                    yhat_raw = yhat_raw_orig

            stats = None
            if yhat_raw is not None and ytrue_raw is not None:
                stats = _compute_kde_stats(yhat_raw, ytrue_raw)

            if stats is None:
                ax.set_xlim(0, 16.2)
                ax.set_ylim(0, 1.0)
                set_loge_power10_ticks(ax, 0, 16.2)
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="0.6")
            else:
                _, _, mse_val, kl_val, x, p_true, p_pred = stats

                ax.plot(x, p_true, color="black", ls="--", lw=1.0,
                        label="True KDE" if k == 0 else None)
                ax.fill_between(x, p_pred, alpha=0.3, color=method_color)
                ax.plot(x, p_pred, color=method_color, lw=1.0,
                        label=method_name if k == 0 else None)

                ax.set_xlim(0, 16.2)
                ax.set_ylim(0, 1.0)
                set_loge_power10_ticks(ax, 0, 16.2)

                ax.text(0.03, 0.08,
                        f"MSE={mse_val:.3g}\nlog\u2081\u2080KL={kl_val:.2f}",
                        ha="left", va="bottom", transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#d6e6f5", edgecolor="none",
                                  alpha=0.85))

            sp, demo = SPECIES_TITLES[k]
            ax.set_title(f"{sp}\n{demo}", loc="left")

            if k == 0:
                ax.legend(loc="upper right", framealpha=0.8,
                          edgecolor="none")

            # Y-axis
            if c == 0:
                ax.set_ylabel("Density")
            else:
                ax.tick_params(axis="y", labelleft=False)

            if is_bottom:
                ax.set_xlabel("TMRCA (gen.)")
            else:
                ax.tick_params(axis="x", labelbottom=False)

        # Row label stored for placement after tight_layout
        methods[m_idx] = (method_name, method_color, method_results, row_offset)

        for j in range(n_species, species_rows * COLS):
            rr, cc = divmod(j, COLS)
            axes[row_offset + rr, cc].axis("off")

    plt.tight_layout(rect=[0.04, 0, 1, 1])

    # Method labels and separators placed after tight_layout
    for m_idx in range(n_methods):
        method_name = methods[m_idx][0]
        row_offset = methods[m_idx][3]
        top_pos = axes[row_offset, 0].get_position()
        bot_pos = axes[row_offset + species_rows - 1, 0].get_position()
        y_center = (top_pos.y1 + bot_pos.y0) / 2
        fig.text(0.015, y_center, method_name,
                 ha="center", va="center", rotation=90,
                 fontsize=17, fontweight="bold",
                 transform=fig.transFigure)

    out = os.path.join(args.output_dir, "figure4_stdpopsim_v3.pdf")
    fig.savefig(out, dpi=300)
    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=300)
    print(f"Saved {out} and {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
