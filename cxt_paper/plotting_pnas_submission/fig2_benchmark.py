"""
Figure 2 (PNAS): True vs predicted TMRCA hexbin scatter plots.

2x3 grid with SMC++ inset:
  top row    = cxt-narrow constant, cxt-narrow sawtooth, cxt-broad sawtooth
  bottom row = Singer constant, Singer sawtooth, [SMC++ constant / SMC++ sawtooth stacked]

Matches the plotting logic of the original fig2_benchmark_comparison.py
and figures.utils.plot_tmrca_scatter.
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from pnas_defaults import (
    apply_pnas_style, savefig, resolve_cache,
    DOUBLE_COL, DEFAULT_OUTPUT, TIMES,
)

BIN_BP = 2000
SEQ_LEN = 1_000_000


def discretize(sequence):
    indices = np.searchsorted(TIMES, np.log(sequence), side="right") - 1
    indices = np.clip(indices, 0, len(TIMES) - 1)
    return np.exp(TIMES[indices])


def _hexbin_panel(ax, yhat, ytrue, title):
    """Hexbin scatter matching figures.utils.plot_tmrca_scatter."""
    eps = 1e-12
    ytrue = np.clip(np.asarray(ytrue).flatten(), eps, None)
    yhat = np.clip(np.asarray(yhat).flatten(), eps, None)
    yt_ln, yh_ln = np.log(ytrue), np.log(yhat)
    mask = np.isfinite(yt_ln) & np.isfinite(yh_ln)
    yt_ln, yh_ln = yt_ln[mask], yh_ln[mask]
    mse = float(np.mean((yh_ln - yt_ln) ** 2)) if yt_ln.size else float("nan")

    if yt_ln.size > 0:
        ax.hexbin(yt_ln, yh_ln, gridsize=120, cmap="plasma", bins="log", alpha=0.5)
        mn = float(min(yt_ln.min(), yh_ln.min()))
        mx = float(max(yt_ln.max(), yh_ln.max()))
        if mx == mn:
            mx = mn + 1.0
        ax.plot([mn, mx], [mn, mx], c="black", ls="-", lw=0.5)

        if yt_ln.size >= 2 and np.std(yt_ln) > 0:
            slope, intercept = np.polyfit(yt_ln, yh_ln, 1)
            xx = np.linspace(mn, mx, 100)
            ax.plot(xx, slope * xx + intercept, c="black", ls=":", lw=0.8)

        ln10 = np.log(10.0)
        exp_min = int(np.floor(mn / ln10))
        exp_max = int(np.ceil(mx / ln10))
        exps = np.arange(exp_min, exp_max + 1)
        ticks = exps * ln10
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"$10^{{{k}}}$" for k in exps])
        ax.set_yticklabels([f"$10^{{{k}}}$" for k in exps])

    ax.set_title(title, loc="left")
    ax.text(0.95, 0.05, f"MSE: {mse:.4f}", transform=ax.transAxes,
            ha="right", va="bottom")
    ax.grid(alpha=0.25)


def _placeholder(ax, title):
    ax.text(0.5, 0.5, "Cache not available", ha="center", va="center",
            fontsize=7, alpha=0.5)
    ax.set_title(title, loc="left")
    ax.grid(False)


def _pair_flat_idx(i, j, n=50):
    """Flat index for haploid pair (i,j) in the C(n,2) enumeration."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def _load_smcpp_scatter(pkl_path, cxt_npz_path):
    """Compute SMC++ scatter data from pkl cache + cxt true TMRCAs."""
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    cxt_data = np.load(cxt_npz_path)
    cxt_ytrues = cxt_data["ytrues"]

    bins = np.arange(0, SEQ_LEN + BIN_BP, BIN_BP)
    smcpp_all, true_all = [], []

    for r in results:
        pair = r["pair"]
        a, b = pair
        hs = r["hidden_states"]
        gamma = r["gamma"]
        N0 = float(r["N0"])
        sites = r["site_midpoints"]

        if np.all(np.isnan(gamma)):
            continue

        smcpp_tmrca = np.log(hs @ gamma * 2 * N0)
        windowed = np.array([
            np.nanmean(smcpp_tmrca[(sites >= x0) & (sites < x1)])
            for x0, x1 in zip(bins[:-1], bins[1:])
        ])
        smcpp_all.append(windowed)

        hap_i, hap_j = 2 * a, 2 * a + 1
        flat_idx = _pair_flat_idx(hap_i, hap_j)
        true_all.append(np.log(cxt_ytrues[flat_idx]))

    if not smcpp_all:
        return np.array([]), np.array([])

    yh = np.array(smcpp_all).flatten()
    yt = np.array(true_all).flatten()
    mask = np.isfinite(yh) & np.isfinite(yt)
    return np.exp(yh[mask]), np.exp(yt[mask])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_pnas_style()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.35)

    # --- Top row: cxt panels (3 regular axes) ---
    cxt_panels = [
        ("main/cache/fig2/constant_cxt.npz",
         r"$\mathbf{cxt}$-narrow: Constant $N_e$"),
        ("main/cache/fig2/sawtooth_narrow_cxt.npz",
         r"$\mathbf{cxt}$-narrow: Sawtooth $N_e$"),
        ("main/cache/fig2/sawtooth_broad_cxt.npz",
         r"$\mathbf{cxt}$-broad: Sawtooth $N_e$"),
    ]
    top_axes = []
    for col, (cache_rel, title) in enumerate(cxt_panels):
        ax = fig.add_subplot(gs[0, col])
        top_axes.append(ax)
        path = resolve_cache(cache_rel)
        if os.path.exists(path):
            data = np.load(path)
            yhats, ytrues = data["yhats"], data["ytrues"]
            ytrues_d = discretize(ytrues)
            _hexbin_panel(ax, yhats.mean(0), ytrues_d, title)
        else:
            _placeholder(ax, title)

    # --- Bottom row left: Singer panels (2 regular axes) ---
    singer_panels = [
        ("main/cache/singer/singer_constant.npz",
         r"Singer: Constant $N_e$"),
        ("main/cache/singer/singer_sawtooth.npz",
         r"Singer: Sawtooth $N_e$"),
    ]
    bot_axes = []
    for col, (cache_rel, title) in enumerate(singer_panels):
        ax = fig.add_subplot(gs[1, col])
        bot_axes.append(ax)
        path = resolve_cache(cache_rel)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            keys = list(data.files)
            if "yhats" in keys and "ytrues" in keys:
                yhats_s, ytrues_s = data["yhats"], data["ytrues"]
                ytrues_s_d = discretize(ytrues_s)
                yh = yhats_s.mean(0) if yhats_s.ndim > 1 else yhats_s
                _hexbin_panel(ax, yh, ytrues_s_d, title)
            else:
                _placeholder(ax, title)
        else:
            _placeholder(ax, title)

    # --- Bottom row right: two SMC++ panels stacked vertically ---
    gs_smcpp = gs[1, 2].subgridspec(2, 1, hspace=0.85)
    smcpp_panels = [
        ("main/cache/fig2/smcpp_constant.pkl",
         "main/cache/fig2/constant_cxt.npz",
         r"SMC++: Const. $N_e$"),
        ("main/cache/fig2/smcpp_sawtooth.pkl",
         "main/cache/fig2/sawtooth_broad_cxt.npz",
         r"SMC++: Sawtooth $N_e$"),
    ]
    smcpp_axes = []
    for row, (pkl_rel, cxt_rel, title) in enumerate(smcpp_panels):
        ax = fig.add_subplot(gs_smcpp[row])
        smcpp_axes.append(ax)
        pkl_path = resolve_cache(pkl_rel)
        cxt_path = resolve_cache(cxt_rel)
        if os.path.exists(pkl_path) and os.path.exists(cxt_path):
            yh_smcpp, yt_smcpp = _load_smcpp_scatter(pkl_path, cxt_path)
            if yh_smcpp.size > 0:
                yt_smcpp_d = discretize(yt_smcpp)
                _hexbin_panel(ax, yh_smcpp, yt_smcpp_d, title)
            else:
                _placeholder(ax, title)
        else:
            _placeholder(ax, title)
        ax.tick_params(labelsize=5)
        ax.set_title(ax.get_title(), fontsize=6)

        for txt in ax.texts:
            if txt.get_text().startswith("MSE:"):
                txt.set_fontsize(5.5)

        yticks = ax.get_yticks()
        ylabels = ax.get_yticklabels()
        ln10 = np.log(10.0)
        new_labels = []
        for t, lab in zip(yticks, ylabels):
            exp = int(round(t / ln10))
            if exp % 2 == 0:
                new_labels.append(f"$10^{{{exp}}}$")
            else:
                new_labels.append("")
        ax.set_yticklabels(new_labels)

    smcpp_axes[0].set_xlabel("")
    smcpp_axes[-1].set_xlabel("True TMRCA (gen.)", fontsize=6)

    for a in bot_axes:
        a.set_xlabel("True TMRCA (gen.)")
    top_axes[0].set_ylabel("Predicted TMRCA (gen.)")
    bot_axes[0].set_ylabel("Predicted TMRCA (gen.)")

    savefig(fig, "figure2", output_dir=args.output_dir)
    print("Done: figure2")


if __name__ == "__main__":
    main()
