"""
Figure 8 (PNAS): Coalescent-time structure across the In(2L)a inversion.

Top: bar chart comparing mean TMRCA across four genomic regions — Outside
(10–20 Mb), Core interval, Inner proximal (+1.0 Mb), Inner distal (−1.0 Mb)
— for each population plus "All".
Bottom: 2×3 grid of rolling-mean TMRCA curves highlighting the inversion
core interval (~20–42 Mb) on chr2L.

Reads genome-wide TMRCA caches produced by fig7_mosquito_rdl.py — never
runs inference.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pnas_defaults import (
    apply_pnas_style, savefig, resolve_cache,
    DOUBLE_COL, DEFAULT_OUTPUT,
)

STEP_BP = 200

INV_START = 20_524_058
INV_END = 42_165_532
OUT_START = 10_000_000
OUT_END = 20_000_000

BP_WINDOW = 500_000
INNER_OFFSET = 1_000_000
INNER_PROX_START = INV_START + INNER_OFFSET
INNER_PROX_END = INNER_PROX_START + BP_WINDOW
INNER_DIST_END = INV_END - INNER_OFFSET
INNER_DIST_START = INNER_DIST_END - BP_WINDOW

ROLL = 1001

COL_OUT = "#D6DEE8"
COL_CORE = "#2F6FDB"
COL_PROX = "#8EC5FF"
COL_DIST = "#1F4E8C"
LINE_CORE = "#1F4E8C"
LINE_BP = "#2F6FDB"

POP_ORDER = ["Mali", "BurkinaFaso", "Cameroon", "Uganda", "Ghana"]
POP_LABELS = {
    "Mali": "Mali",
    "BurkinaFaso": "Burkina Faso",
    "Cameroon": "Cameroon",
    "Uganda": "Uganda",
    "Ghana": "Ghana",
}

R_OUT = "Outside (10\u201320 Mb)"
R_CORE = "Core interval"
R_IP = "Inner prox. (+1 Mb)"
R_ID = "Inner dist. (\u22121 Mb)"


def _as_2d(x):
    x = np.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def rolling_nanmean(y, win):
    if win <= 1:
        return y
    k = win + (win % 2 == 0)
    w = np.ones(k)
    y0 = np.nan_to_num(y, nan=0.0)
    m0 = np.isfinite(y).astype(float)
    num = np.convolve(y0, w, mode="same")
    den = np.convolve(m0, w, mode="same")
    out = num / np.maximum(den, 1e-12)
    out[den == 0] = np.nan
    return out


def region_stats(genome, step_bp, a, b):
    start = max(0, int(a // step_bp))
    end = min(genome.shape[1], int(b // step_bp))
    if end <= start:
        return np.nan, np.nan
    per_rep = np.nanmean(genome[:, start:end], axis=1)
    m = np.nanmean(per_rep)
    n = np.sum(np.isfinite(per_rep))
    s = np.nanstd(per_rep, ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, s


def harmonize_lengths(arrays):
    W = min(a.shape[1] for a in arrays)
    return [a[:, :W] for a in arrays], W


def draw_vlines(ax):
    ax.axvline(INV_START / 1e6, color=LINE_CORE, lw=0.8, alpha=0.2)
    ax.axvline(INV_END / 1e6, color=LINE_CORE, lw=0.8, alpha=0.2)
    ax.axvline(INV_START / 1e6, color=LINE_BP, lw=0.6, ls=":")
    ax.axvline(INV_END / 1e6, color=LINE_BP, lw=0.6, ls=":")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_pnas_style()

    # Load genome caches (LINEAR scale from fig7)
    datasets_raw = []
    for pop_key in POP_ORDER:
        cache_path = resolve_cache(
            f"main/cache/fig7/genome_{pop_key.lower()}.npz"
        )
        if not os.path.exists(cache_path):
            print(f"Skipping {pop_key}: cache not found at {cache_path}")
            continue
        genome = _as_2d(np.load(cache_path)["genome"])
        print(f"  {POP_LABELS[pop_key]}: {genome.shape[0]} replicates, "
              f"{genome.shape[1]} bins")
        datasets_raw.append((POP_LABELS[pop_key], genome))

    if not datasets_raw:
        print("No genome caches found. Run fig7_mosquito_rdl.py first.")
        return

    names = [n for n, _ in datasets_raw]
    arrays = [a for _, a in datasets_raw]
    arrays, W = harmonize_lengths(arrays)
    datasets = list(zip(names, arrays))
    x_mb = (np.arange(W) * STEP_BP) / 1e6

    # ── Regional statistics ──────────────────────────────────────────────
    regions = [
        (R_OUT, OUT_START, OUT_END),
        (R_CORE, INV_START, INV_END),
        (R_IP, INNER_PROX_START, INNER_PROX_END),
        (R_ID, INNER_DIST_START, INNER_DIST_END),
    ]

    pop_means = {k: [] for k, _, _ in regions}
    pop_sems = {k: [] for k, _, _ in regions}
    for _, genome in datasets:
        for k, a, b in regions:
            m, s = region_stats(genome, STEP_BP, a, b)
            pop_means[k].append(m)
            pop_sems[k].append(s)

    labels = names + ["All"]
    bar_means, bar_sems = {}, {}
    for k, _, _ in regions:
        vals = np.array(pop_means[k])
        bar_means[k] = np.r_[vals, np.nanmean(vals)]
        bar_sems[k] = np.r_[
            pop_sems[k],
            np.nanstd(vals, ddof=1) / np.sqrt(len(vals)),
        ]

    # ── Per-population and "All" curves ──────────────────────────────────
    pop_curves = {n: np.nanmean(g, axis=0) for n, g in datasets}
    all_curve = np.nanmean(
        np.vstack(list(pop_curves.values())), axis=0
    )

    # ── Figure: 3×3 GridSpec ─────────────────────────────────────────────
    fig = plt.figure(figsize=(DOUBLE_COL, 3.6))
    gs = GridSpec(
        3, 3, figure=fig,
        height_ratios=[1.25, 0.78, 0.78],
        hspace=0.65, wspace=0.28,
    )

    # ── Top: bar chart ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(labels))
    w = 0.18

    bar_kw = dict(edgecolor="black", capsize=2, lw=0.4, error_kw=dict(lw=0.5))
    ax0.bar(x_pos - 1.5 * w, bar_means[R_OUT], w, yerr=bar_sems[R_OUT],
            color=COL_OUT, label=R_OUT, **bar_kw)
    ax0.bar(x_pos - 0.5 * w, bar_means[R_CORE], w, yerr=bar_sems[R_CORE],
            color=COL_CORE, label=R_CORE, **bar_kw)
    ax0.bar(x_pos + 0.5 * w, bar_means[R_IP], w, yerr=bar_sems[R_IP],
            color=COL_PROX, label=R_IP, **bar_kw)
    ax0.bar(x_pos + 1.5 * w, bar_means[R_ID], w, yerr=bar_sems[R_ID],
            color=COL_DIST, label=R_ID, **bar_kw)

    ax0.set_yscale("log")
    ax0.set_yticks([8e5, 9e5, 1e6, 1.25e6, 1.5e6])
    ax0.set_yticklabels(
        [r"$8 \times 10^5$", r"$9 \times 10^5$", r"$10^6$",
         r"$1.25 \times 10^6$", r"$1.5 \times 10^6$"]
    )
    ax0.yaxis.set_minor_locator(plt.NullLocator())
    ax0.set_ylabel("Mean TMRCA")
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(labels, fontsize=5)
    ax0.legend(frameon=False, ncol=4, fontsize=5, loc="upper left",
               bbox_to_anchor=(0.0, 1.35))
    ax0.grid(axis="y", which="major", linestyle="--", alpha=0.25)

    # ── Bottom: per-population + All panels ──────────────────────────────
    subplot_order = [
        ("Mali", (1, 0)),
        ("Burkina Faso", (1, 1)),
        ("Cameroon", (1, 2)),
        ("Uganda", (2, 0)),
        ("Ghana", (2, 1)),
        ("All", (2, 2)),
    ]
    inv_mask = (x_mb >= INV_START / 1e6) & (x_mb <= INV_END / 1e6)
    panel_idx = 1

    for name, (r, c) in subplot_order:
        if name != "All" and name not in pop_curves:
            continue
        ax = fig.add_subplot(gs[r, c])
        y = all_curve if name == "All" else pop_curves[name]

        ax.plot(x_mb, y, lw=0.4, alpha=0.5, color="silver")

        y_smooth = rolling_nanmean(y, ROLL)

        y_out = y_smooth.copy()
        y_out[inv_mask] = np.nan
        ax.plot(x_mb, y_out, lw=0.6, color="darkgray", alpha=0.7)

        y_in = y_smooth.copy()
        y_in[~inv_mask] = np.nan
        ax.plot(x_mb, y_in, lw=0.6, color="dodgerblue")

        draw_vlines(ax)

        ax.set_yscale("log")
        ax.set_ylim(1e5, 5e6)
        ax.set_title(name, loc="left")
        ax.grid(which="both", linestyle="--", alpha=0.15)

        if c == 0:
            ax.set_ylabel("Mean TMRCA")
        else:
            ax.tick_params(labelleft=False)
        if r == 2:
            ax.set_xlabel("Position (Mb)")
        else:
            ax.tick_params(labelbottom=False)

        panel_idx += 1

    fig.subplots_adjust(hspace=0.65, wspace=0.28)
    savefig(fig, "figure8", output_dir=args.output_dir)
    print("Done: figure8.pdf")


if __name__ == "__main__":
    main()
