"""
Figure 8: Coalescent-time structure across the In(2L)a inversion on chr2L.

Top: bar chart comparing mean TMRCA across four regions — Outside (10–20 Mb),
Core interval, Inner proximal (+1.0 Mb), Inner distal (-1.0 Mb) — for each
population plus "All". Bottom: 2×3 grid of line plots with gray background
and blue line highlighting the core interval (~20–42 Mb).

Reads genome-wide TMRCA caches produced by fig7_mosquito_rdl.py.

Matches paper Figure 8 layout, colors, and region definitions.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, FuncFormatter


STEP_BP = 200

INV_START = 20_524_058
INV_END = 42_165_532

OUT_START = 10_000_000
OUT_END = 20_000_000

BP_WINDOW = 500_000
PROX_BP_START = INV_START - BP_WINDOW
PROX_BP_END = INV_START + BP_WINDOW
DIST_BP_START = INV_END - BP_WINDOW
DIST_BP_END = INV_END + BP_WINDOW

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
POP_LABELS = {"Mali": "Mali", "BurkinaFaso": "Burkina Faso",
              "Cameroon": "Cameroon", "Uganda": "Uganda", "Ghana": "Ghana"}

# Paper Fig 8: Inner proximal/distal regions
R_OUT = "Outside (10–20 Mb)"
R_CORE = "Core interval"
R_IP = "Inner proximal (+1.0 Mb)"
R_ID = "Inner distal (-1.0 Mb)"


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
    ax.axvline(INV_START / 1e6, color=LINE_CORE, lw=1.8, alpha=0.25)
    ax.axvline(INV_END / 1e6, color=LINE_CORE, lw=1.8, alpha=0.25)
    ax.axvline(INV_START / 1e6, color=LINE_BP, lw=1.2, ls=":")
    ax.axvline(INV_END / 1e6, color=LINE_BP, lw=1.2, ls=":")
    for bp in [PROX_BP_START, PROX_BP_END, DIST_BP_START, DIST_BP_END]:
        ax.axvline(bp / 1e6, color=LINE_BP, lw=0.8, ls="--", alpha=0.15)
    for bp in [INNER_PROX_START, INNER_PROX_END, INNER_DIST_START, INNER_DIST_END]:
        ax.axvline(bp / 1e6, color=LINE_BP, lw=1.0, ls="--", alpha=0.35)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig7",
                        help="Cache dir with genome-wide TMRCA from fig7")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load genome caches (already in LINEAR space from fig7) ---
    datasets_raw = []
    for pop_key in POP_ORDER:
        cache_path = os.path.join(args.cache_dir, f"genome_{pop_key.lower()}.npz")
        if not os.path.exists(cache_path):
            print(f"Skipping {pop_key}: cache not found at {cache_path}")
            continue
        genome = _as_2d(np.load(cache_path)["genome"])
        print(f"  {POP_LABELS[pop_key]}: {genome.shape[0]} replicates")
        datasets_raw.append((POP_LABELS[pop_key], genome))

    if not datasets_raw:
        print("No genome caches found. Run fig7_mosquito_rdl.py first.")
        return

    names = [n for n, _ in datasets_raw]
    arrays = [a for _, a in datasets_raw]
    arrays, W = harmonize_lengths(arrays)
    datasets = list(zip(names, arrays))
    x_mb = (np.arange(W) * STEP_BP) / 1e6

    # --- Regional statistics (paper: Inner proximal/distal) ---
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
        bar_sems[k] = np.r_[pop_sems[k],
                             np.nanstd(vals, ddof=1) / np.sqrt(len(vals))]

    # --- Per-population and "All" curves ---
    pop_curves = {n: np.nanmean(g, axis=0) for n, g in datasets}
    all_curve = np.nanmean(np.vstack(list(pop_curves.values())), axis=0)

    # --- Figure: 3x3 GridSpec ---
    fig = plt.figure(figsize=(14.5, 8.5))
    gs = GridSpec(3, 3, figure=fig,
                  height_ratios=[1.25, 0.78, 0.78], hspace=0.38, wspace=0.25)

    # ---- Top: bar chart ----
    ax0 = fig.add_subplot(gs[0, :])
    x = np.arange(len(labels))
    w = 0.18

    ax0.bar(x - 1.5 * w, bar_means[R_OUT], w, yerr=bar_sems[R_OUT],
            color=COL_OUT, edgecolor="black", capsize=4, label=R_OUT)
    ax0.bar(x - 0.5 * w, bar_means[R_CORE], w, yerr=bar_sems[R_CORE],
            color=COL_CORE, edgecolor="black", capsize=4, label=R_CORE)
    ax0.bar(x + 0.5 * w, bar_means[R_IP], w, yerr=bar_sems[R_IP],
            color=COL_PROX, edgecolor="black", capsize=4, label=R_IP)
    ax0.bar(x + 1.5 * w, bar_means[R_ID], w, yerr=bar_sems[R_ID],
            color=COL_DIST, edgecolor="black", capsize=4, label=R_ID)

    ax0.set_yscale("log")
    ax0.set_ylabel("Mean TMRCA (generations)")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=25, ha="right")
    ax0.legend(frameon=False, ncol=4, loc="upper center", fontsize=8.5)
    ax0.grid(axis="y", which="both", linestyle="--", alpha=0.35)

    bar_ticks = [6e5, 7e5, 8e5, 9e5, 1e6, 1.5e6]
    ax0.set_yticks(bar_ticks)
    ax0.set_yticks([], minor=True)
    ax0.set_yticklabels([
        r"$6 \times 10^5$", r"$7 \times 10^5$", r"$8 \times 10^5$",
        r"$9 \times 10^5$", r"$10^6$", r"$1.5 \times 10^6$",
    ])

    # ---- Bottom: per-population + All panels ----
    subplot_order = [
        ("Mali", (1, 0)),
        ("Burkina Faso", (1, 1)),
        ("Cameroon", (1, 2)),
        ("Uganda", (2, 0)),
        ("Ghana", (2, 1)),
        ("All", (2, 2)),
    ]

    inv_mask = (x_mb >= INV_START / 1e6) & (x_mb <= INV_END / 1e6)

    for name, (r, c) in subplot_order:
        if name != "All" and name not in pop_curves:
            continue
        ax = fig.add_subplot(gs[r, c])
        y = all_curve if name == "All" else pop_curves[name]

        ax.plot(x_mb, y, lw=0.7, alpha=0.6, color="silver")

        if name != "All" and name in dict(datasets):
            g = dict(datasets)[name]
            mu = np.nanmean(g, axis=0)
            sd = np.nanstd(g, axis=0)
            ax.fill_between(x_mb, mu - sd, mu + sd,
                            color="lightgrey", alpha=0.35, linewidth=0)

        y_smooth = rolling_nanmean(y, ROLL)

        y_out = y_smooth.copy()
        y_out[inv_mask] = np.nan
        ax.plot(x_mb, y_out, lw=1.5, color="darkgray", alpha=0.8)

        y_in = y_smooth.copy()
        y_in[~inv_mask] = np.nan
        ax.plot(x_mb, y_in, lw=1.5, color="dodgerblue")

        draw_vlines(ax)

        ax.set_yscale("log")
        ax.set_ylim(1e5, 5e6)
        ax.set_title(name, fontsize=11)
        ax.grid(which="both", linestyle="--", alpha=0.22)

        if c == 0:
            ax.set_ylabel("Mean TMRCA")
        else:
            ax.tick_params(labelleft=False)
        if r == 2:
            ax.set_xlabel("Position (Mb)")
        else:
            ax.tick_params(labelbottom=False)

    fig.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(args.output_dir, "figure8_inversion.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
