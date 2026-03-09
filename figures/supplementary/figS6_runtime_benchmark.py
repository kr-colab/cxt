"""
Figure S6: Wall-clock runtime comparison between cxt, SMC++, and SINGER.

Reproduces the revision/figure_benchmark panel: SMC++ runtime decomposition
with CXT pairwise scaling and SINGER overlay, plus a relative runtime panel.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.lines import Line2D

from figures.paths import (
    BENCHMARK_VALIDATION_JSONL,
    SMCPP_TIMING_JSONL,
    SINGER_TIMING_PATH,
)

CXT_GPUS_TO_SHOW = [1, 3]


# ==================================================
# Data loaders
# ==================================================

def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_singer_data(path):
    with open(path, "r") as f:
        first_line = f.readline().strip()

    has_header = any(word in first_line.lower() for word in
                     ["rec_rate", "recomb", "mutation", "sample", "replicate", "runtime", "time"])

    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=0 if has_header else None,
        skiprows=1 if has_header else 0,
        names=["mutation_rate", "sample_size", "replicate", "col3", "col4", "col5", "runtime_sec"],
    )

    df["mutation_rate"] = pd.to_numeric(df["mutation_rate"], errors="coerce")
    df["sample_size"]   = pd.to_numeric(df["sample_size"], errors="coerce")
    df["replicate"]     = pd.to_numeric(df["replicate"], errors="coerce")
    df["runtime_sec"]   = pd.to_numeric(df["runtime_sec"], errors="coerce")

    df = df.dropna(subset=["mutation_rate", "sample_size", "runtime_sec"]).copy()
    df["sample_size"] = df["sample_size"].astype(int)

    n_hap = 2 * df["sample_size"]
    df["n_pairs"] = (n_hap * (n_hap - 1) // 2).astype(int)
    return df


def load_smcpp_runs(path):
    df = read_jsonl(path)
    runs = df[df["event_type"] == "benchmark_complete"].copy()
    if runs.empty:
        raise ValueError("No benchmark_complete rows found in SMC++ JSONL")

    tim = pd.json_normalize(runs["timings"]).add_prefix("timings.")
    runs = pd.concat(
        [runs.drop(columns=["timings"], errors="ignore").reset_index(drop=True),
         tim.reset_index(drop=True)],
        axis=1,
    )

    runs["n_pairs"]  = pd.to_numeric(runs.get("n_pairs", runs.get("sample_size")), errors="coerce")
    runs["prep_sec"] = pd.to_numeric(runs.get("timings.vcf2smc_all_pairs"), errors="coerce")
    runs["est_sec"]  = pd.to_numeric(runs.get("timings.smcpp_estimate"), errors="coerce")
    runs["dec_sec"]  = pd.to_numeric(runs.get("timings.smcpp_posterior", 0.0), errors="coerce").fillna(0.0)

    if "timings.posterior_total" in runs.columns:
        runs["dec_sec"] = pd.to_numeric(runs["timings.posterior_total"], errors="coerce").fillna(runs["dec_sec"])

    runs["total_sec"] = runs["prep_sec"] + runs["est_sec"] + runs["dec_sec"]
    runs["fixed_sec"] = runs["total_sec"] - runs["dec_sec"]
    runs = runs.dropna(subset=["n_pairs", "total_sec", "fixed_sec"]).copy()
    return runs


def make_block_label(blocks):
    blocks = list(blocks)
    n = len(blocks)
    block_len_mb = (blocks[0][1] - blocks[0][0]) / 0.1e6
    if np.isclose(block_len_mb, round(block_len_mb)):
        block_len_mb = int(round(block_len_mb))
    else:
        block_len_mb = round(block_len_mb, 2)
    word = "block" if n == 1 else "blocks"
    return f"{n} \u00d7 {block_len_mb} Mb {word}"


def load_cxt_agg(path, wanted="1 \u00d7 1 Mb block"):
    df = read_jsonl(path)
    df["block_label"] = df["blocks"].apply(make_block_label)
    df = df[df["block_label"] == wanted].copy()
    if df.empty:
        raise ValueError(f"No rows found for block_label == {wanted!r} in CXT JSONL")

    group_cols = ["num_devices", "num_pairs"]
    agg = (
        df.groupby(group_cols)["runtime_seconds"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    agg["std"] = (agg["max"] - agg["min"]) / 2.0
    return agg


# ==================================================
# Plotting helpers
# ==================================================

def sec_fmt(x, _):
    if not np.isfinite(x) or x <= 0:
        return ""
    x = float(x)
    if x < 60:
        return f"{x:.0f}s"
    if x < 3600:
        return f"{x/60:.0f}m"
    return f"{x/3600:.1f}h"


def mean_sd_by_x(df, xcol, ycol):
    g = df[[xcol, ycol]].dropna().groupby(xcol)[ycol]
    xs = np.array(sorted(g.groups.keys()), dtype=float)
    mu = np.array([g.get_group(x).mean() for x in xs], dtype=float)
    sd = np.array([g.get_group(x).std(ddof=1) if len(g.get_group(x)) > 1 else 0.0
                   for x in xs], dtype=float)
    return xs, mu, sd


def plot_series(ax, df, xcol, ycol, label, marker, color,
                point_alpha=0.18, line_alpha=1.0, band_alpha=0.12, z=4):
    ax.scatter(df[xcol], df[ycol], s=45, alpha=point_alpha,
               edgecolor="none", color=color, zorder=z - 2)
    xs, mu, sd = mean_sd_by_x(df, xcol, ycol)
    ax.plot(xs, mu, marker=marker, lw=2.8, ms=7.5, color=color,
            alpha=line_alpha, label=label, zorder=z)
    ax.fill_between(xs, np.maximum(mu - sd, 1e-6), mu + sd,
                    color=color, alpha=band_alpha, zorder=1)
    return xs, mu, sd


def overlay_cxt_pairwise_on_ax(ax, agg,
                               gpus_to_show=(1, 3),
                               colors_gpu=None,
                               linestyle=(0, (4, 2)),
                               band_alpha=0.10,
                               line_alpha=0.85):
    if colors_gpu is None:
        colors_gpu = {1: "dodgerblue", 3: "royalblue"}
    for g in sorted(set(gpus_to_show)):
        sg = agg[agg["num_devices"] == g].sort_values("num_pairs")
        if sg.empty:
            continue
        x = sg["num_pairs"].to_numpy()
        y = sg["mean"].to_numpy()
        s = sg["std"].to_numpy()

        ax.fill_between(x, np.maximum(y - s, 1e-6), y + s,
                        color=colors_gpu.get(g, "gray"), alpha=band_alpha, zorder=0)
        ax.plot(x, y,
                linestyle=linestyle, lw=2.2,
                color=colors_gpu.get(g, "gray"), alpha=line_alpha, zorder=2,
                label=f"CXT runtime ({g} GPU{'s' if g > 1 else ''})")


def overlay_singer_on_ax(ax, singer_df,
                         colors_mu=None,
                         linestyle=(0, (2, 1)),
                         band_alpha=0.08,
                         line_alpha=0.80):
    if colors_mu is None:
        colors_mu = {3.5e-09: "darkgreen", 1.3e-08: "forestgreen"}
    for mut in sorted(singer_df["mutation_rate"].unique()):
        sub = singer_df[singer_df["mutation_rate"] == mut].copy()
        g = sub.groupby("n_pairs")["runtime_sec"].agg(["mean", "std"]).reset_index().sort_values("n_pairs")
        x = g["n_pairs"].to_numpy()
        y = g["mean"].to_numpy()
        s = g["std"].fillna(0).to_numpy()

        c = colors_mu.get(mut, "darkgreen")
        ax.fill_between(x, np.maximum(y - s, 1e-6), y + s,
                        color=c, alpha=band_alpha, zorder=0)
        ax.plot(x, y,
                linestyle=linestyle, lw=2.2,
                color=c, alpha=line_alpha, zorder=2,
                label=f"SINGER (\u03bc={mut:.1e})")


# ==================================================
# Relative panel
# ==================================================

def compute_relative_df(singer_df, cxt_agg, gpus_to_show=(1, 3)):
    s_mean = (
        singer_df.groupby(["mutation_rate", "n_pairs"])["runtime_sec"]
        .mean()
        .reset_index()
        .rename(columns={"runtime_sec": "singer_mean"})
        .sort_values(["mutation_rate", "n_pairs"])
    )

    rel_rows = []
    for g in sorted(set(gpus_to_show)):
        cg = cxt_agg[cxt_agg["num_devices"] == g].sort_values("num_pairs")
        if cg.empty:
            continue

        x_c = cg["num_pairs"].to_numpy(dtype=float)
        y_c = cg["mean"].to_numpy(dtype=float)
        x_min, x_max = np.nanmin(x_c), np.nanmax(x_c)

        for mut, sub in s_mean.groupby("mutation_rate"):
            x_s = sub["n_pairs"].to_numpy(dtype=float)
            y_s = sub["singer_mean"].to_numpy(dtype=float)

            m = (x_s >= x_min) & (x_s <= x_max) & np.isfinite(y_s)
            if not np.any(m):
                continue

            x_use = x_s[m]
            y_s_use = y_s[m]
            y_c_interp = np.interp(x_use, x_c, y_c)

            fold = y_s_use / np.maximum(y_c_interp, 1e-12)
            rel_rows.append(pd.DataFrame({
                "mutation_rate": mut,
                "num_devices": g,
                "n_pairs": x_use.astype(int),
                "fold": fold,
                "log10_fold": np.log10(fold),
            }))

    if not rel_rows:
        return pd.DataFrame(columns=["mutation_rate", "num_devices", "n_pairs", "fold", "log10_fold"])
    return pd.concat(rel_rows, ignore_index=True).sort_values(
        ["mutation_rate", "num_devices", "n_pairs"]
    ).reset_index(drop=True)


def plot_relative_panel(ax_rel, rel,
                        gpus_to_show=(1, 3),
                        colors_mu=None,
                        ls_gpu=None,
                        mk_gpu=None):
    if colors_mu is None:
        colors_mu = {3.5e-09: "darkgreen", 1.3e-08: "forestgreen"}
    if ls_gpu is None:
        ls_gpu = {1: "-", 3: (0, (3, 2))}
    if mk_gpu is None:
        mk_gpu = {1: "o", 3: "s"}

    ax_rel.axhline(0.0, linestyle=":", alpha=0.6, linewidth=1.0, zorder=0)
    ax_rel.grid(True, which="both", linestyle=":", alpha=0.35)
    ax_rel.set_axisbelow(True)

    for mut in sorted(rel["mutation_rate"].unique()):
        c = colors_mu.get(mut, "darkgreen")
        for g in sorted(set(gpus_to_show)):
            sub = rel[(rel["mutation_rate"] == mut) & (rel["num_devices"] == g)].sort_values("n_pairs")
            if sub.empty:
                continue
            ax_rel.plot(
                sub["n_pairs"].to_numpy(),
                sub["log10_fold"].to_numpy(),
                color=c,
                linestyle=ls_gpu.get(g, "-"),
                marker=mk_gpu.get(g, "o"),
                ms=4.2,
                lw=1.9,
                alpha=0.95,
            )

    ax_rel.set_title("Relative runtime", fontsize=11)
    ax_rel.set_xlabel("# pairs", fontsize=10)
    ax_rel.set_ylabel("log$_{10}$(SINGER / CXT)", fontsize=10)

    y = rel["log10_fold"].to_numpy(dtype=float)
    if y.size:
        pad = 0.15 * (np.nanmax(y) - np.nanmin(y) + 1e-12)
        lo = np.floor((np.nanmin(y) - pad) * 2) / 2
        hi = np.ceil((np.nanmax(y) + pad) * 2) / 2
        ax_rel.set_ylim(lo, hi)

    mu_handles = [
        Line2D([0], [0], color=colors_mu.get(mut, "darkgreen"), lw=2.2,
               label=f"SINGER \u03bc={mut:.1e}")
        for mut in sorted(rel["mutation_rate"].unique())
    ]
    gpu_handles = [
        Line2D([0], [0], color="gray", lw=2.2,
               linestyle=ls_gpu.get(g, "-"),
               marker=mk_gpu.get(g, "o"),
               ms=5, label=f"{g} GPU")
        for g in sorted(set(gpus_to_show))
    ]

    leg1 = ax_rel.legend(handles=mu_handles, loc="upper left",
                         fontsize=9.0, frameon=True, framealpha=0.92)
    ax_rel.add_artist(leg1)
    ax_rel.legend(handles=gpu_handles, loc="upper right",
                  fontsize=9.0, frameon=True, framealpha=0.92)


# ==================================================
# Main
# ==================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cxt-data", default=BENCHMARK_VALIDATION_JSONL)
    parser.add_argument("--smcpp-data", default=SMCPP_TIMING_JSONL)
    parser.add_argument("--singer-data", default=SINGER_TIMING_PATH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for label, path in [("CXT", args.cxt_data), ("SMC++", args.smcpp_data), ("SINGER", args.singer_data)]:
        if not path or not os.path.exists(path):
            print(f"{label} data not found at: {path}")
            return

    runs      = load_smcpp_runs(args.smcpp_data)
    cxt_agg   = load_cxt_agg(args.cxt_data, wanted="1 \u00d7 1 Mb block")
    singer_df = load_singer_data(args.singer_data)

    # --- Figure: main + relative panel ---
    fig = plt.figure(figsize=(12.0, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.9, 2.4], wspace=0.25)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_rel = fig.add_subplot(gs[0, 1])

    # --- Left: SMC++ decomposition + CXT + SINGER ---
    ax_top.set_yscale("log")
    ax_top.grid(True, which="both", linestyle=":", alpha=0.35)
    ax_top.set_axisbelow(True)

    plot_series(ax_top, runs, "n_pairs", "total_sec",
                "SMC++ Total", marker="o", color="dodgerblue")
    plot_series(ax_top, runs, "n_pairs", "fixed_sec",
                "SMC++ Prep+estimate (total\u2212decode)",
                marker="s", color="steelblue",
                point_alpha=0.14, line_alpha=0.95, band_alpha=0.10, z=4)
    plot_series(ax_top, runs, "n_pairs", "dec_sec",
                "SMC++ Decode",
                marker="D", color="lightskyblue",
                point_alpha=0.10, line_alpha=0.75, band_alpha=0.07, z=4)

    overlay_cxt_pairwise_on_ax(ax_top, cxt_agg, gpus_to_show=CXT_GPUS_TO_SHOW)
    overlay_singer_on_ax(ax_top, singer_df)

    ax_top.set_xlabel("Number of pairs")
    ax_top.set_ylabel("Runtime")
    ax_top.yaxis.set_major_formatter(FuncFormatter(sec_fmt))
    ax_top.yaxis.set_major_locator(LogLocator(base=10))
    ax_top.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax_top.yaxis.set_minor_formatter(NullFormatter())
    ax_top.set_title("SMC++ runtime decomposition + CXT pairwise scaling + SINGER",
                     fontsize=12)

    xmax_all = max(
        float(np.nanmax(runs["n_pairs"])),
        float(np.nanmax(cxt_agg[cxt_agg["num_devices"].isin(CXT_GPUS_TO_SHOW)]["num_pairs"])),
        float(np.nanmax(singer_df["n_pairs"])),
    )
    xmin_all = min(
        float(np.nanmin(runs["n_pairs"])),
        float(np.nanmin(cxt_agg[cxt_agg["num_devices"].isin(CXT_GPUS_TO_SHOW)]["num_pairs"])),
        float(np.nanmin(singer_df["n_pairs"])),
    )
    ax_top.set_xlim(max(0, xmin_all - 25), xmax_all * 1.05)
    ax_top.set_ylim(1, 20000)

    h_top, l_top = ax_top.get_legend_handles_labels()
    ax_top.legend(
        h_top, l_top,
        loc="lower right", ncol=2, fontsize=9.0,
        frameon=True, framealpha=0.92, borderpad=0.5,
        labelspacing=0.35, handlelength=2.0, handletextpad=0.6,
        columnspacing=0.9,
    )

    # --- Right: relative panel ---
    rel = compute_relative_df(singer_df, cxt_agg, gpus_to_show=CXT_GPUS_TO_SHOW)
    plot_relative_panel(ax_rel, rel, gpus_to_show=CXT_GPUS_TO_SHOW)
    ax_rel.set_xlim(ax_top.get_xlim())

    fig.tight_layout()
    out = os.path.join(args.output_dir, "figS6_runtime_benchmark.png")
    fig.savefig(out, dpi=260)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
