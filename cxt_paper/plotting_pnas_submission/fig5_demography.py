"""
Figure 5 (PNAS): IICR demography inference for H. sapiens, A. thaliana, B. taurus.

Loads cached tree sequences, TMRCA arrays, Singer, and SMC++ model fits.
Three panels showing expected coalescence rate, inference limit,
cxt / Singer+Polegon / SMC++ predictions.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import stdpopsim
import tskit

from pnas_defaults import (
    apply_pnas_style, savefig, resolve_cache,
    DOUBLE_COL, DEFAULT_OUTPUT,
    CXT_BLUE, SINGER_NAVY, SMCPP_CYAN, TRUE_BLACK,
)


def coalescence_rates(ancestor_times, time_windows, epsilon=1e-3):
    """Pair coalescence rates from ancestor times in time windows."""
    num_windows = time_windows.size - 1
    idx = np.digitize(ancestor_times, time_windows) - 1
    pdf = np.bincount(idx[idx < num_windows], minlength=num_windows)
    pdf = pdf / ancestor_times.size
    cdf = np.append(0, np.cumsum(pdf))
    survival = np.append(1.0 - cdf, 0.0)
    last = np.min(np.flatnonzero(survival < epsilon))
    log_survival = np.log(survival[:last])
    tw = time_windows[:last]
    rates = (log_survival[:-1] - log_survival[1:]) / np.diff(tw)
    if rates.size < num_windows:
        rates = np.append(
            rates,
            1 / np.mean(ancestor_times[idx == last - 1] - tw[-1])
        )
    return np.append(rates, [np.nan] * (num_windows - rates.size))

SPECIES_CONFIGS = [
    {
        "key": "homsap",
        "species": "HomSap",
        "demography": "Zigzag_1S14",
        "mutation_rate": 1.29e-8,
        "title": r"$\mathit{H.\;sapiens}$",
        "seq_length": 10e6,
        "seed": int(10e6),
    },
    {
        "key": "aratha",
        "species": "AraTha",
        "demography": "SouthMiddleAtlas_1D17",
        "mutation_rate": 7e-9,
        "title": r"$\mathit{A.\;thaliana}$",
        "seq_length": 10e6,
        "seed": 10_000_000,
    },
    {
        "key": "bostau",
        "species": "BosTau",
        "demography": "HolsteinFriesian_1M13",
        "mutation_rate": 1.2e-8,
        "title": r"$\mathit{B.\;taurus}$",
        "seq_length": 10e6,
        "seed": 10_337_0001,
    },
]

NUM_TIME_WINDOWS = 40


def smcpp_coalrate_from_model(model_json_path, time_grid):
    """Read an SMC++ model.final.json and return coalescence rate on *time_grid*."""
    with open(model_json_path) as f:
        model = json.load(f)["model"]
    N0 = float(model["N0"])
    t_knots = np.array(model["knots"], dtype=float) * 2.0 * N0
    log_N = np.log(N0) + np.array(model["y"], dtype=float)
    log_N_grid = np.interp(time_grid, t_knots, log_N,
                           left=log_N[0], right=log_N[-1])
    return 1.0 / (2.0 * np.exp(log_N_grid))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_pnas_style()

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.0))

    for idx, (ax, cfg) in enumerate(zip(axes, SPECIES_CONFIGS)):
        key = cfg["key"]

        species = stdpopsim.get_species(cfg["species"])
        demogr = species.get_demographic_model(cfg["demography"])
        pop_name = demogr.populations[0].name

        # Tree sequence (for max_time)
        ts_path = resolve_cache(f"main/cache/fig5/{key}.ts")
        if not os.path.exists(ts_path):
            print(f"  Tree sequence not found for {key}: {ts_path}")
            ax.text(0.5, 0.5, "Cache missing", ha="center", va="center",
                    fontsize=6, alpha=0.5, transform=ax.transAxes)
            ax.set_title(cfg["title"], loc="left")
            continue

        ts = tskit.load(ts_path)
        max_time = ts.max_time

        time_windows = np.logspace(
            2, np.floor(np.log10(max_time)), NUM_TIME_WINDOWS + 1)
        time_windows[0] = 0.0
        fine_time_grid = np.logspace(
            2, np.floor(np.log10(max_time)), 1000)

        # 1. Expected coalescence rate from demographic model
        coalrate_ck, _ = demogr.model.debug().coalescence_rate_trajectory(
            lineages={pop_name: 2}, steps=fine_time_grid,
        )
        ax.plot(fine_time_grid, coalrate_ck, "-", color=TRUE_BLACK,
                lw=0.8, label="Expectation")

        # 2. Inference limit (true TMRCAs)
        true_path = resolve_cache(f"main/cache/fig5/{key}_true_tmrcas.npy")
        if os.path.exists(true_path):
            true_tmrcas = np.load(true_path)
            ytrue_cr = coalescence_rates(true_tmrcas.flatten(), time_windows)
            ax.step(time_windows[:-1], ytrue_cr, where="post",
                    color="firebrick", lw=0.8, label="Inference limit")

        # 3. cxt predictions
        tmrca_path = resolve_cache(f"main/cache/fig5/tmrca_{key}.npz")
        if os.path.exists(tmrca_path):
            tmrca = np.load(tmrca_path)["tmrca"]
            yhat_cr = coalescence_rates(np.exp(tmrca.flatten()), time_windows)
            ax.step(time_windows[:-1], yhat_cr, where="post",
                    color=CXT_BLUE, lw=0.8, label="cxt")

        # 4. Singer+Polegon
        singer_path = resolve_cache(
            f"main/cache/singer/singer_demography_{key}.npz")
        if os.path.exists(singer_path):
            sd = np.load(singer_path)
            if "singer_tmrcas_flat" in sd:
                singer_cr = coalescence_rates(
                    sd["singer_tmrcas_flat"], time_windows)
                ax.step(time_windows[:-1], singer_cr, where="post",
                        color=SINGER_NAVY, lw=0.8, ls="--",
                        label="Singer+Polegon")

        # 5. SMC++
        smcpp_json = resolve_cache(
            f"main/cache/fig5/smcpp_{key}/model.final.json")
        if os.path.exists(smcpp_json):
            smcpp_cr = smcpp_coalrate_from_model(smcpp_json, fine_time_grid)
            ax.plot(fine_time_grid, smcpp_cr, "-", color=SMCPP_CYAN,
                    lw=0.8, alpha=0.9, label="SMC++")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time (gen.)")
        desc = demogr.description.split("(")[0].rstrip()
        ax.set_title(f"{cfg['title']}\n{desc}", loc="left", fontsize=6)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("IICR / 2")
    axes[-1].legend(loc="upper right", fontsize=5, framealpha=0.85)
    plt.tight_layout()
    savefig(fig, "figure5", output_dir=args.output_dir)
    print("Done: figure5")


if __name__ == "__main__":
    main()
