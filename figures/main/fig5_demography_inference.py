"""
Figure 5: Inverse-instantaneous coalescence rate (demography estimation)
for H. sapiens (Zigzag), B. taurus (HolsteinFriesian), and A. thaliana (SMA).

Each panel shows the expected coalescence rate, inference limit,
cxt prediction, Singer+Polegon, and SMC++ model curves.
"""

import argparse
import json
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import stdpopsim
import torch
import tskit

from cxt.api2 import translate
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import coalescence_rates, setup_cxt_model
from figures.utils import analyze_ts_with_smcpp_multi


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


def run_species(cfg, model, args):
    """Simulate, infer, and compute coalescence rates for one species."""
    num_pairs = 25
    num_blocks = 10
    window_size = 2e3
    num_time_windows = 40

    species = stdpopsim.get_species(cfg["species"])
    demogr = species.get_demographic_model(cfg["demography"])
    contig = species.get_contig("chr1", right=cfg["seq_length"])
    pop_name = demogr.populations[0].name
    engine = stdpopsim.get_engine("msprime")

    ts_path = os.path.join(args.cache_dir, f"{cfg['key']}.ts")
    if os.path.exists(ts_path):
        ts = tskit.load(ts_path)
    else:
        ts = engine.simulate(
            contig=contig, samples={pop_name: num_pairs},
            demographic_model=demogr, seed=cfg["seed"],
        ).trim()
        ts.dump(ts_path)

    pivot_pairs = [(i, j) for i in range(num_pairs) for j in range(i + 1, num_pairs)]
    blocks = [(int(i), int(i + 1e6)) for i in np.linspace(0, num_blocks * 1e6 - 1e6, num_blocks)]

    true_path = os.path.join(args.cache_dir, f"{cfg['key']}_true_tmrcas.npy")
    if os.path.exists(true_path):
        true_tmrcas = np.load(true_path)
    else:
        true_tmrcas = []
        for i in range(num_pairs):
            for j in range(i + 1, num_pairs):
                true_tmrcas.append(interpolate_tmrcas(ts, window_size, cfg["seq_length"], i, j))
        true_tmrcas = np.array(true_tmrcas)
        np.save(true_path, true_tmrcas)

    time_windows = np.logspace(2, np.floor(np.log10(ts.max_time)), num_time_windows + 1)
    time_windows[0] = 0.0
    fine_time_grid = np.logspace(2, np.floor(np.log10(ts.max_time)), 1000)
    coalrate_ck, _ = demogr.model.debug().coalescence_rate_trajectory(
        lineages={pop_name: 2}, steps=fine_time_grid,
    )
    ytrue_coalrate = coalescence_rates(true_tmrcas.flatten(), time_windows)

    tmrca_path = os.path.join(args.cache_dir, f"tmrca_{cfg['key']}.npz")
    if os.path.exists(tmrca_path):
        tmrca = np.load(tmrca_path)["tmrca"]
    else:
        tmrca, index_map = translate(
            input_data=ts, data_type="ts",
            model=model, pivot_pairs=pivot_pairs,
            blocks=blocks, devices=args.devices,
            B_per_device=128, B=128,
            build_workers=32, mutation_rate=cfg["mutation_rate"],
        )
        np.savez_compressed(tmrca_path, tmrca=tmrca)

    tmrca_flat = np.exp(tmrca.flatten())
    yhat_coalrate = coalescence_rates(tmrca_flat, time_windows)

    return {
        "fine_time_grid": fine_time_grid,
        "coalrate_ck": coalrate_ck,
        "time_windows": time_windows,
        "ytrue_coalrate": ytrue_coalrate,
        "yhat_coalrate": yhat_coalrate,
        "title": cfg["title"],
        "description": demogr.description,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig5")
    parser.add_argument("--devices", nargs="+", default=None)
    args = parser.parse_args()

    if args.devices is None:
        args.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = setup_cxt_model(model_type="broad")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, cfg in zip(axes, SPECIES_CONFIGS):
        res = run_species(cfg, model, args)

        ax.plot(res["fine_time_grid"], res["coalrate_ck"], "-", color="black", label="Expectation")
        ax.step(res["time_windows"][:-1], res["ytrue_coalrate"], where="post",
                color="firebrick", label="Inference limit")
        ax.step(res["time_windows"][:-1], res["yhat_coalrate"], where="post",
                color="dodgerblue", label="cxt")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time (Generations)")
        ax.set_title(f"{res['title']}\n{res['description']}", loc="left")
        ax.set_ylim(10e-7, 5e-5)
        ax.grid(True)

    axes[0].set_ylabel("IICR / 2")
    axes[-1].legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    out = os.path.join(args.output_dir, "figure5_demography.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
