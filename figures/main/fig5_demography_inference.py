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
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import stdpopsim
import torch
import tskit

import cxt
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import coalescence_rates
from figures.paths import SINGER_BASE
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

MCMC_REPS = np.arange(50, 100, 5)


# ---------------------------------------------------------------------------
# Singer helpers
# ---------------------------------------------------------------------------

def _load_polegon_trees(path):
    tc = tskit.TableCollection.load(path)
    tc.mutations.clear()
    return tc.tree_sequence()


def _interpolate_robust(ts, window_size, sequence_length, a, b):
    windows = np.linspace(0, sequence_length, int(sequence_length // window_size) + 1)
    windows[-1] = ts.sequence_length
    tmrca = ts.diversity(sample_sets=[(a, b)], windows=windows, mode="branch") / 2
    return tmrca.T[0]


def load_singer_tmrcas(species_key, cache_dir, window_size=2e3, seq_length=10e6):
    """Load or compute Singer TMRCAs (flattened, linear-scale) for one species."""
    cache_path = os.path.join(cache_dir, f"singer_demography_{species_key}.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        if "singer_tmrcas_flat" in d:
            return d["singer_tmrcas_flat"]

    polegon_dir = os.path.join(SINGER_BASE, "demographic-inference/polegon-output")
    num_pairs = 25
    all_reps = []
    for rep in MCMC_REPS:
        fp = os.path.join(polegon_dir, f"{species_key}.{rep}.polegon.trees")
        print(f"  Singer: loading {os.path.basename(fp)}")
        ts = _load_polegon_trees(fp)
        tmrcas = []
        for i in range(num_pairs):
            for j in range(i + 1, num_pairs):
                tmrcas.append(_interpolate_robust(ts, window_size, seq_length, i, j))
        all_reps.append(np.array(tmrcas))

    flat = np.array(all_reps).mean(0).flatten()
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, singer_tmrcas_flat=flat)
    return flat


# ---------------------------------------------------------------------------
# SMC++ helpers
# ---------------------------------------------------------------------------

def smcpp_coalrate_from_model(model_json_path, time_grid):
    """Read an SMC++ model.final.json and return coalescence rate on *time_grid*."""
    with open(model_json_path) as f:
        model = json.load(f)["model"]
    N0 = float(model["N0"])
    knots_coal = np.array(model["knots"], dtype=float)
    y = np.array(model["y"], dtype=float)

    t_knots = knots_coal * 2.0 * N0
    log_N = np.log(N0) + y

    log_N_grid = np.interp(time_grid, t_knots, log_N,
                           left=log_N[0], right=log_N[-1])
    return 1.0 / (2.0 * np.exp(log_N_grid))


def run_smcpp_species(ts, mu, cache_dir, species_key, sif_path=None):
    """Run SMC++ estimate for one species and return model JSON path."""
    work_dir = os.path.join(cache_dir, f"smcpp_{species_key}")
    model_json = os.path.join(work_dir, "model.final.json")

    if os.path.exists(model_json):
        return model_json

    pairs = [(i, i + 1) for i in range(0, min(ts.num_samples, 24), 2)]
    result = analyze_ts_with_smcpp_multi(
        ts, pairs=pairs, mu=mu, tmp_dir=work_dir,
        do_posterior=False, sif_path=sif_path,
    )
    return result["model_json"]


# ---------------------------------------------------------------------------
# CXT core
# ---------------------------------------------------------------------------

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
        tmrca, index_map = cxt.translate(
            ts, model, pivot_pairs=pivot_pairs,
            blocks=blocks, devices=args.devices,
            B_per_device=128, B=128,
            build_workers=32, mutation_rate=cfg["mutation_rate"],
        )
        np.savez_compressed(tmrca_path, tmrca=tmrca)

    tmrca_flat = np.exp(tmrca.flatten())
    yhat_coalrate = coalescence_rates(tmrca_flat, time_windows)

    return {
        "ts": ts,
        "fine_time_grid": fine_time_grid,
        "coalrate_ck": coalrate_ck,
        "time_windows": time_windows,
        "ytrue_coalrate": ytrue_coalrate,
        "yhat_coalrate": yhat_coalrate,
        "title": cfg["title"],
        "description": demogr.description,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig5")
    parser.add_argument("--singer-cache-dir", default="figures/output/main/cache/singer")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--sif-path", default=None)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.singer_cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    model = cxt.load_model("broad", device="cpu")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, cfg in zip(axes, SPECIES_CONFIGS):
        res = run_species(cfg, model, args)
        tw = res["time_windows"]

        ax.plot(res["fine_time_grid"], res["coalrate_ck"], "-",
                color="black", label="Expectation")
        ax.step(tw[:-1], res["ytrue_coalrate"], where="post",
                color="firebrick", label="Inference limit")
        ax.step(tw[:-1], res["yhat_coalrate"], where="post",
                color="dodgerblue", label="cxt")

        # Singer+Polegon
        try:
            singer_flat = load_singer_tmrcas(
                cfg["key"], args.singer_cache_dir,
            )
            singer_cr = coalescence_rates(singer_flat, tw)
            ax.step(tw[:-1], singer_cr, where="post", color="darkblue",
                    label="Singer+Polegon", linestyle="--")
        except Exception as e:
            print(f"  Singer skipped for {cfg['key']}: {e}")

        # SMC++
        try:
            model_json = run_smcpp_species(
                res["ts"], cfg["mutation_rate"],
                args.cache_dir, cfg["key"], sif_path=args.sif_path,
            )
            smcpp_cr = smcpp_coalrate_from_model(model_json, res["fine_time_grid"])
            ax.plot(res["fine_time_grid"], smcpp_cr, "-",
                    color="cyan", alpha=0.9, label="SMC++")
        except Exception as e:
            print(f"  SMC++ skipped for {cfg['key']}: {e}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time (Generations)")
        desc = res["description"].split("(")[0].rstrip()
        ax.set_title(f"{res['title']}\n{desc}", loc="left")
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
