"""Download every registered model to .verification/checkpoints and produce
a verification figure for each one comparing predicted vs true TMRCA."""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

os.environ["CXT_CHECKPOINT_CACHE"] = CACHE_DIR
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cxt
from cxt.checkpoint import CHECKPOINT_REGISTRY, load_model
from cxt.utils import simulate_parameterized_tree_sequence
from cxt.preprocess import interpolate_tmrcas
from cxt.translate import generate, to_log_times

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_REPS = 10

MODEL_CONFIGS = {
    "broad":       {"seq_len": 1_000_000, "samples": 25, "adapter": False},
    "narrow":      {"seq_len": 1_000_000, "samples": 25, "adapter": False},
    "residual":    {"seq_len": 1_000_000, "samples": 25, "adapter": False},
    "broad_w200":  {"seq_len":   100_000, "samples": 25, "adapter": False, "Ne": 5e4},
    "w200_wmissing":        {"seq_len": 100_000, "samples": 25, "adapter": False, "Ne": 5e4},
    "broad+adapter":        {"seq_len": 1_000_000, "samples": 5, "adapter": True},
    "w200_wmissing_adapter": {"seq_len": 100_000,  "samples": 5, "adapter": True},
}

PIVOT_PAIRS_STANDARD = [(0, 1), (2, 3)]
PIVOT_PAIRS_ADAPTER = [(0, 1), (2, 3)]


def simulate_and_cache(samples, seq_len, seed=SEED, Ne=2e4):
    ts = simulate_parameterized_tree_sequence(
        seed=seed, samples=samples, sequence_length=seq_len,
        population_size=Ne,
    )
    return ts


def compute_true_tmrcas(ts, pairs, seq_len):
    window_size = seq_len // 500
    trues = []
    for a, b in pairs:
        t = interpolate_tmrcas(ts, window_size, seq_len, a, b)
        trues.append(np.log(t))
    return np.array(trues)


def run_base_model(model_name, cfg):
    seq_len = cfg["seq_len"]
    samples = cfg["samples"]
    Ne = cfg.get("Ne", 2e4)
    pairs = PIVOT_PAIRS_STANDARD

    print(f"  Simulating ts (samples={samples}, seq_len={seq_len:,}, Ne={Ne:.0f})...")
    ts = simulate_and_cache(samples, seq_len, Ne=Ne)

    print(f"  Loading model...")
    model = load_model(model_name, device=DEVICE)

    missingness_bitmask = None
    if "wmissing" in model_name:
        missingness_bitmask = np.zeros(seq_len, dtype=bool)

    print(f"  Running inference ({N_REPS} reps, {len(pairs)} pairs)...")
    tmrca, index_map = cxt.translate(
        ts, model,
        blocks=[(0, seq_len)],
        pivot_pairs=pairs,
        devices=[DEVICE],
        B=len(pairs), B_per_device=len(pairs),
        n_reps=N_REPS,
        base_seed=SEED,
        top_k=50,
        cache_matching=True,
        progress=False,
        decode_bar=False,
        build_workers=1,
        missingness_bitmask=missingness_bitmask,
    )

    print(f"  Computing true TMRCAs...")
    true_tmrcas = compute_true_tmrcas(ts, pairs, seq_len)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tmrca, true_tmrcas, pairs, seq_len


def run_adapter_model(model_name, cfg):
    seq_len = cfg["seq_len"]
    samples = cfg["samples"]
    Ne = cfg.get("Ne", 2e4)
    pairs = PIVOT_PAIRS_ADAPTER

    print(f"  Simulating ts (samples={samples}, seq_len={seq_len:,}, Ne={Ne:.0f})...")
    ts = simulate_and_cache(samples, seq_len, Ne=Ne)

    print(f"  Loading adapter model...")
    wrapped = load_model(model_name, device=DEVICE)
    backbone = wrapped.backbone
    adapter = wrapped.adapter

    print(f"  Running inference ({N_REPS} reps, {len(pairs)} pairs)...")
    tmrca, index_map = cxt.translate(
        ts, backbone,
        blocks=[(0, seq_len)],
        pivot_pairs=pairs,
        devices=[DEVICE],
        B=len(pairs), B_per_device=len(pairs),
        n_reps=N_REPS,
        base_seed=SEED,
        top_k=50,
        cache_matching=True,
        progress=False,
        decode_bar=False,
        build_workers=1,
        adapter=adapter,
    )

    print(f"  Computing true TMRCAs...")
    true_tmrcas = compute_true_tmrcas(ts, pairs, seq_len)

    del wrapped, backbone, adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tmrca, true_tmrcas, pairs, seq_len


def plot_model(model_name, tmrca, true_tmrcas, pairs, seq_len, elapsed):
    """Create and save a verification figure for one model."""
    window_size = seq_len // 500
    x = np.arange(0, seq_len, window_size) / 1e6 if seq_len >= 1e6 else np.arange(0, seq_len, window_size) / 1e3

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 3.2 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for i, (a, b) in enumerate(pairs):
        ax = axes[i]
        pred = tmrca[:, i, :]
        pred_mean = pred.mean(axis=0)
        pred_std = pred.std(axis=0)
        ytrue = true_tmrcas[i]

        ax.fill_between(x, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std,
                        alpha=0.22, color="C0", label="Predicted +/- 2s")
        ax.plot(x, pred_mean, color="C0", linewidth=1.1, label="Predicted (mean)")
        ax.plot(x, ytrue, color="C3", linewidth=0.9, alpha=0.8, label="True TMRCA")

        mse = float(np.mean((pred_mean - ytrue) ** 2))
        corr = float(np.corrcoef(pred_mean, ytrue)[0, 1])
        ax.set_title(f"Pair ({a},{b})  MSE={mse:.3f}  r={corr:.3f}", fontsize=11)
        ax.set_ylabel("log TMRCA", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    x_label = "Genomic position (Mb)" if seq_len >= 1e6 else "Genomic position (kb)"
    axes[-1].set_xlabel(x_label, fontsize=10)

    safe_name = model_name.replace("+", "_plus_")
    fig.suptitle(
        f"{model_name}  |  seq_len={seq_len:,}  |  {N_REPS} reps  |  {elapsed:.1f}s  |  {DEVICE}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, f"verify_{safe_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}\n")
    return out_path


def main():
    print(f"Device: {DEVICE}")
    print(f"Cache : {CACHE_DIR}")
    print(f"Models: {sorted(MODEL_CONFIGS.keys())}")
    print(f"Reps  : {N_REPS}")
    print("=" * 60)

    results = {}
    for model_name in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_name]
        print(f"\n[{model_name}]")
        t0 = time.time()

        try:
            if cfg["adapter"]:
                tmrca, true_tmrcas, pairs, seq_len = run_adapter_model(model_name, cfg)
            else:
                tmrca, true_tmrcas, pairs, seq_len = run_base_model(model_name, cfg)

            elapsed = time.time() - t0

            for i, (a, b) in enumerate(pairs):
                pred_mean = tmrca[:, i, :].mean(axis=0)
                ytrue = true_tmrcas[i]
                mse = float(np.mean((pred_mean - ytrue) ** 2))
                corr = float(np.corrcoef(pred_mean, ytrue)[0, 1])
                results[(model_name, a, b)] = {"mse": mse, "r": corr}

            plot_model(model_name, tmrca, true_tmrcas, pairs, seq_len, elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            results[(model_name, -1, -1)] = {"error": str(e)}

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, val in sorted(results.items()):
        model_name, a, b = key
        if "error" in val:
            print(f"  {model_name:30s}  FAILED: {val['error'][:60]}")
        else:
            print(f"  {model_name:30s}  pair=({a},{b})  MSE={val['mse']:.3f}  r={val['r']:.3f}")
    print()


if __name__ == "__main__":
    main()
