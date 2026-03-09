"""Download the broad model to .verification/checkpoints, simulate a tree
sequence, run inference, and plot predicted vs true TMRCA."""

import os
import sys

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
from cxt.utils import simulate_parameterized_tree_sequence
from cxt.preprocess import interpolate_tmrcas

SEED = 42
SEQ_LEN = 1_000_000
WINDOW_SIZE = 2000
N_WINDOWS = SEQ_LEN // WINDOW_SIZE
N_REPS = 15
MODEL_TYPE = "broad"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Cache dir: {CACHE_DIR}")
print()

print("Loading model...")
model = cxt.load_model(MODEL_TYPE, device=DEVICE)
print("  Done.\n")

print("Simulating tree sequence (seed=42, 25 samples, 1 Mb)...")
ts = simulate_parameterized_tree_sequence(seed=SEED, samples=25, sequence_length=SEQ_LEN)
n_trees = ts.num_trees
n_sites = ts.num_sites
print(f"  {n_trees} trees, {n_sites} sites\n")

pivot_pairs = [(0, 1), (2, 3), (4, 5)]

print(f"Running inference ({N_REPS} reps, {len(pivot_pairs)} pairs)...")
tmrca, index_map = cxt.translate(
    ts, model,
    blocks=[(0, SEQ_LEN)],
    pivot_pairs=pivot_pairs,
    devices=[DEVICE],
    B=len(pivot_pairs), B_per_device=len(pivot_pairs),
    n_reps=N_REPS,
    base_seed=SEED,
    top_k=50,
    cache_matching=True,
    progress=True,
    decode_bar=False,
    build_workers=1,
)
print(f"  tmrca shape: {tmrca.shape}\n")

print("Computing true TMRCAs from tree sequence...")
true_tmrcas = []
for a, b in pivot_pairs:
    t = interpolate_tmrcas(ts, WINDOW_SIZE, SEQ_LEN, a, b)
    true_tmrcas.append(np.log(t))
true_tmrcas = np.array(true_tmrcas)
print(f"  true_tmrcas shape: {true_tmrcas.shape}\n")

# --- Plotting ---
x = np.arange(0, SEQ_LEN, WINDOW_SIZE) / 1e6

fig, axes = plt.subplots(len(pivot_pairs), 1, figsize=(12, 3.5 * len(pivot_pairs)),
                         sharex=True)
if len(pivot_pairs) == 1:
    axes = [axes]

for i, (a, b) in enumerate(pivot_pairs):
    ax = axes[i]

    pred = tmrca[:, i, :]
    pred_mean = pred.mean(axis=0)
    pred_std = pred.std(axis=0)
    ytrue = true_tmrcas[i]

    ax.fill_between(x, pred_mean - 2 * pred_std, pred_mean + 2 * pred_std,
                    alpha=0.25, color="C0", label="Predicted ± 2σ")
    ax.plot(x, pred_mean, color="C0", linewidth=1.2, label="Predicted (mean)")
    ax.plot(x, ytrue, color="C3", linewidth=1.0, alpha=0.8, label="True TMRCA")

    mse = float(np.mean((pred_mean - ytrue) ** 2))
    corr = float(np.corrcoef(pred_mean, ytrue)[0, 1])
    ax.set_title(f"Pair ({a}, {b})  —  MSE={mse:.3f}, r={corr:.3f}", fontsize=12)
    ax.set_ylabel("log TMRCA", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

axes[-1].set_xlabel("Genomic position (Mb)", fontsize=11)
fig.suptitle(f"cxt verification  |  model={MODEL_TYPE}  |  device={DEVICE}  |  seed={SEED}",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()

out_path = os.path.join(SCRIPT_DIR, "verification_figure.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to {out_path}")
