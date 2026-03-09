"""Verify that cxt.translate produces consistent results across all three
input types: tree sequence, genotype matrix, and VCF file.

Downloads the broad model (if not cached), simulates a tree sequence,
exports it as genotype matrix and VCF, runs inference through all three
paths, and compares the outputs + plots them."""

import os
import sys
import io
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
from cxt.utils import simulate_parameterized_tree_sequence
from cxt.preprocess import interpolate_tmrcas

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 77
SEQ_LEN = 1_000_000
N_REPS = 5
MODEL_TYPE = "broad"

PIVOT_PAIRS = [(0, 1), (4, 5)]
BLOCKS = [(0, SEQ_LEN)]

COMMON_KW = dict(
    blocks=BLOCKS,
    pivot_pairs=PIVOT_PAIRS,
    devices=[DEVICE],
    B=len(PIVOT_PAIRS),
    B_per_device=len(PIVOT_PAIRS),
    n_reps=N_REPS,
    base_seed=SEED,
    top_k=50,
    cache_matching=True,
    progress=False,
    decode_bar=False,
    build_workers=1,
)


def write_vcf(ts, vcf_path):
    """Write a tree sequence to a minimal VCF that cxt.vcf_parser can read."""
    with open(vcf_path, "w") as f:
        ts.write_vcf(f)


def main():
    print(f"Device: {DEVICE}")
    print(f"Model : {MODEL_TYPE}")
    print(f"Seed  : {SEED}")
    print("=" * 60)

    # --- Simulate ---
    print("\nSimulating tree sequence...")
    ts = simulate_parameterized_tree_sequence(seed=SEED, samples=25, sequence_length=SEQ_LEN)
    print(f"  {ts.num_trees} trees, {ts.num_sites} sites, {ts.num_samples} haplotypes")

    # --- Prepare inputs ---
    gm = ts.genotype_matrix().T
    positions = ts.tables.sites.position
    print(f"  Genotype matrix: {gm.shape}")

    vcf_path = os.path.join(SCRIPT_DIR, "test_input.vcf")
    print(f"  Writing VCF to {vcf_path}...")
    write_vcf(ts, vcf_path)
    vcf_size = os.path.getsize(vcf_path)
    print(f"  VCF size: {vcf_size / 1024:.0f} KB")

    # --- Load model ---
    print(f"\nLoading {MODEL_TYPE} model...")
    model = cxt.load_model(MODEL_TYPE, device=DEVICE)

    # --- True TMRCA ---
    print("Computing true TMRCAs...")
    window_size = SEQ_LEN // 500
    true_tmrcas = []
    for a, b in PIVOT_PAIRS:
        t = interpolate_tmrcas(ts, window_size, SEQ_LEN, a, b)
        true_tmrcas.append(np.log(t))
    true_tmrcas = np.array(true_tmrcas)

    # ===== 1. Tree sequence input =====
    print("\n[1/3] translate(ts, ...) — tree sequence input")
    t0 = time.time()
    tmrca_ts, idx_ts = cxt.translate(ts, model, **COMMON_KW)
    t_ts = time.time() - t0
    print(f"  shape: {tmrca_ts.shape}  ({t_ts:.1f}s)")

    # ===== 2. Genotype matrix input =====
    print("\n[2/3] translate((gm, positions), ...) — genotype matrix input")
    t0 = time.time()
    tmrca_gm, idx_gm = cxt.translate((gm, positions), model, **COMMON_KW)
    t_gm = time.time() - t0
    print(f"  shape: {tmrca_gm.shape}  ({t_gm:.1f}s)")

    # ===== 3. VCF input =====
    print("\n[3/3] translate(vcf_path, ...) — VCF file input")
    t0 = time.time()
    tmrca_vcf, idx_vcf = cxt.translate(vcf_path, model, **COMMON_KW)
    t_vcf = time.time() - t0
    print(f"  shape: {tmrca_vcf.shape}  ({t_vcf:.1f}s)")

    # --- Consistency checks ---
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECKS")
    print("=" * 60)

    ts_mean = tmrca_ts.mean(axis=0)
    gm_mean = tmrca_gm.mean(axis=0)
    vcf_mean = tmrca_vcf.mean(axis=0)

    ts_vs_gm = float(np.max(np.abs(ts_mean - gm_mean)))
    ts_vs_vcf = float(np.max(np.abs(ts_mean - vcf_mean)))
    gm_vs_vcf = float(np.max(np.abs(gm_mean - vcf_mean)))

    ts_gm_corr = float(np.corrcoef(ts_mean.flatten(), gm_mean.flatten())[0, 1])
    ts_vcf_corr = float(np.corrcoef(ts_mean.flatten(), vcf_mean.flatten())[0, 1])
    gm_vcf_corr = float(np.corrcoef(gm_mean.flatten(), vcf_mean.flatten())[0, 1])

    print(f"  ts  vs gm : max|diff|={ts_vs_gm:.4f}  r={ts_gm_corr:.6f}")
    print(f"  ts  vs vcf: max|diff|={ts_vs_vcf:.4f}  r={ts_vcf_corr:.6f}")
    print(f"  gm  vs vcf: max|diff|={gm_vs_vcf:.4f}  r={gm_vcf_corr:.6f}")

    # The ts and gm paths go through identical code (ts just extracts gm/pos first),
    # so with same seed they must match exactly.
    if np.allclose(tmrca_ts, tmrca_gm, atol=1e-6):
        print("  [PASS] ts == gm  (exact match)")
    else:
        print("  [WARN] ts != gm  (should be identical)")

    # VCF may differ slightly due to float32 position casting and site filtering
    if gm_vcf_corr > 0.99:
        print("  [PASS] gm ~ vcf  (correlation > 0.99)")
    else:
        print(f"  [WARN] gm vs vcf correlation only {gm_vcf_corr:.4f}")

    # --- Accuracy vs truth ---
    print("\nACCURACY vs TRUE TMRCA")
    print("-" * 60)
    for label, tmrca in [("ts", tmrca_ts), ("gm", tmrca_gm), ("vcf", tmrca_vcf)]:
        for i, (a, b) in enumerate(PIVOT_PAIRS):
            pred_mean = tmrca[:, i, :].mean(axis=0)
            ytrue = true_tmrcas[i]
            mse = float(np.mean((pred_mean - ytrue) ** 2))
            corr = float(np.corrcoef(pred_mean, ytrue)[0, 1])
            print(f"  {label:4s}  pair=({a},{b})  MSE={mse:.3f}  r={corr:.3f}")

    # --- Plot ---
    print("\nGenerating figure...")
    x = np.arange(0, SEQ_LEN, window_size) / 1e6
    input_labels = ["Tree sequence", "Genotype matrix", "VCF file"]
    input_data = [tmrca_ts, tmrca_gm, tmrca_vcf]
    colors = ["C0", "C1", "C4"]

    n_pairs = len(PIVOT_PAIRS)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 4.0 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for i, (a, b) in enumerate(PIVOT_PAIRS):
        ax = axes[i]
        ytrue = true_tmrcas[i]
        ax.plot(x, ytrue, color="black", linewidth=1.4, alpha=0.7, label="True TMRCA", zorder=10)

        for j, (label, tmrca, col) in enumerate(zip(input_labels, input_data, colors)):
            pred = tmrca[:, i, :]
            pred_mean = pred.mean(axis=0)
            pred_std = pred.std(axis=0)
            mse = float(np.mean((pred_mean - ytrue) ** 2))
            corr = float(np.corrcoef(pred_mean, ytrue)[0, 1])

            offset = (j - 1) * 0.0005
            ax.plot(x + offset, pred_mean, color=col, linewidth=0.9, alpha=0.85,
                    label=f"{label}  (MSE={mse:.3f}, r={corr:.3f})")
            ax.fill_between(x + offset,
                            pred_mean - 1.5 * pred_std,
                            pred_mean + 1.5 * pred_std,
                            alpha=0.12, color=col)

        ax.set_title(f"Pair ({a}, {b})", fontsize=12)
        ax.set_ylabel("log TMRCA", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Genomic position (Mb)", fontsize=11)
    fig.suptitle(
        f"Input-type comparison  |  model={MODEL_TYPE}  |  {N_REPS} reps  |  seed={SEED}  |  {DEVICE}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "verify_input_types.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")

    # --- Pairwise diff plot ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
    diff_pairs = [
        ("ts vs gm", ts_mean, gm_mean, "C0"),
        ("ts vs vcf", ts_mean, vcf_mean, "C1"),
        ("gm vs vcf", gm_mean, vcf_mean, "C4"),
    ]
    for ax, (label, a_data, b_data, col) in zip(axes2, diff_pairs):
        for i, (pa, pb) in enumerate(PIVOT_PAIRS):
            diff = a_data[i] - b_data[i]
            ax.plot(x, diff, linewidth=0.7, alpha=0.8, label=f"pair ({pa},{pb})")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"Δ ({label})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        max_diff = float(np.max(np.abs(a_data - b_data)))
        ax.set_title(f"{label}  |  max|Δ|={max_diff:.5f}", fontsize=10)

    axes2[-1].set_xlabel("Genomic position (Mb)", fontsize=11)
    fig2.suptitle("Pairwise differences between input types (mean predictions)",
                  fontsize=12, fontweight="bold", y=1.01)
    fig2.tight_layout()
    out2 = os.path.join(SCRIPT_DIR, "verify_input_diffs.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  -> {out2}")

    print("\nDone.")


if __name__ == "__main__":
    main()
