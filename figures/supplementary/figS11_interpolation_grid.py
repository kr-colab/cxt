"""
Figure S11: Interpolation and extrapolation benchmark across mutation/recombination rates.

Evaluates cxt-broad on a 10x10 grid of mutation x recombination rates derived
from stdpopsim species, computing MSE and KL divergence at each grid point,
both with and without stochastic diversity bias correction.

Layout: 2-wide x 3-tall (6 panels).
  Row 1: (A) rho/theta ratio heatmap, (B) 2D Gaussian fit scatter
  Row 2: (C) MSE heatmap (uncorrected), (D) log10(KLD) heatmap (uncorrected)
  Row 3: (E) Corrected MSE heatmap, (F) Corrected log10(KLD) heatmap

Faithful reproduction of revision/figureS9_S10/experiment.ipynb.
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import stdpopsim
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from scipy.stats import chi2, gaussian_kde

import cxt
from cxt.correction import stochastic_diversity_bias_correction
from cxt.preprocess import interpolate_tmrcas
from cxt.utils import simulate_parameterized_tree_sequence
from cxt.utils import TIMES, discretize


WINDOW_BP = 2000
SEQ_LEN = 1_000_000
SEED = 103370001


def _is_very_close(rate, target, tol=1e-12):
    return abs(rate - target) < tol


def get_species_grid():
    """Build 10x10 linspace grid from stdpopsim species rates."""
    rows = []
    for sp in stdpopsim.all_species():
        if sp.id in ('StrAga', 'EscCol', 'ChlRei'):
            continue
        rows.append({
            'species': sp.name,
            'mutation_rate': sp.genome.mean_mutation_rate,
            'recombination_rate': sp.genome.mean_recombination_rate,
        })

    recs = sorted(r['recombination_rate'] for r in rows
                  if not (_is_very_close(r['recombination_rate'], 2.318298e-07)
                          or _is_very_close(r['recombination_rate'], 3.05311036e-11)))
    muts = sorted(r['mutation_rate'] for r in rows
                  if not _is_very_close(r['mutation_rate'], 2.10000000e-10))

    grid_rec = np.linspace(recs[0], recs[-1], 10)
    grid_mut = np.linspace(muts[0], muts[-1], 10)
    return grid_mut, grid_rec, rows


def discrete_ground_truth(ts, a, b, sequence_length=SEQ_LEN, window_size=WINDOW_BP):
    ytrue = np.log(interpolate_tmrcas(ts, window_size, sequence_length, a, b))
    ytrue = discretize(ytrue, TIMES)
    return TIMES[ytrue]


def discrete_ground_truth_pairs(ts, pairs):
    return np.array([discrete_ground_truth(ts, a, b) for a, b in pairs])


def calculate_kld(ytrues, yhats):
    p_raw = yhats.flatten()
    q_raw = ytrues.flatten()
    p_raw = p_raw[~np.isnan(p_raw)]
    q_raw = q_raw[~np.isnan(q_raw)]
    x = np.linspace(min(p_raw.min(), q_raw.min()),
                    max(p_raw.max(), q_raw.max()), 512)
    p = gaussian_kde(p_raw)(x)
    q = gaussian_kde(q_raw)(x)
    p = np.clip(p / p.sum(), 1e-12, None)
    q = np.clip(q / q.sum(), 1e-12, None)
    from scipy.stats import entropy
    return float(entropy(p, q))


def sci_notation(val, sig_fig=1):
    if val == 0:
        return "0"
    s = f"{val:.{sig_fig}e}"
    m, e = s.split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def plot_heatmap_examples(yhats_all, ytrues_all, mses_raw, klds_raw,
                          output_path, tag=""):
    """Generate the 4x2 line+KDE figure for selected grid points."""
    indices = [10, 19, 80, 89]
    indices_labels = [80, 89, 10, 19]
    n_rows = len(indices)

    plt.rcParams.update({'font.size': 12})
    fontsize = 12

    def millions(x, pos):
        if x == 0:
            return "0"
        return f"{x/1e6:.1f}x10\u2076"

    formatter = FuncFormatter(millions)

    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 8),
                             gridspec_kw={'width_ratios': [2.5, 1]})
    k = 0
    for i, idx in zip([2, 3, 0, 1], indices):
        pred_replicates = yhats_all[idx]
        true_replicates = ytrues_all[idx]

        predicted_mean = pred_replicates.mean(0).flatten()
        predicted_std = pred_replicates.std(0).flatten()
        true_mean = true_replicates.flatten()
        x_values = np.arange(len(predicted_mean)) * WINDOW_BP

        ax_line = axes[i, 0]
        ax_line.plot(x_values, predicted_mean, color="#4682B4")
        ax_line.fill_between(x_values,
                             predicted_mean - predicted_std,
                             predicted_mean + predicted_std,
                             color="#ADDFFF", alpha=1)
        ax_line.plot(x_values, predicted_mean - predicted_std,
                     linestyle="-", linewidth=0.2, color="black")
        ax_line.plot(x_values, predicted_mean + predicted_std,
                     linestyle="-", linewidth=0.2, color="black")
        ax_line.plot(x_values, true_mean, color="black", linewidth=0.5,
                     drawstyle="steps-mid")

        title = (f"[{indices_labels[k]+1}/100]  MSE: {mses_raw[idx]:.3f}, "
                 f"log\u2081\u2080(KLD): {np.log10(klds_raw[idx]):.2f}")
        ax_line.set_title(title, fontsize=fontsize, loc="left")
        k += 1
        ax_line.tick_params(labelbottom=False)
        ax_line.set_xlim(0, 500 * WINDOW_BP)
        ax_line.set_xlabel('')
        ax_line.grid(True)
        ax_line.set_ylabel('')

        ax_kde = axes[i, 1]
        sns.kdeplot(true_mean, fill=True, alpha=0.25, linewidth=1.4,
                    bw_adjust=1.5, ax=ax_kde, label='True')
        sns.kdeplot(predicted_mean, linestyle="dashed", fill=False, linewidth=2,
                    bw_adjust=1.5, ax=ax_kde, label='Predicted')
        ax_kde.set_xlabel("")
        ax_kde.set_ylabel("")
        ax_kde.spines["top"].set_visible(False)
        ax_kde.spines["right"].set_visible(False)
        ax_kde.grid(True)
        if i == 3:
            ax_kde.legend(loc='upper left', fontsize=fontsize)

    axes[3][0].xaxis.set_major_formatter(formatter)
    axes[3][0].tick_params(labelbottom=True)
    axes[3][0].set_xlabel('Sequence [bp]')
    axes[3][1].tick_params(labelbottom=True)
    axes[3][1].set_xlabel('log(Time) [generations]')

    fig.text(0.01, 0.5, "log(Time) [generations]", va="center",
             rotation="vertical", fontsize=12)
    fig.text(0.675, 0.5, "Density", va="center",
             rotation="vertical", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/supplementary")
    parser.add_argument("--cache-dir",
                        default="figures/output/supplementary/cache/figS11")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    grid_mut, grid_rec, species_rows = get_species_grid()
    n = len(grid_mut)

    model = cxt.load_model("broad", device="cpu")

    num_haploid = 20
    pivot_pairs = [(i, j) for i in range(num_haploid)
                   for j in range(i + 1, num_haploid)]
    blocks = [(0, SEQ_LEN)]

    # --- Simulate & infer across grid ---
    mse_cache = os.path.join(args.cache_dir, "mses.npz")
    need_recompute = True
    if os.path.exists(mse_cache):
        d = np.load(mse_cache)
        if "corrected_mses" in d and "corrected_klds" in d:
            mses_raw = d["mses"]
            klds_raw = d["klds"]
            corrected_mses_raw = d["corrected_mses"]
            corrected_klds_raw = d["corrected_klds"]
            need_recompute = False
        else:
            os.rename(mse_cache, mse_cache + ".bak")
            print("Old mses.npz lacks corrected fields; backed up and will recompute metrics.")

    if need_recompute:
        ts_list = []
        for mut in grid_mut:
            for rec in grid_rec:
                ts = simulate_parameterized_tree_sequence(
                    seed=SEED, recombination_rate=rec, mutation_rate=mut)
                ts_list.append(ts)

        mses_list = []
        klds_list = []
        corrected_mses_list = []
        corrected_klds_list = []

        for i, ts in enumerate(ts_list):
            print(f"Grid point {i+1}/{len(ts_list)}")
            tmrca_path = os.path.join(args.cache_dir, f"tmrca_{i}.npy")
            corrected_path = os.path.join(args.cache_dir, f"corrected_tmrca_{i}.npy")

            if os.path.exists(tmrca_path):
                tmrca = np.load(tmrca_path)
            else:
                tmrca, _ = cxt.translate(
                    ts, model, pivot_pairs=pivot_pairs,
                    blocks=blocks, devices=args.devices,
                    B_per_device=args.batch_size, B=args.batch_size,
                    build_workers=8, mutation_rate=None,
                )
                np.save(tmrca_path, tmrca)

            if os.path.exists(corrected_path):
                corrected_tmrca = np.load(corrected_path)
            else:
                current_mutation_rate = json.loads(
                    ts.provenance(-1).record
                )['parameters']['rate']
                corrected_tmrca = stochastic_diversity_bias_correction(
                    tree_sequence=ts,
                    mutation_rate=current_mutation_rate,
                    predictions=tmrca,
                    pivot_pairs=np.array(pivot_pairs),
                    rng=np.random.default_rng(20_000_001),
                )
                np.save(corrected_path, corrected_tmrca)

            true_tmrcas = discrete_ground_truth_pairs(ts, pivot_pairs)

            mse = float(np.mean((true_tmrcas - tmrca.mean(0)) ** 2))
            kld = calculate_kld(true_tmrcas, tmrca.mean(0))
            mses_list.append(mse)
            klds_list.append(kld)

            corrected_mse = float(np.mean((true_tmrcas - corrected_tmrca.mean(0)) ** 2))
            corrected_kld = calculate_kld(true_tmrcas, corrected_tmrca.mean(0))
            corrected_mses_list.append(corrected_mse)
            corrected_klds_list.append(corrected_kld)

            print(f"  MSE={mse:.4f}, Corrected MSE={corrected_mse:.4f}, "
                  f"KLD={kld:.4f}, Corrected KLD={corrected_kld:.4f}")

        mses_raw = np.array(mses_list)
        klds_raw = np.array(klds_list)
        corrected_mses_raw = np.array(corrected_mses_list)
        corrected_klds_raw = np.array(corrected_klds_list)
        np.savez_compressed(mse_cache,
                            mses=mses_raw, klds=klds_raw,
                            corrected_mses=corrected_mses_raw,
                            corrected_klds=corrected_klds_raw)

    # --- Build matrices ---
    mse_matrix = mses_raw.reshape(n, n)
    kld_matrix = klds_raw.reshape(n, n)
    corrected_mse_matrix = corrected_mses_raw.reshape(n, n)
    corrected_kld_matrix = corrected_klds_raw.reshape(n, n)

    rhotheta_matrix = np.zeros((n, n))
    for i, rec in enumerate(grid_rec):
        for j, mut in enumerate(grid_mut):
            rhotheta_matrix[i, j] = rec / mut

    # --- Species scatter data ---
    added_m = [1.29e-8] * 2400000 + [1e-8] * 655000 + [1e-8] * 645000 + [5e-8] * 655000
    added_r = [1.28e-8] * 2400000 + [1e-8] * 655000 + [5e-8] * 645000 + [1e-8] * 655000

    rates_npz = os.path.join(args.cache_dir, "rates.npz")
    revision_rates = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "revision", "figureS9_S10", "cache", "rates.npz")
    if os.path.exists(rates_npz):
        rates = np.load(rates_npz)["rates"]
    elif os.path.exists(revision_rates):
        import shutil
        shutil.copy2(revision_rates, rates_npz)
        rates = np.load(rates_npz)["rates"]
        print(f"Copied rates.npz from revision cache ({rates.shape[0]} rate pairs)")
    else:
        filt_rows = [r for r in species_rows
                     if r['species'] not in ('Anolis carolinensis',
                                             'Apis mellifera',
                                             'Gasterosteus aculeatus')]
        rates = np.array([[r['mutation_rate'], r['recombination_rate']]
                          for r in filt_rows])
        print("Warning: rates.npz not found, using stdpopsim species only")

    rs_all = np.concatenate([rates[:, 1], np.array(added_r)])
    ms_all = np.concatenate([rates[:, 0], np.array(added_m)])

    # --- Tick labels ---
    xtl = [sci_notation(v, 1) for v in grid_rec]
    ytl = [sci_notation(v, 1) for v in grid_mut]

    plt.rcParams.update({'font.size': 11})

    # --- 3x2 heatmaps figure ---
    fig, axes = plt.subplots(3, 2, figsize=(15, 22), constrained_layout=True)

    # (A) Top-left: rho/theta ratio
    h1 = sns.heatmap(rhotheta_matrix,
                     xticklabels=xtl, yticklabels=ytl,
                     cmap='magma', annot=True, fmt=".2f",
                     ax=axes[0][0], cbar_kws={'shrink': 1.0})
    axes[0][0].set_ylabel('Mutation rate')
    axes[0][0].invert_yaxis()
    axes[0][0].set_title("Recombination/Mutation Ratio")

    # (B) Top-right: species scatter with sigma ellipses
    ax = axes[0][1]
    ax.set_xticks(grid_rec, minor=True)
    ax.set_yticks(grid_mut, minor=True)
    ax.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
    ax.grid(which='major', visible=False)
    ax.scatter(rs_all, ms_all, s=10, color="gray", alpha=0.2, edgecolor="none")

    data2 = np.vstack([rs_all, ms_all]).T
    mu = data2.mean(axis=0)
    cov = np.cov(data2, rowvar=False)
    q = chi2.ppf([0.6827, 0.9545, 0.9973], df=2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    angle_rad = np.deg2rad(angle)
    ellipse_colors = ["#08306b", "#08519c", "#1f77b4"]
    for r_val, color, label in zip(np.sqrt(q), ellipse_colors,
                                   ["1\u03c3", "2\u03c3", "3\u03c3"]):
        w, h = 2 * np.sqrt(eigvals) * r_val
        ell = Ellipse(mu, w, h, angle=angle,
                      edgecolor=color, facecolor="none",
                      linestyle="--", linewidth=1.2)
        ax.add_patch(ell)
        dx = (w / 2) * np.cos(angle_rad) * 0.7
        dy = (h / 2) * np.sin(angle_rad) * 0.7
        ax.text(mu[0] + abs(dx), mu[1] + abs(dy),
                label, color=color, fontsize=9, ha="left", va="bottom",
                clip_on=True)

    xmin, xmax = 8e-10, 4.4e-08
    ymin, ymax = 1.5e-09, 3.8e-08
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                      linewidth=1.5, edgecolor="black", facecolor="none",
                      zorder=5)
    ax.add_patch(rect)
    pad_frac = 0.4
    ax.set_xlim(xmin - (xmax - xmin) * 0.25 * pad_frac,
                xmax + (xmax - xmin) * 1.75 * pad_frac)
    ax.set_ylim(ymin - (ymax - ymin) * 0.25 * pad_frac,
                ymax + (ymax - ymin) * pad_frac)
    ax.autoscale(enable=False)
    ax.set_title("2D Gaussian Fit of Dataset", pad=10)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    # (C) Middle-left: MSE heatmap (uncorrected)
    sns.heatmap(mse_matrix,
                cmap='magma', annot=True, fmt=".2f",
                xticklabels=xtl, yticklabels=ytl,
                ax=axes[1][0], cbar_kws={'shrink': 1.0})
    axes[1][0].set_title("MSE Heatmap")
    axes[1][0].set_xlabel("Recombination Rate")
    axes[1][0].set_ylabel('Mutation rate')
    axes[1][0].invert_yaxis()

    # (D) Middle-right: log10(KLD) heatmap (uncorrected)
    sns.heatmap(np.log10(kld_matrix),
                cmap='magma', annot=True, fmt=".2f",
                xticklabels=xtl, yticklabels=ytl,
                ax=axes[1][1], cbar_kws={'shrink': 1.0})
    axes[1][1].set_title("log\u2081\u2080(KLD) Heatmap")
    axes[1][1].set_xlabel("Recombination Rate")
    axes[1][1].set_ylabel("")
    axes[1][1].set_yticks([])

    # (E) Bottom-left: Corrected MSE heatmap
    sns.heatmap(corrected_mse_matrix,
                cmap='magma', annot=True, fmt=".2f",
                xticklabels=xtl, yticklabels=ytl,
                ax=axes[2][0], cbar_kws={'shrink': 1.0})
    axes[2][0].set_title("Corrected MSE Heatmap")
    axes[2][0].set_xlabel("Recombination Rate")
    axes[2][0].set_ylabel('Mutation rate')
    axes[2][0].invert_yaxis()

    # (F) Bottom-right: Corrected log10(KLD) heatmap
    sns.heatmap(np.log10(corrected_kld_matrix),
                cmap='magma', annot=True, fmt=".2f",
                xticklabels=xtl, yticklabels=ytl,
                ax=axes[2][1], cbar_kws={'shrink': 1.0})
    axes[2][1].set_title("Corrected log\u2081\u2080(KLD) Heatmap")
    axes[2][1].set_xlabel("Recombination Rate")
    axes[2][1].set_ylabel("")
    axes[2][1].set_yticks([])

    first_cb = h1.collections[0].colorbar
    cb_pos = first_cb.ax.get_position()
    phantom = fig.add_axes(cb_pos)
    phantom.axis('off')

    out = os.path.join(args.output_dir, "figS11_interpolation_grid.png")
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")
    plt.close(fig)

    # --- Heatmap examples (4x2 line+KDE figures) ---
    example_indices = [10, 19, 80, 89]

    for variant, tmrca_prefix, mses_arr, klds_arr in [
        ("uncorrected", "tmrca", mses_raw, klds_raw),
        ("corrected", "corrected_tmrca", corrected_mses_raw, corrected_klds_raw),
    ]:
        yhats_all = {}
        ytrues_all = {}

        for idx in example_indices:
            tmrca_path = os.path.join(args.cache_dir, f"{tmrca_prefix}_{idx}.npy")
            if not os.path.exists(tmrca_path):
                print(f"Skipping {variant} heatmap_examples: {tmrca_prefix}_{idx}.npy not found")
                break
            tmrca = np.load(tmrca_path)

            mut_i = idx // n
            rec_i = idx % n
            mut = grid_mut[mut_i]
            rec = grid_rec[rec_i]

            ts = simulate_parameterized_tree_sequence(
                seed=SEED, recombination_rate=rec, mutation_rate=mut)

            true_tmrcas = discrete_ground_truth(ts, 0, 1)
            yhats_all[idx] = tmrca[:, 0, :]
            ytrues_all[idx] = true_tmrcas.reshape(1, -1)
        else:
            suffix = "" if variant == "uncorrected" else "_corrected"
            out_examples = os.path.join(args.output_dir,
                                        f"heatmap_examples{suffix}.png")
            plot_heatmap_examples(yhats_all, ytrues_all, mses_arr, klds_arr,
                                  out_examples, tag=variant)


if __name__ == "__main__":
    main()

