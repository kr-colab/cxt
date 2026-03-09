"""
Plot Singer+Polegon figures from pre-computed Polegon MCMC output.

Produces panels that mirror the cxt figures for direct comparison:
  - Fig 2:  singer_constant.png, singer_sawtooth.png  (TMRCA scatter)
  - Fig 4:  singer_stdpopsim_v3.png                   (KDE panels)
  - Fig 5:  singer_demography.png                     (coalescence-rate step)

Data paths point to the shared Singer benchmark directory.
"""

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde
import tskit

from cxt.preprocess import interpolate_tmrcas
from cxt.utils import TIMES, coalescence_rates
from figures.utils import plot_tmrca_scatter

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

from figures.paths import SINGER_BASE
MCMC_REPS = np.arange(50, 100, 5)

BIN_BP = 2000
SEQ_LEN = 1_000_000

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_polegon_trees(path):
    """Load Polegon .trees files, clearing mutations to avoid validation errors."""
    tc = tskit.TableCollection.load(path)
    tc.mutations.clear()
    return tc.tree_sequence()


def _interp_worker(args):
    ts, a, b = args
    return interpolate_tmrcas(ts, BIN_BP, SEQ_LEN, a, b)


def interpolate_tmrcas_robust(ts, window_size, sequence_length, a, b):
    windows = np.linspace(0, sequence_length, int(sequence_length // window_size) + 1)
    windows[-1] = ts.sequence_length
    tmrca = ts.diversity(sample_sets=[(a, b)], windows=windows, mode="branch") / 2
    return tmrca.T[0]


def discretize(sequence):
    indices = np.searchsorted(TIMES, np.log(sequence), side="right") - 1
    return np.exp(TIMES[np.clip(indices, 0, len(TIMES) - 1)])


def kde_pdf(samples, grid, bw_method=None):
    if np.all(samples == samples[0]):
        loc = float(samples[0])
        pdf = np.exp(-0.5 * ((grid - loc) / (1e-6 + 0.01 * (np.ptp(grid) or 1.0))) ** 2)
        pdf /= np.trapezoid(pdf, grid)
        return pdf
    kde = gaussian_kde(samples, bw_method=bw_method)
    pdf = kde(grid)
    area = np.trapezoid(pdf, grid)
    return pdf / area if area > 0 else pdf


def kl_divergence(p_grid, q_grid, x_grid, eps=1e-12):
    p, q = p_grid + eps, q_grid + eps
    p /= np.trapezoid(p, x_grid)
    q /= np.trapezoid(q, x_grid)
    return float(np.trapezoid(p * (np.log(p) - np.log(q)), x_grid))


def robust_grid(a, b, n=512, q_lo=0.005, q_hi=0.995):
    lo = np.nanmin([np.quantile(a, q_lo), np.quantile(b, q_lo)])
    hi = np.nanmax([np.quantile(a, q_hi), np.quantile(b, q_hi)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = min(np.nanmin(a), np.nanmin(b)), max(np.nanmax(a), np.nanmax(b))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, n)


_ln10 = np.log(10.0)


def set_loge_power10_ticks(ax, xmin, xmax):
    kmin = int(np.ceil(xmin / _ln10))
    kmax = int(np.floor(xmax / _ln10))
    ticks = [k * _ln10 for k in range(kmin, kmax + 1)]
    if ticks:
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: r"$10^{%d}$" % round(np.log10(np.e) * x)
        ))


# -----------------------------------------------------------------------
# Fig 2: Singer TMRCA scatter (constant & sawtooth)
# -----------------------------------------------------------------------

def plot_fig2_singer(output_dir, cache_dir):
    """Singer scatter plots for constant and sawtooth demographies."""
    polegon_dir = os.path.join(SINGER_BASE, "generic-models/polegon-output")
    sim_dir = os.path.join(SINGER_BASE, "generic-models/simulations")
    pairs = [(i, j) for i in range(50) for j in range(i + 1, 50)]

    for scenario, file_stem in [("constant", "constant_2824078380"),
                                ("sawtooth", "sawtooth_2824078380")]:
        cache_path = os.path.join(cache_dir, f"singer_{scenario}.npz")
        ts_path = os.path.join(sim_dir, f"tss_{scenario}.pkl")

        with open(ts_path, "rb") as f:
            tss = pickle.load(f)
        ts = tss[0] if isinstance(tss, list) else tss

        if os.path.exists(cache_path):
            d = np.load(cache_path)
            yhats, ytrues = d["yhats"], d["ytrues"]
        else:
            yhats_mcmc = []
            for rep in MCMC_REPS:
                fp = os.path.join(polegon_dir, f"{file_stem}.{rep}.polegon.trees")
                ts_singer = load_polegon_trees(fp)
                rep_tmrcas = []
                for a, b in pairs:
                    rep_tmrcas.append(interpolate_tmrcas(ts_singer, BIN_BP, SEQ_LEN, a, b))
                yhats_mcmc.append(rep_tmrcas)
                print(f"  Loaded {os.path.basename(fp)}")
            yhats = np.array(yhats_mcmc)

            with ProcessPoolExecutor(max_workers=24) as ex:
                ytrues = np.array(list(ex.map(
                    _interp_worker, [(ts, a, b) for a, b in pairs]
                )))

            np.savez_compressed(cache_path, yhats=yhats, ytrues=ytrues)

        yhat_mean = np.array(yhats).mean(0) if yhats.ndim == 3 else yhats
        ytrues_d = discretize(ytrues)
        plot_tmrca_scatter(
            yhat_mean, ytrues_d,
            os.path.join(output_dir, f"singer_{scenario}.png"),
            tool=rf"$\mathbf{{Singer+Polegon}}$: {scenario.capitalize()} Ne",
        )
        print(f"  Saved singer_{scenario}.png")


# -----------------------------------------------------------------------
# Fig 4: Singer KDE panels (stdpopsim v3 species)
# -----------------------------------------------------------------------

V3_SPECIES = [
    "MusMus_DomesticusEurope_1F22",
    "MusMus_MusculusKorea_1F22",
    "MusMus_CastaneusIndia_1F22",
    "RatNor_PiecewiseConstant",
    "GorGor_GorillaGhost_5P23",
    "OrySat_BottleneckMigration_3C07",
    "SusScr_PiecewiseConstant",
    "PhoSin_Vaquita2Epoch_1R22",
]


def _simulate_v3_species(spec, cache_dir):
    """Simulate a stdpopsim v3 species and compute true TMRCAs (log-space)."""
    from figures.utils import STDPOPSIM_V3_PARAMS, simulate_segment

    short_name = spec.split("_")[0]
    params = STDPOPSIM_V3_PARAMS.get(short_name)
    if params is None:
        return None

    true_cache = os.path.join(cache_dir, f"true_tmrcas_{spec}.npz")
    if os.path.exists(true_cache):
        return np.load(true_cache)["ytrues"]

    print(f"  Simulating {short_name} for ground truth ...")
    tss, meta = simulate_segment(
        seed=params["seed"], species_name=short_name,
        left=params.get("left"), right=params.get("right"),
        length=params.get("length"),
        num_samples=params["num_samples"],
        population_size=params.get("population_size"),
    )
    # Find the matching demography
    ts = None
    for t, m in zip(tss, meta):
        if spec.replace(short_name + "_", "") in m.get("id", ""):
            ts = t
            break
    if ts is None:
        ts = tss[0]

    pairs = [(i, j) for i in range(50) for j in range(i + 1, 50)]
    ytrues = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        ytrues = list(ex.map(_interp_worker, [(ts, a, b) for a, b in pairs]))
    ytrues = np.log(np.array(ytrues))
    np.savez_compressed(true_cache, ytrues=ytrues)
    return ytrues


def plot_fig4_singer(output_dir, cache_dir):
    """Singer KDE panels for stdpopsim v3 out-of-distribution species."""
    polegon_dir = os.path.join(SINGER_BASE, "stdpopsim-v3-species/polegon-output")
    pairs = [(i, j) for i in range(50) for j in range(i + 1, 50)]

    cache_path = os.path.join(cache_dir, "singer_v3_tmrcas.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        tmrcas_singer = d["tmrcas_singer"]
    else:
        tmrcas_singer = []
        for spec in V3_SPECIES:
            rep_results = []
            for rep in MCMC_REPS:
                fname = f"{spec}.{rep}.polegon.trees"
                fp = os.path.join(polegon_dir, fname)
                print(f"  Loading {fname}")
                ts = load_polegon_trees(fp)
                tmrcas = []
                for a, b in pairs:
                    tmrcas.append(np.log(interpolate_tmrcas_robust(ts, BIN_BP, SEQ_LEN, a, b)))
                rep_results.append(np.array(tmrcas))
            tmrcas_singer.append(np.array(rep_results))
        tmrcas_singer = np.array(tmrcas_singer)
        np.savez_compressed(cache_path, tmrcas_singer=tmrcas_singer)

    cols = min(len(V3_SPECIES), 4)
    n = len(V3_SPECIES)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 1.8 * rows), squeeze=False)

    for k, spec in enumerate(V3_SPECIES):
        singer_mean = tmrcas_singer[k].mean(0)
        yhat = singer_mean.flatten()

        ytrue = _simulate_v3_species(spec, cache_dir)
        if ytrue is None:
            print(f"  WARNING: no ground truth for {spec}, skipping")
            continue
        ytrue = ytrue.flatten()

        mask = np.isfinite(yhat) & np.isfinite(ytrue)
        yhat, ytrue = yhat[mask], ytrue[mask]
        ytrue_d = TIMES[np.clip(np.searchsorted(TIMES, ytrue, side="right") - 1, 0, len(TIMES) - 1)]

        mse_val = float(np.mean((yhat - ytrue_d) ** 2))
        x = robust_grid(yhat, ytrue_d, n=512)
        p_true = kde_pdf(ytrue_d, x)
        p_pred = kde_pdf(yhat, x)
        kl_val = np.log10(kl_divergence(p_true, p_pred, x))

        r, c = k // cols, k % cols
        ax = axes[r, c]
        ax.plot(x, p_true, color="black", label="True KDE")
        ax.plot(x, p_pred, color="darkblue", label="Singer KDE")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 16.2)
        ax.grid(alpha=0.3)
        set_loge_power10_ticks(ax, 0, 16.2)
        ax.set_title(spec.replace("_", " "), loc="left", fontsize=9)
        ax.text(0.98, 0.97, f"MSE = {mse_val:.3g}\nlog\u2081\u2080(KL) = {kl_val:.3g}",
                ha="right", va="top", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2), fontsize=8)
        if k == 0:
            ax.legend(loc="best", fontsize=7)
        if c == 0:
            ax.set_ylabel("Density")
        else:
            ax.tick_params(axis="y", labelleft=False)
        if r == rows - 1:
            ax.set_xlabel("TMRCA (generations)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    plt.tight_layout()
    out = os.path.join(output_dir, "singer_stdpopsim_v3.png")
    fig.savefig(out, dpi=300)
    print(f"  Saved {out}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Fig 5: Singer coalescence-rate step plots (demography inference)
# -----------------------------------------------------------------------

DEMOGRAPHY_SPECIES = [
    {"key": "homsap", "title": r"$\mathit{H.\;sapiens}$",  "window_size": 2e3, "seq_length": 10e6},
    {"key": "aratha", "title": r"$\mathit{A.\;thaliana}$", "window_size": 2e3, "seq_length": 10e6},
    {"key": "bostau", "title": r"$\mathit{B.\;taurus}$",   "window_size": 2e3, "seq_length": 10e6},
]


def plot_fig5_singer(output_dir, cache_dir, fig5_cache_dir):
    """Singer coalescence-rate curves overlaid on demography inference plots."""
    polegon_dir = os.path.join(SINGER_BASE, "demographic-inference/polegon-output")
    num_pairs = 25
    num_time_windows = 40

    singer_tmrcas_flat = {}

    for cfg in DEMOGRAPHY_SPECIES:
        species_key = cfg["key"]
        cache_path = os.path.join(cache_dir, f"singer_demography_{species_key}.npz")

        if os.path.exists(cache_path):
            d = np.load(cache_path)
            if "singer_tmrcas_flat" in d:
                singer_tmrcas_flat[species_key] = d["singer_tmrcas_flat"]
                print(f"  Loaded cached {species_key}")
                continue
            # Legacy cache without raw TMRCAs — need to recompute
            print(f"  Legacy cache for {species_key}, recomputing...")

        true_tmrcas_all = []
        for rep in MCMC_REPS:
            fp = os.path.join(polegon_dir, f"{species_key}.{rep}.polegon.trees")
            print(f"  Loading {os.path.basename(fp)}")
            ts = load_polegon_trees(fp)
            tmrcas = []
            for i in range(num_pairs):
                for j in range(i + 1, num_pairs):
                    tmrcas.append(interpolate_tmrcas_robust(
                        ts, cfg["window_size"], cfg["seq_length"], i, j))
            true_tmrcas_all.append(np.array(tmrcas))

        flat = np.array(true_tmrcas_all).mean(0).flatten()
        np.savez_compressed(cache_path, singer_tmrcas_flat=flat)
        singer_tmrcas_flat[species_key] = flat

    SP_CFGS = {
        "homsap":  {"species": "HomSap", "demography": "Zigzag_1S14"},
        "aratha":  {"species": "AraTha", "demography": "SouthMiddleAtlas_1D17"},
        "bostau":  {"species": "BosTau", "demography": "HolsteinFriesian_1M13"},
    }

    import stdpopsim
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, cfg in zip(axes, DEMOGRAPHY_SPECIES):
        species_key = cfg["key"]

        ts_path = os.path.join(fig5_cache_dir, f"{species_key}.ts")
        true_path = os.path.join(fig5_cache_dir, f"{species_key}_true_tmrcas.npy")
        tmrca_path = os.path.join(fig5_cache_dir, f"tmrca_{species_key}.npz")

        if not os.path.exists(ts_path):
            raise FileNotFoundError(
                f"Fig5 tree cache not found: {ts_path}. Run fig5 first."
            )
        ts = tskit.load(ts_path)
        tw = np.logspace(2, np.floor(np.log10(ts.max_time)), num_time_windows + 1)
        tw[0] = 0.0
        fine_time_grid = np.logspace(2, np.floor(np.log10(ts.max_time)), 1000)

        sp = SP_CFGS[species_key]
        species = stdpopsim.get_species(sp["species"])
        demogr = species.get_demographic_model(sp["demography"])
        pop_name = demogr.populations[0].name

        coalrate_ck, _ = demogr.model.debug().coalescence_rate_trajectory(
            lineages={pop_name: 2}, steps=fine_time_grid,
        )

        ytrue_cr = None
        if os.path.exists(true_path):
            true_tmrcas = np.load(true_path)
            ytrue_cr = coalescence_rates(true_tmrcas.flatten(), tw)

        yhat_cxt_cr = None
        if os.path.exists(tmrca_path):
            tmrca = np.load(tmrca_path)["tmrca"]
            yhat_cxt_cr = coalescence_rates(np.exp(tmrca.flatten()), tw)

        singer_cr = coalescence_rates(singer_tmrcas_flat[species_key], tw)

        ax.plot(fine_time_grid, coalrate_ck, "-", color="black", label="Expectation")
        if ytrue_cr is not None:
            ax.step(tw[:-1], ytrue_cr, where="post", color="firebrick", label="Inference limit")
        if yhat_cxt_cr is not None:
            ax.step(tw[:-1], yhat_cxt_cr, where="post", color="dodgerblue", label="cxt")
        ax.step(tw[:-1], singer_cr, where="post", color="darkblue",
                label="Singer+Polegon", linestyle="--")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time (Generations)")
        desc = demogr.description.split("(")[0].rstrip()
        ax.set_title(f"{cfg['title']}\n{desc}", loc="left")
        ax.grid(True)

    axes[0].set_ylabel("IICR / 2")
    axes[-1].legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    out = os.path.join(output_dir, "figure5_demography.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir-main", default="figures/output/main")
    parser.add_argument("--output-dir-supp", default="figures/output/supplementary")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/singer")
    parser.add_argument("--fig4-cache-dir", default="figures/output/main/cache/fig4")
    parser.add_argument("--fig5-cache-dir", default="figures/output/main/cache/fig5")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir_main, exist_ok=True)
    os.makedirs(args.output_dir_supp, exist_ok=True)

    print("=== Fig 2: Singer TMRCA scatter (constant & sawtooth) ===")
    plot_fig2_singer(args.output_dir_main, args.cache_dir)

    print("\n=== Fig 4: Singer KDE panels (stdpopsim v3) ===")
    plot_fig4_singer(args.output_dir_main, args.cache_dir)

    print("\n=== Fig 5: Singer demography coalescence rates ===")
    plot_fig5_singer(args.output_dir_main, args.cache_dir, args.fig5_cache_dir)

    print("\nAll Singer figures done.")


if __name__ == "__main__":
    main()
