"""Verify the Rdl selective-sweep signal across all five Ag1000G populations.

Runs cxt inference from scratch on the RDL region (25.1–25.6 Mb on chr2L)
for Mali, Burkina Faso, Cameroon, Uganda, and Ghana using the w200_wmissing
model (adapter variant for Ghana).

Requires:
  - Ag1000G tree sequence: ~/cxt_paper_archive/ag1000g/gamb.2L.gff.dated.ne.trees
  - Accessibility mask:     ~/cxt_paper_archive/ag1000g/agp3.is_accessible.txt.npz
  - Checkpoints:            set CXT_CHECKPOINT_CACHE or BASE_DIR

The expected signal is a sharp TMRCA trough at the Rdl locus (25.36–25.43 Mb),
consistent with a selective sweep driven by insecticide resistance.
"""

import os
import sys
import time
import gc
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

_BASE_DIR = os.environ.get("BASE_DIR", "/sietch_colab/data_share/cxt_scratch")
if "CXT_CHECKPOINT_CACHE" not in os.environ:
    os.environ["CXT_CHECKPOINT_CACHE"] = os.path.join(_BASE_DIR, "checkpoints")

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import tskit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import cxt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ARCHIVE = os.environ.get(
    "CXT_PAPER_ARCHIVE",
    os.path.join(os.path.expanduser("~"), "cxt_paper_archive"),
)
AG1000G_DIR = os.environ.get("AG1000G_DATA_DIR", os.path.join(_ARCHIVE, "ag1000g"))
AG1000G_ACCESSIBILITY = os.environ.get(
    "AG1000G_ACCESSIBILITY",
    os.path.join(_ARCHIVE, "ag1000g", "agp3.is_accessible.txt.npz"),
)
TS_CACHE_DIR = os.path.join(SCRIPT_DIR, "cache_rdl")

POPULATIONS = {
    "Mali": "Mali",
    "BurkinaFaso": "Burkina Faso",
    "Cameroon": "Cameroon",
    "Uganda": "Uganda",
    "Ghana": "Ghana",
}
POP_ORDER = ["Mali", "BurkinaFaso", "Cameroon", "Uganda", "Ghana"]
POP_COLORS = {
    "Mali": "#E74C3C",
    "BurkinaFaso": "#F39C12",
    "Cameroon": "#2ECC71",
    "Uganda": "#3498DB",
    "Ghana": "#9B59B6",
}

MUTATION_RATE = 3.5e-9
BLOCK_SIZE = int(0.1e6)
STEP_BP = 200
N_INDIVIDUALS = 25

REGION_START = 25_100_000
REGION_END = 25_600_000
RDL_START = 25_363_652
RDL_END = 25_434_556

RDL_BLOCKS = [(int(i * BLOCK_SIZE), int((i + 1) * BLOCK_SIZE))
              for i in range(REGION_START // BLOCK_SIZE, REGION_END // BLOCK_SIZE)]

fmt_mb = FuncFormatter(lambda v, _: f"{v / 1e6:.2f}")


def _extract_population_ts(full_ts, country_name, pop_key,
                           n_individuals=N_INDIVIDUALS):
    """Filter 2La=1 heterozygotes for a given country, with caching."""
    ts_path = os.path.join(TS_CACHE_DIR, f"ts_{pop_key.lower()}.trees")
    if os.path.exists(ts_path):
        return tskit.load(ts_path)

    ids = []
    for ind in full_ts.individuals():
        meta = json.loads(ind.metadata.decode("ascii"))
        if meta["country"] == country_name and int(meta["2La"]) == 1:
            ids.append(ind.nodes)
    n_use = min(len(ids), n_individuals)
    if n_use < n_individuals:
        print(f"    Warning: only {n_use} inv-het individuals for {country_name}")
    ts_pop = full_ts.simplify(samples=np.concatenate(ids[:n_use]))
    ts_pop.dump(ts_path)
    return ts_pop


def _infer_rdl(ts_pop, model, missingness_bitmask, adapter=None):
    """Run cxt inference on the 5 RDL-region blocks and assemble genome."""
    pivot_pairs = [tuple(ind.nodes) for ind in ts_pop.individuals()]

    devices = [DEVICE]
    if torch.cuda.device_count() > 1:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    tmrca, index_map = cxt.translate(
        ts_pop, model,
        pivot_pairs=pivot_pairs,
        blocks=RDL_BLOCKS,
        devices=devices,
        B_per_device=min(512, len(pivot_pairs)),
        B=min(512, len(pivot_pairs)),
        build_workers=min(36, os.cpu_count() or 4),
        mutation_rate=MUTATION_RATE,
        missingness_bitmask=missingness_bitmask,
        adapter=adapter,
    )

    n_pairs = len(pivot_pairs)
    genomes = []
    for i in range(n_pairs):
        mask = index_map[:, 1] == i
        genomes.append(tmrca[:, mask, :].mean(0).flatten())
    return np.exp(np.array(genomes))


def main():
    t0_total = time.time()

    print(f"Device:      {DEVICE}")
    print(f"Checkpoints: {os.environ['CXT_CHECKPOINT_CACHE']}")
    print(f"AG1000G:     {AG1000G_DIR}")
    print(f"TS cache:    {TS_CACHE_DIR}")
    print(f"RDL blocks:  {len(RDL_BLOCKS)} × {BLOCK_SIZE // 1000}kb "
          f"= {REGION_START / 1e6:.1f}–{REGION_END / 1e6:.1f} Mb")
    print("=" * 60)

    os.makedirs(TS_CACHE_DIR, exist_ok=True)

    # -- Load input data --
    tree_path = os.path.join(AG1000G_DIR, "gamb.2L.gff.dated.ne.trees")
    print(f"Loading chr2L tree sequence ({tree_path}) ...")
    t0 = time.time()
    full_ts = tskit.load(tree_path)
    print(f"  {full_ts.num_samples} samples, {full_ts.num_sites} sites, "
          f"{full_ts.num_trees} trees  ({time.time() - t0:.1f}s)")

    print(f"Loading accessibility mask ...")
    unaccessible_bitmask = ~np.load(AG1000G_ACCESSIBILITY)["access_2L"]

    # -- Load models --
    print(f"\nLoading w200_wmissing model ...")
    base_model = cxt.load_model("w200_wmissing", device=DEVICE)

    print(f"Loading w200_wmissing_adapter model ...")
    adapter_wrapped = cxt.load_model("w200_wmissing_adapter", device=DEVICE)

    # -- Run inference per population --
    rdl_data = {}
    timings = {}

    for pop_key in POP_ORDER:
        country_name = POPULATIONS[pop_key]
        print(f"\n{'─' * 60}")
        print(f"  {pop_key} ({country_name})")
        print(f"{'─' * 60}")

        ts_pop = _extract_population_ts(full_ts, country_name, pop_key)
        print(f"  Extracted: {ts_pop.num_samples} samples, "
              f"{ts_pop.num_sites} sites")

        t0 = time.time()
        if pop_key == "Ghana":
            ts_pop = ts_pop.simplify(samples=np.arange(0, 10))
            print(f"  Adapter mode: simplified to {ts_pop.num_samples} samples")
            genome = _infer_rdl(
                ts_pop, adapter_wrapped.backbone, unaccessible_bitmask,
                adapter=adapter_wrapped.adapter)
        else:
            genome = _infer_rdl(
                ts_pop, base_model, unaccessible_bitmask)

        elapsed = time.time() - t0
        rdl_data[pop_key] = genome
        timings[pop_key] = elapsed
        print(f"  Result: {genome.shape}  ({elapsed:.1f}s)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del base_model, adapter_wrapped, full_ts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -- Plot --
    n_bins = (REGION_END - REGION_START) // STEP_BP
    base_x = np.arange(n_bins) * STEP_BP + REGION_START
    n_pops = len(POP_ORDER)

    fig, axes = plt.subplots(n_pops, 1, figsize=(10, 2.4 * n_pops),
                             sharex=True, sharey=True)
    if n_pops == 1:
        axes = [axes]

    for i, pop_key in enumerate(POP_ORDER):
        ax = axes[i]
        genome = np.atleast_2d(rdl_data[pop_key])
        name = POPULATIONS[pop_key]
        color = POP_COLORS[pop_key]

        mean_t = np.nanmean(genome, axis=0)
        std_t = np.nanstd(genome, axis=0)
        n = genome.shape[0]

        for j in range(n):
            alpha = 0.30 + 0.40 * (j / max(1, n - 1)) if n > 1 else 0.7
            lw = 0.6 + 0.4 * (j % 5 == 0)
            ax.plot(base_x, genome[j], lw=lw, color=color, alpha=alpha,
                    zorder=0)

        eps = 1e-12
        lower = np.clip(mean_t - std_t, eps, None)
        upper = np.clip(mean_t + std_t, eps, None)
        ax.fill_between(base_x, lower, upper, color=color, alpha=0.25,
                        zorder=1)
        ax.plot(base_x, lower, ls="--", lw=0.8, color=color,
                alpha=0.7, zorder=2)
        ax.plot(base_x, upper, ls="--", lw=0.8, color=color,
                alpha=0.7, zorder=2)
        ax.plot(base_x, mean_t, lw=1.0, color=color, zorder=3,
                label=f"Mean ({n} pairs)")

        ax.axvspan(RDL_START, RDL_END, color="crimson", alpha=0.12, zorder=0)
        mid = (RDL_START + RDL_END) / 2
        valid = mean_t[np.isfinite(mean_t) & (mean_t > 0)]
        if valid.size:
            y_arrow = np.exp(
                (np.log(valid.min()) + np.log(valid.max())) / 3)
            ax.annotate(
                "", xy=(RDL_END, y_arrow), xytext=(RDL_START, y_arrow),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
                zorder=6)
            ax.text(mid, y_arrow * 1.08, "RDL", fontsize=9, ha="center",
                    va="bottom", fontweight="bold", zorder=6)

        ax.set_yscale("log")
        ax.set_xlim(REGION_START, REGION_END)
        ax.set_ylim(0.5e2, 5e6)
        ax.grid(True, alpha=0.2, which="both", linestyle="--")
        ax.set_title(f"{name}  ({timings[pop_key]:.0f}s)",
                     fontsize=11, loc="left", fontweight="bold")
        ax.set_ylabel("TMRCA (gen.)")
        ax.xaxis.set_major_formatter(fmt_mb)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Position on chr2L (Mb)", fontsize=10)

    elapsed_total = time.time() - t0_total
    fig.suptitle(
        f"Rdl locus verification (from scratch)  |  5 populations"
        f"  |  {DEVICE}  |  {elapsed_total:.0f}s",
        fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, "verify_rdl_5pops.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for pop_key in POP_ORDER:
        g = rdl_data[pop_key]
        n = g.shape[0]
        print(f"  {POPULATIONS[pop_key]:15s}  {n:2d} pairs  {timings[pop_key]:6.1f}s")
    print(f"\n  Total: {elapsed_total:.1f}s")
    print(f"  Figure: {out_path}")


if __name__ == "__main__":
    main()
