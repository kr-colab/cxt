"""
Shared utilities for paper figure generation.

Contains consolidated plotting, simulation, and SMC++ helper functions
used across multiple figure scripts.
"""

import copy
import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tmrca_scatter(
    yhat_tmrca,
    ytrue_tmrca,
    filename=None,
    subtitle=None,
    tool=r'$\mathbf{cxt}$',
    stackit=False,
    fontsize=12,
    ax=None,
    return_mse=False,
):
    """
    Hexbin of predicted vs. true TMRCA (in generations).
    Plots in ln-space, ticks labeled as 10^k.
    If *ax* is provided, draws into it; otherwise creates a new figure.
    """
    def _to_array(x):
        if stackit and isinstance(x, (list, tuple)):
            return np.stack(x).mean(0)
        return np.asarray(x)

    eps = 1e-12
    ytrue = np.clip(_to_array(ytrue_tmrca), eps, None)
    yhat = np.clip(_to_array(yhat_tmrca), eps, None)
    ytrue_ln, yhat_ln = np.log(ytrue), np.log(yhat)

    mask = np.isfinite(ytrue_ln) & np.isfinite(yhat_ln)
    ytrue_ln, yhat_ln = ytrue_ln[mask], yhat_ln[mask]

    mse = float(np.mean((yhat_ln - ytrue_ln) ** 2)) if ytrue_ln.size else float("nan")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        created_fig = True
    plt.rcParams.update({"font.size": fontsize})

    ax.hexbin(ytrue_ln, yhat_ln, gridsize=120, cmap="plasma", bins="log", alpha=0.5)

    if ytrue_ln.size:
        mn = float(min(ytrue_ln.min(), yhat_ln.min()))
        mx = float(max(ytrue_ln.max(), yhat_ln.max()))
        if mx == mn:
            mx = mn + 1.0
        ax.plot([mn, mx], [mn, mx], c="black", ls="-", lw=0.5)

        if ytrue_ln.size >= 2 and np.std(ytrue_ln) > 0:
            slope, intercept = np.polyfit(ytrue_ln, yhat_ln, 1)
            xx = np.linspace(mn, mx, 100)
            ax.plot(xx, slope * xx + intercept, c="black", ls=":", lw=1)

        ln10 = np.log(10.0)
        exp_min = int(np.floor(mn / ln10))
        exp_max = int(np.ceil(mx / ln10))
        exps = np.arange(exp_min, exp_max + 1)
        ticks = exps * ln10
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"$10^{{{k}}}$" for k in exps])
        ax.set_yticklabels([f"$10^{{{k}}}$" for k in exps])

    title = tool + (": " + subtitle if subtitle else "")
    ax.set_title(title, loc="left", fontsize=fontsize)
    ax.text(0.60, 0.10, f"MSE: {mse:.4f}", transform=ax.transAxes, va="top", fontsize=fontsize)
    ax.set_xlabel("True Time (Generations)")
    ax.set_ylabel("Predicted Time (Generations)")
    ax.grid(alpha=0.25)

    if filename:
        ax.figure.tight_layout()
        ax.figure.savefig(filename, dpi=300)

    return (ax, mse) if return_mse else ax


# ---------------------------------------------------------------------------
# stdpopsim simulation helpers
# ---------------------------------------------------------------------------

STDPOPSIM_V2_PARAMS = {
    "AedAeg": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "population_size": 1_000_000},
    "AnaPla": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25},
    "AnoCar": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "population_size": 3_050_000},
    "AnoGam": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25},
    "AraTha": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "genetic_map_tuple": [None, "SalomeAveraged_TAIR10"]},
    "CaeEle": {"seed": 103370001, "left": 10e6, "right": 11e6, "num_samples": 25, "population_size": 10000, "genetic_map_tuple": [None, "RockmanRIAIL_ce11"]},
    "DroMel": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25},
    "DroSec": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "population_size": 100000},
    "GasAcu": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "population_size": 10000},
    "HelAnn": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "population_size": 673_968},
    "HomSap": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "genetic_map_tuple": [None, "HapMapII_GRCh38"]},
    "BosTau": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25},
    "CanFam": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "population_size": 13000, "genetic_map_tuple": [None, "Campbell2016_CanFam3_1"]},
    "PanTro": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25},
    "PapAnu": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "genetic_map_tuple": [None, "Pyrho_PAnubis1_0"]},
    "PonAbe": {"seed": 103370001, "left": 20e6, "right": 21e6, "num_samples": 25, "genetic_map_tuple": [None, "NaterPA_PonAbe3"]},
}

STDPOPSIM_V3_PARAMS = {
    "MusMus": {"seed": 103370001, "length": 1e6, "num_samples": 25, "population_size": 500_000},
    "RatNor": {"seed": 103370001, "length": 1e6, "num_samples": 25, "population_size": 1.24e5},
    "GorGor": {"seed": 103370001, "length": 1e6, "num_samples": 25, "population_size": 25200},
    "OrySat": {"seed": 103370001, "length": 1e6, "num_samples": 25, "population_size": 46875},
    "SusScr": {"seed": 103370001, "length": 1e6, "num_samples": 25, "population_size": 270_000},
    "PhoSin": {"seed": 103370001, "length": 1e6, "num_samples": 25, "population_size": 3500},
}

EXCLUDED_DEMOGRAPHIES = {
    "Multi-population model of ancient Eurasia",
    "Out-of-Africa with archaic admixture into Papuans",
    "Multi-population model of ancient Europe",
}


def get_sampling_populations(model):
    """Return populations that allow present-day sampling."""
    populations = []
    for pop in model.populations:
        if hasattr(pop, "default_sampling_time"):
            if isinstance(pop.default_sampling_time, float) and pop.default_sampling_time > 0:
                continue
            if pop.allow_samples:
                populations.append(pop)
        elif pop.allow_samples:
            populations.append(pop)
    return populations


def equal_sample_counts(sampling_populations: List, num_samples: int = 25) -> Dict[str, int]:
    """Distribute *num_samples* equally across *sampling_populations*."""
    n = len(sampling_populations)
    base, remainder = divmod(num_samples, n)
    counts = {pop.name: base for pop in sampling_populations}
    for i in range(remainder):
        counts[sampling_populations[i].name] += 1
    return counts


def simulate_segment(
    seed,
    species_name,
    genetic_map_tuple=None,
    left=20e6,
    right=21e6,
    length=None,
    num_samples=25,
    population_size=None,
    metadata_only=False,
):
    """
    Simulate tree sequences for all demographic models of a stdpopsim species.

    Returns (tree_sequences, metadata) or metadata if *metadata_only*.
    """
    import stdpopsim

    if genetic_map_tuple is None:
        genetic_map_tuple = [None]

    tree_sequences, metadata = [], []

    for genetic_map in genetic_map_tuple:
        species = stdpopsim.get_species(species_name)
        chromosome = species.genome.chromosomes[0]

        demographic_models = [
            m for m in species.demographic_models
            if m.description not in EXCLUDED_DEMOGRAPHIES
        ]
        if not demographic_models:
            ne = population_size if population_size is not None else 20_000
            demographic_models = [stdpopsim.PiecewiseConstantSize(ne)]

        for demography in demographic_models:
            item = {
                "seed": seed,
                "species_name": species_name,
                "genetic_map": genetic_map,
                "demography": demography.description,
                "id": demography.id,
                "num_sites": 0,
            }

            populations = get_sampling_populations(demography)
            samples = equal_sample_counts(populations, num_samples=num_samples)
            item["samples"] = samples

            if genetic_map is not None:
                contig = species.get_contig(
                    chromosome.id, left=left, right=right,
                    mutation_rate=demography.mutation_rate, genetic_map=genetic_map,
                )
            elif left is None or right is None:
                contig = species.get_contig(length=length, mutation_rate=demography.mutation_rate)
            else:
                contig = species.get_contig(
                    chromosome.id, left=left, right=right,
                    mutation_rate=demography.mutation_rate,
                )

            item["mutation_rate"] = contig.mutation_rate
            item["recombination_rate"] = contig.recombination_map.rate[0]
            item["ratio"] = item["recombination_rate"] / item["mutation_rate"]

            engine = stdpopsim.get_engine("msprime")
            if not metadata_only:
                ts = engine.simulate(demography, contig, samples, seed=seed)
                item["num_sites"] = ts.num_sites
                tree_sequences.append(ts)
            metadata.append(item)

    return metadata if metadata_only else (tree_sequences, metadata)


# ---------------------------------------------------------------------------
# SMC++ helpers
# ---------------------------------------------------------------------------

def _which_container():
    """Return 'singularity' or 'apptainer' executable name."""
    from shutil import which as _which
    for exe in ("singularity", "apptainer"):
        if _which(exe):
            return exe
    raise RuntimeError("Neither 'singularity' nor 'apptainer' is on PATH.")


def _ensure_sif(image_ref="docker://terhorst/smcpp:latest", sif_path=None):
    """Download the SMC++ SIF once and return its absolute path."""
    exe = _which_container()
    if sif_path is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "smcpp")
        os.makedirs(cache_dir, exist_ok=True)
        sif_path = os.path.join(cache_dir, "smcpp_latest.sif")
    elif os.path.isdir(sif_path):
        sif_path = os.path.join(sif_path, "smcpp_latest.sif")
    os.makedirs(os.path.dirname(os.path.abspath(sif_path)), exist_ok=True)

    if not os.path.exists(sif_path):
        tmp_sif = sif_path + ".tmp"
        if os.path.exists(tmp_sif):
            os.remove(tmp_sif)
        subprocess.run([exe, "pull", tmp_sif, image_ref], check=True)
        os.replace(tmp_sif, sif_path)

    return os.path.abspath(sif_path)


def analyze_ts_with_smcpp(
    ts,
    pair=(0, 1),
    mu=None,
    tmp_dir="tmp",
    bin_bp=2000,
    n_bins=500,
    chrom="1",
    cores=16,
    sif_path=None,
    image_ref="docker://terhorst/smcpp:latest",
):
    """
    Run SMC++ estimation (once for the first pair) and posterior decode.

    Reuses the fitted model for subsequent pairs.
    Returns dict with tmrca_raw, hidden_states, gamma, sites, etc.
    """
    if mu is None:
        raise ValueError("mu (mutation rate) required")

    container_exe = _which_container()
    sif_path = _ensure_sif(image_ref=image_ref, sif_path=sif_path)
    os.makedirs(tmp_dir, exist_ok=True)

    vcf = os.path.join(tmp_dir, "data.vcf")
    vcfgz = vcf + ".gz"
    smc = os.path.join(tmp_dir, f"pair_{pair[0]}_{pair[1]}.smc.gz")
    model_json = os.path.join(tmp_dir, "model.final.json")
    npz_path = os.path.join(tmp_dir, f"posterior_{pair[0]}_{pair[1]}.npz")

    def _run(*args, cwd=None):
        cmd = [container_exe, "run", "--bind", os.path.abspath(tmp_dir), sif_path, *args]
        subprocess.run(cmd, check=True, cwd=cwd)

    def _write_vcf():
        names = [f"tsk_{i}" for i in range(25)]
        with open(vcf, "w", newline="\n") as f:
            f.write(ts.as_vcf())
        with open(vcfgz, "wb") as outgz:
            subprocess.run(["bgzip", "-c", vcf], check=True, stdout=outgz)
        os.remove(vcf)
        subprocess.run(["tabix", "-f", "-p", "vcf", vcfgz], check=True)
        return names

    if not os.path.exists(model_json):
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        names = _write_vcf()
        pop = "Pop0:" + ",".join(names)
        i, j = pair
        _run("vcf2smc", "-d", f"tsk_{i}", f"tsk_{i}", vcfgz, smc, chrom, pop)
        _run("estimate", "--cores", str(int(cores)), str(mu), os.path.basename(smc), cwd=tmp_dir)
    else:
        if not (os.path.exists(vcfgz) and os.path.exists(vcfgz + ".tbi")):
            _write_vcf()
        i, j = pair
        pop = "Pop0:" + ",".join([f"tsk_{k}" for k in range(25)])
        _run("vcf2smc", "-d", f"tsk_{i}", f"tsk_{i}", vcfgz, smc, chrom, pop)

    _run("posterior", os.path.basename(model_json), os.path.basename(npz_path),
         os.path.basename(smc), cwd=tmp_dir)

    data = np.load(npz_path)
    with open(model_json) as f:
        model_dict = json.load(f)
    N0 = float(model_dict["model"]["N0"])

    hs = data["hidden_states"].astype(float)
    site_key = [k for k in data.files if k.endswith("_sites")][0]
    gamma_key = site_key[:-6]
    sites = data[site_key].astype(float)
    gamma = data[gamma_key].astype(float)

    # SMC++ posterior can produce all-NaN gamma for certain model
    # parameterisations due to numerical underflow in the forward-backward
    # algorithm.  A tiny shrinkage of the y-spline toward its mean nudges the
    # parameters out of the degenerate region without materially changing the
    # demographic fit.
    if np.all(np.isnan(gamma)):
        y_orig = model_dict["model"]["y"]
        y_mean = float(np.mean(y_orig))
        for shrink in (0.99, 0.98, 0.97, 0.96, 0.90):
            y_adj = [y_mean + shrink * (yi - y_mean) for yi in y_orig]
            model_dict["model"]["y"] = y_adj
            perturbed_json = os.path.join(tmp_dir, "model.perturbed.json")
            with open(perturbed_json, "w") as f:
                json.dump(model_dict, f)
            _run("posterior", os.path.basename(perturbed_json),
                 os.path.basename(npz_path), os.path.basename(smc), cwd=tmp_dir)
            data = np.load(npz_path)
            gamma = data[gamma_key].astype(float)
            if not np.all(np.isnan(gamma)):
                hs = data["hidden_states"].astype(float)
                sites = data[site_key].astype(float)
                # Persist the working model so subsequent pairs use it too
                shutil.copy(perturbed_json, model_json)
                print(f"[smcpp] posterior NaN fixed with y-shrink={shrink}")
                break

    n = min(len(sites), gamma.shape[1])
    sites = np.where(np.isfinite(sites[:n]) & (sites[:n] > 0), sites[:n], 1.0)
    gamma = gamma[:, :n]

    K = gamma.shape[0]
    hs_fin = hs[np.isfinite(hs)]
    if len(hs_fin) == K + 1:
        t_rep = 0.5 * (hs_fin[:-1] + hs_fin[1:])
    elif len(hs_fin) == K:
        t_rep = np.empty(K, dtype=float)
        t_rep[:-1] = 0.5 * (hs_fin[:-1] + hs_fin[1:])
        t_rep[-1] = hs_fin[-1] + 0.5 * (hs_fin[-1] - hs_fin[-2]) if K >= 2 else hs_fin[-1]
    else:
        t_rep = hs_fin[:K]

    tmrca_raw = (t_rep @ gamma).astype(float)
    bounds = np.concatenate(([0.0], np.cumsum(sites)))
    mids = 0.5 * (bounds[:-1] + bounds[1:])

    return {
        "pair": pair,
        "tmrca_raw": tmrca_raw,
        "hidden_states": t_rep,
        "gamma": gamma,
        "sites": sites,
        "site_midpoints": mids,
        "N0": N0,
        "work_dir": os.path.abspath(tmp_dir),
        "sif_path": sif_path,
        "container": container_exe,
    }


def analyze_ts_with_smcpp_multi(
    ts,
    pairs=None,
    mu=None,
    tmp_dir="tmp_smcpp_run",
    bin_bp=2000,
    n_bins=500,
    chrom="1",
    cores=16,
    do_posterior=True,
    posterior_pair_index=0,
    sif_path=None,
    extra_vcf2smc_args=None,
):
    """
    Run SMC++ jointly on multiple pairs with a single estimate step.

    Optionally perform posterior decoding for one pair and grid the result.
    """
    if pairs is None:
        pairs = [(0, 1)]
    if mu is None:
        raise ValueError("mu (mutation rate) required")
    if sif_path is not None and not os.path.exists(sif_path):
        raise FileNotFoundError(f"sif_path not found: {sif_path}")
    if sif_path is not None:
        sif_path = os.path.abspath(sif_path)

    if os.path.exists(tmp_dir):
        abs_tmp = os.path.abspath(tmp_dir)
        if abs_tmp in ("/", os.path.abspath(".")):
            raise RuntimeError(f"Refusing to delete dangerous path: {abs_tmp}")
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    container_exe = _which_container()

    def _sing(*args, cwd=None):
        cmd = [container_exe, "run", "--bind", os.path.abspath(tmp_dir), sif_path, *args]
        subprocess.run(cmd, check=True, cwd=cwd)

    vcf = os.path.join(tmp_dir, "data.vcf")
    vcfgz = vcf + ".gz"
    model_js = os.path.join(tmp_dir, "model.final.json")

    vcf_str = ts.as_vcf()
    sample_names = None
    for line in vcf_str.splitlines():
        if line.startswith("#CHROM"):
            sample_names = line.rstrip("\n").split("\t")[9:]
            break
    if not sample_names:
        raise RuntimeError("Could not parse sample names from VCF header.")

    with open(vcf, "w", newline="\n") as f:
        f.write(vcf_str)
    with open(vcfgz, "wb") as outgz:
        subprocess.run(["bgzip", "-c", vcf], check=True, stdout=outgz)
    os.remove(vcf)
    subprocess.run(["tabix", "-f", "-p", "vcf", vcfgz], check=True)

    pop = "Pop0:" + ",".join(sample_names)
    smc_paths = []
    for i, j in pairs:
        out_smc = os.path.join(tmp_dir, f"pair_{i}_{j}.smc.gz")
        args = ["vcf2smc", "-d", sample_names[i], sample_names[j]]
        if extra_vcf2smc_args:
            args.extend(extra_vcf2smc_args)
        args.extend([vcfgz, out_smc, chrom, pop])
        _sing(*args)
        smc_paths.append(out_smc)

    smc_basenames = [os.path.basename(p) for p in smc_paths]
    _sing("estimate", "--cores", str(int(cores)), str(mu), "--knots", "24", *smc_basenames, cwd=tmp_dir)

    result = {
        "model_json": os.path.abspath(model_js),
        "smc_files": [os.path.abspath(p) for p in smc_paths],
        "work_dir": os.path.abspath(tmp_dir),
    }

    if do_posterior:
        pi, pj = pairs[posterior_pair_index]
        smc_for_posterior = os.path.join(tmp_dir, f"pair_{pi}_{pj}.smc.gz")
        npz_path = os.path.join(tmp_dir, f"posterior_pair_{pi}_{pj}.npz")

        _sing("posterior", os.path.basename(model_js), os.path.basename(npz_path),
              os.path.basename(smc_for_posterior), cwd=tmp_dir)

        data = np.load(npz_path)
        with open(model_js) as f:
            N0 = float(json.load(f)["model"]["N0"])

        hs = data["hidden_states"].astype(float)
        site_key = [k for k in data.files if k.endswith("_sites")][0]
        gamma_key = site_key[:-6]
        sites = data[site_key].astype(float)
        gamma = data[gamma_key].astype(float)

        n = min(len(sites), gamma.shape[1])
        sites = np.where(np.isfinite(sites[:n]) & (sites[:n] > 0), sites[:n], 1.0)
        gamma = gamma[:, :n]

        K = gamma.shape[0]
        hs_fin = hs[np.isfinite(hs)]
        if len(hs_fin) == K + 1:
            t_rep = 0.5 * (hs_fin[:-1] + hs_fin[1:])
        elif len(hs_fin) == K:
            t_rep = np.empty(K, dtype=float)
            t_rep[:-1] = 0.5 * (hs_fin[:-1] + hs_fin[1:])
            t_rep[-1] = hs_fin[-1] + 0.5 * (hs_fin[-1] - hs_fin[-2]) if K >= 2 else hs_fin[-1]
        else:
            t_rep = hs_fin[:K]
        t_rep = 2.0 * N0 * np.asarray(t_rep, float)

        tmrca_raw = (t_rep @ gamma).astype(float)
        bounds = np.concatenate(([0.0], np.cumsum(sites)))
        mids = 0.5 * (bounds[:-1] + bounds[1:])

        L = bin_bp * n_bins
        x_grid = np.linspace(0, L - bin_bp, n_bins)
        valid = np.isfinite(tmrca_raw) & (tmrca_raw > 0) & np.isfinite(mids)
        if valid.sum() >= 2:
            log_grid = np.interp(x_grid, mids[valid], np.log(tmrca_raw[valid]))
            tmrca_grid = np.exp(log_grid)
        else:
            log_grid = np.full_like(x_grid, np.nan, float)
            tmrca_grid = np.full_like(x_grid, np.nan, float)

        result.update({
            "N0": N0,
            "x_grid": x_grid,
            "tmrca_grid": tmrca_grid,
            "log_tmrca_grid": log_grid,
            "tmrca_raw": tmrca_raw,
            "sites": sites,
            "site_midpoints": mids,
            "posterior_pair": (pi, pj),
        })

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def savefig(fig, name, output_dir, dpi=300, **kwargs):
    """Save a figure to *output_dir*/*name*, creating directories as needed."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    print(f"Saved {path}")
    plt.close(fig)
