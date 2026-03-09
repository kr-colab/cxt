#!/usr/bin/env python3
"""
SMC++ runtime benchmark for Fig S6.

Produces timing_log.jsonl with benchmark_complete events containing:
  sample_size, region, n_pairs, n_samples, n_diploids, sequence_length,
  timings: {setup_tmp_dir, vcf_export_and_write, bgzip_and_tabix,
            vcf2smc_all_pairs, smcpp_estimate, smcpp_posterior,
            posterior_total, total}

Matches revision/figure_mosquito/experiment_smcpp_time_benchmark.ipynb.
Requires: singularity/apptainer, bgzip, tabix, SMC++ SIF image.

Usage:
  python -m scripts.benchmarks.benchmark_smcpp_runtime [--output-dir DIR] [--data-dir PATH]
"""

import argparse
import json
import os
import shutil
import subprocess
import time

import tskit


def _which_container():
    for exe in ("singularity", "apptainer"):
        p = shutil.which(exe)
        if p:
            return exe
    raise RuntimeError("Neither singularity nor apptainer found on PATH")


def run_smcpp_benchmark(
    ts,
    pairs,
    mu,
    tmp_dir,
    chrom="1",
    cores=16,
    sif_path=None,
):
    """
    Run full SMC++ pipeline: vcf2smc (all pairs) -> estimate -> posterior.
    Returns timings dict compatible with figS6 load_smcpp_runs().
    """
    if sif_path is None:
        cache = os.path.join(os.path.expanduser("~"), ".cache", "smcpp")
        os.makedirs(cache, exist_ok=True)
        sif_path = os.path.join(cache, "smcpp_latest.sif")
        if not os.path.exists(sif_path):
            exe = _which_container()
            subprocess.run([exe, "pull", sif_path + ".tmp", "docker://terhorst/smcpp:latest"], check=True)
            os.rename(sif_path + ".tmp", sif_path)
    sif_path = os.path.abspath(sif_path)

    timings = {}
    t0 = time.perf_counter()
    last = t0

    def _mark(label, dt=None):
        nonlocal last
        if dt is None:
            now = time.perf_counter()
            dt = now - last
            last = now
        timings[label] = dt
        return dt

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    _mark("setup_tmp_dir")

    vcf = os.path.join(tmp_dir, "data.vcf")
    vcfgz = vcf + ".gz"
    model_json = os.path.join(tmp_dir, "model.final.json")

    vcf_str = ts.as_vcf()
    sample_names = None
    for line in vcf_str.splitlines():
        if line.startswith("#CHROM"):
            sample_names = line.rstrip("\n").split("\t")[9:]
            break
    if not sample_names:
        raise RuntimeError("Could not parse VCF sample names")

    with open(vcf, "w", newline="\n") as f:
        f.write(vcf_str)
    _mark("vcf_export_and_write")
    with open(vcfgz, "wb") as outgz:
        subprocess.run(["bgzip", "-c", vcf], check=True, stdout=outgz)
    os.remove(vcf)
    subprocess.run(["tabix", "-f", "-p", "vcf", vcfgz], check=True)
    _mark("bgzip_and_tabix")

    exe = _which_container()
    pop = "Pop0:" + ",".join(sample_names)
    vcf2smc_start = time.perf_counter()
    for i, j in pairs:
        out_smc = os.path.join(tmp_dir, f"pair_{i}_{j}.smc.gz")
        cmd = [
            exe, "run", "--bind", tmp_dir, sif_path,
            "vcf2smc", "-d", sample_names[i], sample_names[j],
            os.path.basename(vcfgz), os.path.basename(out_smc), chrom, pop,
        ]
        subprocess.run(cmd, check=True, cwd=tmp_dir, capture_output=True)
    timings["vcf2smc_all_pairs"] = time.perf_counter() - vcf2smc_start
    last = time.perf_counter()

    smc_basenames = [f"pair_{i}_{j}.smc.gz" for i, j in pairs]
    subprocess.run(
        [exe, "run", "--bind", tmp_dir, sif_path,
         "estimate", "--cores", str(int(cores)), str(mu), "--knots", "24"]
        + smc_basenames,
        check=True,
        cwd=tmp_dir,
        capture_output=True,
    )
    _mark("smcpp_estimate")

    pi, pj = pairs[0]
    npz_name = f"posterior_{pi}_{pj}.npz"
    post_start = time.perf_counter()
    subprocess.run(
        [exe, "run", "--bind", tmp_dir, sif_path,
         "posterior", "model.final.json", npz_name, f"pair_{pi}_{pj}.smc.gz"],
        check=True,
        cwd=tmp_dir,
        capture_output=True,
    )
    timings["smcpp_posterior"] = time.perf_counter() - post_start
    timings["posterior_total"] = timings["smcpp_posterior"]
    timings["posterior_setup"] = 0.0
    last = time.perf_counter()

    timings["total"] = time.perf_counter() - t0
    return timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="revision/figure_mosquito/tmp_smcpp_local_0",
        help="Output directory; timing_log.jsonl written here",
    )
    parser.add_argument(
        "--data-dir",
        default="/sietch_colab/data_share/cxt/mosquito/benchmarks",
        help="Directory with benchmark_samples{N}_region{R}.trees",
    )
    parser.add_argument("--mu", type=float, default=3.5e-9, help="Mutation rate")
    parser.add_argument("--sif-path", default=None, help="SMC++ SIF image path")
    parser.add_argument("--cores", type=int, default=48)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50],
    )
    parser.add_argument("--regions", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "timing_log.jsonl")

    existing = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("event_type") == "benchmark_complete":
                        existing.add((rec["sample_size"], rec["region"]))
                except json.JSONDecodeError:
                    pass

    for sample_size in args.sample_sizes:
        for region in args.regions:
            if (sample_size, region) in existing:
                print(f"[SKIP] samples={sample_size}, region={region}")
                continue

            path = os.path.join(
                args.data_dir,
                f"benchmark_samples{sample_size}_region{region}.trees",
            )
            if not os.path.exists(path):
                print(f"[SKIP] Missing: {path}")
                continue

            print(f"[RUN] samples={sample_size}, region={region}")
            ts = tskit.load(path).trim()
            n_diploids = ts.num_samples // 2
            pairs = [(i, i) for i in range(n_diploids)]
            n_pairs = len(pairs)

            def log_event(evt, ss, reg, step, dur, **kw):
                rec = {
                    "timestamp": time.time(),
                    "event_type": evt,
                    "sample_size": ss,
                    "region": reg,
                    "step": step,
                    "duration_sec": dur,
                    "n_pairs": n_pairs,
                    "n_samples": ts.num_samples,
                    "n_diploids": n_diploids,
                    "sequence_length": float(ts.sequence_length),
                    **kw,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

            log_event("benchmark_start", sample_size, region, "start", 0)

            tmp_dir = os.path.join(args.output_dir, f"tmp_s{sample_size}_r{region}")
            try:
                timings = run_smcpp_benchmark(
                    ts,
                    pairs=pairs,
                    mu=args.mu,
                    tmp_dir=tmp_dir,
                    cores=args.cores,
                    sif_path=args.sif_path,
                )
                log_event(
                    "benchmark_complete",
                    sample_size,
                    region,
                    "complete",
                    timings["total"],
                    timings=timings,
                    ran_posterior=True,
                )
                print(f"[DONE] {timings['total']:.1f}s")
            except Exception as e:
                print(f"[ERROR] {e}")
            finally:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
