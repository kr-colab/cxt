#!/usr/bin/env python3
"""
SINGER runtime benchmark for Fig S6.

Produces a whitespace-delimited file compatible with figS6 load_singer_data().
Format (tab-separated, no header or with header):

  mutation_rate  sample_size  replicate  col3  col4  col5  runtime_sec

Where:
  - mutation_rate: e.g. 3.5e-09 or 1.3e-08 (recombination/mutation rate used)
  - sample_size: number of diploid individuals
  - replicate: region/replicate index (0, 1, 2, ...)
  - col3, col4, col5: optional (stopping_iter, r_hat, ess from SINGER output)
  - runtime_sec: wall-clock seconds for SINGER to complete

FigS6 computes n_pairs = 2 * sample_size * (2 * sample_size - 1) // 2
(haploid pairs from diploids).

This script is a TEMPLATE. Replace run_singer_on_tree_sequence() with your
actual SINGER invocation. The revision data came from running SINGER on
simulated mosquito data at various sample sizes and mutation rates.

Usage:
  python -m scripts.benchmarks.benchmark_singer_runtime [--output PATH]
"""

import argparse
import os
import time


def run_singer_on_tree_sequence(ts_path: str, mutation_rate: float, **kwargs) -> float:
    """
    Run SINGER on a tree sequence and return wall-clock runtime in seconds.

    TEMPLATE: Implement this function to invoke your SINGER pipeline.
    Example structure:
      1. Load tree sequence from ts_path
      2. Convert to input format SINGER expects
      3. t0 = time.perf_counter()
      4. Run SINGER (subprocess or library call)
      5. return time.perf_counter() - t0

    Optional return: (runtime_sec, stopping_iter, r_hat, ess) for extra columns.
    """
    raise NotImplementedError(
        "Replace this with your SINGER invocation. "
        "See revision data at /sietch_colab/data_share/cxt/singer-benchmarks/mosquitos-runtime/singer_timing.txt"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/sietch_colab/data_share/cxt/singer-benchmarks/mosquitos-runtime/singer_timing.txt",
        help="Output file path",
    )
    parser.add_argument(
        "--data-dir",
        default="/sietch_colab/data_share/cxt/mosquito/benchmarks",
        help="Directory with tree sequences",
    )
    parser.add_argument(
        "--mutation-rates",
        type=float,
        nargs="+",
        default=[3.5e-9, 1.3e-8],
        help="Mutation/recombination rates to benchmark",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25, 30, 35, 40, 45, 50],
        help="Diploid sample sizes",
    )
    parser.add_argument("--replicates", type=int, default=3, help="Replicates per config")
    parser.add_argument("--header", action="store_true", help="Write header line")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    lines = []
    if args.header:
        lines.append(
            "rec_rate\tnum_diploids\tregion\tstopping_iter\tr_hat\tess\twalltime_sec"
        )

    for mu in args.mutation_rates:
        for sample_size in args.sample_sizes:
            for rep in range(args.replicates):
                ts_name = f"benchmark_samples{sample_size}_region{rep}.trees"
                ts_path = os.path.join(args.data_dir, ts_name)
                if not os.path.exists(ts_path):
                    print(f"[SKIP] {ts_path}")
                    continue

                print(f"[RUN] mu={mu:.1e}, n={sample_size}, rep={rep}")
                try:
                    result = run_singer_on_tree_sequence(ts_path, mu)
                    if isinstance(result, (list, tuple)):
                        runtime, si, rhat, ess = result[0], result[1], result[2], result[3]
                    else:
                        runtime = float(result)
                        si, rhat, ess = 0, 0.0, 0.0
                    lines.append(f"{mu}\t{sample_size}\t{rep}\t{si}\t{rhat}\t{ess}\t{runtime}")
                    print(f"  {runtime:.1f}s")
                except NotImplementedError:
                    print("  [TEMPLATE] Implement run_singer_on_tree_sequence()")
                    return
                except Exception as e:
                    print(f"  [ERROR] {e}")

    if lines:
        with open(args.output, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
