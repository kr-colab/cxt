"""
Figure 8: Coalescent-time structure across the In(2L)a inversion on chr2L.

Shows mean TMRCA summaries for an outside background region (10-20 Mb),
the full inversion core interval, and two interior 0.5 Mb windows
positioned 1 Mb inside each breakpoint.

NOTE: Requires pre-computed genome-wide TMRCA caches from fig7 or
equivalent Ag1000G analysis.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


POPULATIONS = ["BurkinaFaso", "Mali", "Cameroon", "Ghana", "Uganda"]

INV2LA_LEFT = 20_524_058
INV2LA_RIGHT = 42_165_532
BACKGROUND_START = 10_000_000
BACKGROUND_END = 20_000_000
INNER_OFFSET = 1_000_000
INNER_WIDTH = 500_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="figures/output/main")
    parser.add_argument("--cache-dir", default="figures/output/main/cache/fig7",
                        help="Cache dir with genome-wide TMRCA from fig7")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    BIN_BP = 200

    summary = {"pop": [], "region": [], "mean": [], "se": []}

    for pop in POPULATIONS:
        cache_path = os.path.join(args.cache_dir, f"genome_{pop}.npz")
        if not os.path.exists(cache_path):
            print(f"Skipping {pop}: cache not found at {cache_path}")
            continue

        genome = np.load(cache_path)["genome"]
        tmrca = np.exp(genome)

        def _region_stats(start_bp, end_bp):
            i0 = max(0, int(start_bp // BIN_BP))
            i1 = min(tmrca.shape[1], int(end_bp // BIN_BP))
            vals = tmrca[:, i0:i1].flatten()
            vals = vals[np.isfinite(vals)]
            return np.mean(vals), np.std(vals) / np.sqrt(len(vals))

        regions = {
            "Outside (10-20 Mb)": (BACKGROUND_START, BACKGROUND_END),
            "Inversion core": (INV2LA_LEFT, INV2LA_RIGHT),
            "Inner proximal (+1 Mb)": (INV2LA_LEFT + INNER_OFFSET, INV2LA_LEFT + INNER_OFFSET + INNER_WIDTH),
            "Inner distal (-1 Mb)": (INV2LA_RIGHT - INNER_OFFSET - INNER_WIDTH, INV2LA_RIGHT - INNER_OFFSET),
        }

        for region_name, (s, e) in regions.items():
            m, se = _region_stats(s, e)
            summary["pop"].append(pop)
            summary["region"].append(region_name)
            summary["mean"].append(m)
            summary["se"].append(se)

    region_names = list(dict.fromkeys(summary["region"]))
    pops_found = list(dict.fromkeys(summary["pop"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(region_names))
    width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 1, len(pops_found)))

    for i, pop in enumerate(pops_found):
        means = [summary["mean"][j] for j in range(len(summary["pop"])) if summary["pop"][j] == pop]
        ses = [summary["se"][j] for j in range(len(summary["pop"])) if summary["pop"][j] == pop]
        ax.bar(x + i * width, means, width, label=pop, yerr=ses, color=colors[i], alpha=0.8)

    ax.set_xticks(x + width * len(pops_found) / 2)
    ax.set_xticklabels(region_names, fontsize=9)
    ax.set_ylabel("Mean TMRCA (generations)")
    ax.set_title("In(2L)a inversion coalescence structure", loc="left")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(args.output_dir, "figure8_inversion.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
