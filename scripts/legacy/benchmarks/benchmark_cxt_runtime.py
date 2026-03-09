#!/usr/bin/env python3
"""
CXT runtime benchmark for Fig S6.

Produces cxt_benchmark_runtime_blocks_regions_validation.jsonl with records:
  region, devices, num_devices, blocks, num_samples (haploids), num_pairs,
  sequence_length, runtime_seconds, ...

Matches revision/figure_benchmark/experiment.ipynb cell 9.
Uses pre-simulated tree sequences from mosquito benchmarks.

Usage:
  python -m scripts.benchmarks.benchmark_cxt_runtime [--results-path PATH] [--data-dir PATH]

Data dir should contain benchmark_samples50_region{0,1,2}.trees (50 diploids = 100 haploids).
"""

import argparse
import json
import os
import time

import numpy as np
import tskit

import cxt


# ============================================================
# Config
# ============================================================

BATCH_HAPLOIDS = 50
MAX_PAIRS_PER_CALL = BATCH_HAPLOIDS * (BATCH_HAPLOIDS - 1) // 2  # 1225
MAX_DIPLOIDS = 50
MAX_HAPLOIDS = 2 * MAX_DIPLOIDS
SEQUENCE_LENGTH = 1_000_000
RNG_SEED = 12345
CHUNK_SIZE = 25


def _blocks_to_key(blocks_list):
    return tuple((float(b[0]), float(b[1])) for b in blocks_list)


def load_completed_keys(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = float(rec["region"])
            devices = tuple(rec["devices"])
            blocks_key = _blocks_to_key(rec["blocks"])
            n = int(rec["num_samples"])
            done.add((r, devices, blocks_key, n))
    return done


def all_pairs(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def nC2(n: int) -> int:
    return n * (n - 1) // 2


PIVOT_PAIRS_50 = all_pairs(BATCH_HAPLOIDS)


def make_batches_cover_all_pairs(ts, pool_haploids, batch_haploids=50, chunk_size=25, seed=0):
    if pool_haploids < batch_haploids or pool_haploids > ts.num_samples:
        raise ValueError("pool_haploids out of range")
    rng = np.random.default_rng(seed)
    all_nodes = np.asarray(ts.samples(), dtype=np.int64)
    pool = all_nodes[:pool_haploids]
    batches = [pool[:batch_haploids].copy()]
    seen = pool[:batch_haploids].copy()
    idx = batch_haploids

    while idx < pool_haploids:
        new = pool[idx : min(idx + chunk_size, pool_haploids)]
        m = new.size
        k = batch_haploids - m
        if k <= 0:
            raise ValueError("chunk_size too large")
        n_mixed = int(np.ceil(seen.size / k))
        perm = rng.permutation(seen)
        for bi in range(n_mixed):
            start = bi * k
            end = min((bi + 1) * k, perm.size)
            old_block = perm[start:end]
            if old_block.size < k:
                remaining = perm[~np.isin(perm, old_block)]
                pad = rng.choice(remaining, size=k - old_block.size, replace=False)
                old_block = np.concatenate([old_block, pad], axis=0)
            batch = np.concatenate([new, old_block], axis=0)
            assert batch.size == batch_haploids and np.unique(batch).size == batch_haploids
            batches.append(batch)
        seen = np.concatenate([seen, new], axis=0)
        idx += m
    return batches


def build_batches_and_pivots(ts, pool_haploids, *, batch_haploids=50, chunk_size=25, seed=0):
    all_nodes = np.asarray(ts.samples(), dtype=np.int64)
    if pool_haploids > all_nodes.size:
        raise ValueError(f"pool_haploids={pool_haploids} > ts.num_samples={all_nodes.size}")
    pool = all_nodes[:pool_haploids]
    rng = np.random.default_rng(seed)

    if pool_haploids < batch_haploids:
        pivot_pairs = all_pairs(pool_haploids)
        need = batch_haploids - pool_haploids
        outside = all_nodes[pool_haploids:]
        if outside.size >= need:
            pad = rng.choice(outside, size=need, replace=False)
        else:
            pad = (
                rng.choice(outside, size=outside.size, replace=False)
                if outside.size > 0
                else np.array([], dtype=np.int64)
            )
            pad = np.concatenate(
                [pad, rng.choice(pool, size=need - len(pad), replace=False)]
            )
        batch = np.concatenate([pool, pad], axis=0)
        return [batch], pivot_pairs, len(pivot_pairs), len(pivot_pairs)

    batches = make_batches_cover_all_pairs(
        ts, pool_haploids=pool_haploids,
        batch_haploids=batch_haploids, chunk_size=chunk_size, seed=seed,
    )
    return batches, PIVOT_PAIRS_50, len(PIVOT_PAIRS_50), len(batches) * len(PIVOT_PAIRS_50)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-path",
        default="revision/figure_benchmark/cxt_benchmark_runtime_blocks_regions_validation.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--data-dir",
        default="/sietch_colab/data_share/cxt/mosquito/benchmarks",
        help="Directory with benchmark_samples50_region{N}.trees",
    )
    parser.add_argument(
        "--regions",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--num-haploids",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2"],
        help="GPU devices",
    )
    parser.add_argument(
        "--blocks",
        default="0,100000",
        help="Comma-separated start,end for single 100kb block (figS6 uses 1×1Mb label)",
    )
    args = parser.parse_args()

    blocks = [tuple(map(float, args.blocks.split(",")))]
    blocks_key = _blocks_to_key(blocks)
    completed = load_completed_keys(args.results_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.results_path)) or ".", exist_ok=True)

    model = cxt.load_model("broad", device="cpu")

    tss = []
    for region in args.regions:
        path = os.path.join(args.data_dir, f"benchmark_samples50_region{region}.trees")
        if not os.path.exists(path):
            print(f"[SKIP] Missing: {path}")
            continue
        ts = tskit.load(path)
        assert ts.num_samples == MAX_HAPLOIDS
        tss.append((region, ts))

    for blocks_tuple in [blocks]:
        blocks_key = _blocks_to_key(blocks_tuple)
        devices = args.devices

        for region, ts in tss:
            for pool_haploids in args.num_haploids:
                if pool_haploids > ts.num_samples:
                    continue
                key = (region, tuple(devices), blocks_key, int(pool_haploids))
                if key in completed:
                    print(f"[SKIP] r={region}, devices={devices}, pool={pool_haploids}")
                    continue

                batches, pivot_pairs, pairs_per_batch, processed_pairs = build_batches_and_pivots(
                    ts, pool_haploids=pool_haploids,
                    batch_haploids=BATCH_HAPLOIDS, chunk_size=CHUNK_SIZE,
                    seed=RNG_SEED + 1000 * pool_haploids,
                )
                eval_pairs = nC2(pool_haploids)

                print(f"[RUN] r={region}, devices={devices}, pool={pool_haploids}, pairs={eval_pairs}")

                total_start = time.perf_counter()
                batch_runtimes = []
                for bi, sample_nodes in enumerate(batches):
                    ts_simplified = ts.simplify(samples=sample_nodes)
                    assert ts_simplified.num_samples == BATCH_HAPLOIDS

                    start = time.perf_counter()
                    cxt.translate(
                        ts_simplified, model,
                        pivot_pairs=pivot_pairs,
                        blocks=blocks_tuple,
                        devices=devices,
                        B_per_device=512,
                        B=512,
                        build_workers=36,
                        mutation_rate=None,
                    )
                    elapsed = time.perf_counter() - start
                    batch_runtimes.append(elapsed)
                    print(f"   [BATCH {bi+1}/{len(batches)}] {elapsed:.2f}s")

                total_elapsed = time.perf_counter() - total_start

                record = {
                    "region": float(region),
                    "devices": list(devices),
                    "num_devices": len(devices),
                    "blocks": [[float(b[0]), float(b[1])] for b in blocks_tuple],
                    "num_samples": int(pool_haploids),
                    "num_pairs": int(eval_pairs),
                    "sequence_length": float(SEQUENCE_LENGTH),
                    "runtime_seconds": float(total_elapsed),
                    "num_batches": len(batches),
                    "batch_haploids": BATCH_HAPLOIDS,
                    "pairs_per_batch": pairs_per_batch,
                    "processed_pairs": processed_pairs,
                    "runtime_seconds_batches": [float(x) for x in batch_runtimes],
                    "chunk_size": CHUNK_SIZE,
                }
                with open(args.results_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"[DONE] r={region}, pool={pool_haploids} in {total_elapsed:.2f}s")
                completed.add(key)


if __name__ == "__main__":
    main()
