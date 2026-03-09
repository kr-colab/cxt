#!/usr/bin/env python3
import os, json, argparse, pathlib
from typing import List, Optional, Tuple
import numpy as np
import tskit
import multiprocessing as mp

from cxt.utils import xor, xnor
from cxt.sfs import calculate_window_sfs


def ts2X_vectorized_bichan(ts, window_size=2000, step_size=2000,
                           pivot_A=0, pivot_B=1, sequence_length=1e6, offset=0):
    site_positions = ts.tables.sites.position - offset
    gm = ts.genotype_matrix().T  # [samples, sites]
    step_size = window_size 
    # mask invariants / bad sites once
    bad = (gm >= 2).any(0) | (gm.sum(0) >= ts.num_samples)
    gm = gm[:, ~bad]
    site_positions = site_positions[~bad]

    num_samples = gm.shape[0]
    freq = gm.sum(0)                          # [sites]
    xor_mask = xor(gm[pivot_A], gm[pivot_B])  # [sites]  (0/1)

    w_xor  = freq * xor_mask
    w_xnor = freq * (1 - xor_mask)

    seq_len = sequence_length  # avoid hardcoding 1e6
    w_multipliers = np.array([2, 8, 32, 64])
    n_steps = int(np.ceil(seq_len / step_size))

    # out: [2 channels, n_multipliers, n_steps, num_samples]
    Xs = np.zeros((2, len(w_multipliers), n_steps, num_samples), dtype=np.int32)

    for i, m in enumerate(w_multipliers):
        ws = window_size * m
        Xs[0, i] = calculate_window_sfs(
            positions=site_positions,
            pivot_frequencies=w_xor,
            window_size=ws, step_size=step_size,
            sequence_length=seq_len, num_samples=num_samples,
        )
        Xs[1, i] = calculate_window_sfs(
            positions=site_positions,
            pivot_frequencies=w_xnor,
            window_size=ws, step_size=step_size,
            sequence_length=seq_len, num_samples=num_samples,
        )

    return Xs  # cast to float16 later if needed

def process_X(ts, pairs, window_size=2000, sequence_length=1e6, dtype=np.float16):
    P = len(pairs)
    # probe shape once
    a0, b0 = pairs[0]
    X0 = ts2X_vectorized_bichan(ts, window_size=window_size, pivot_A=a0, pivot_B=b0, sequence_length=sequence_length)
    out = np.empty((P,) + X0.shape, dtype=X0.dtype)
    out[0] = X0
    for k, (pa, pb) in enumerate(pairs[1:], start=1):
        out[k] = ts2X_vectorized_bichan(ts, window_size=window_size, pivot_A=pa, pivot_B=pb, sequence_length=sequence_length)
    return out.astype(dtype, copy=False)


import numpy as np
import tskit
from typing import List, Tuple, Optional

def process_y(
    ts,
    pairs,
    window_size=2000,
    sequence_length: Optional[int] = None,
    transform=None,
    dtype=np.float16,
    interp_fn=None,
    **interp_kwargs,
):
    """
    Compute window-averaged TMRCA for many pairs using YOUR interpolate function.
    y : (P, L) array
    """
    if interp_fn is None:
        interp_fn = interpolate_tmrcas

    P = len(pairs)
    if P == 0:
        return np.empty((0, 0), dtype=dtype)

    # Probe once to get L
    a0, b0 = pairs[0]
    y0 = np.asarray(interp_fn(ts, window_size=window_size, sequence_length=sequence_length, sample_a=a0, sample_b=b0, **interp_kwargs))
    if y0.ndim != 1:
        raise ValueError(f"interp_fn must return a 1D array; got shape {y0.shape}")
    L = y0.shape[0]

    out = np.empty((P, L), dtype=y0.dtype)
    out[0] = y0

    # Fill the rest
    for k, (pa, pb) in enumerate(pairs[1:], start=1):
        yk = np.asarray(interp_fn(ts, window_size=window_size, sequence_length=sequence_length, sample_a=pa, sample_b=pb, **interp_kwargs))
        if yk.shape[0] != L:
            raise ValueError(
                f"Inconsistent window count from interp_fn: pair 0 -> {L}, pair {k} -> {yk.shape[0]}.\n"
                "Hint: ensure (end-start) and window_size are identical across calls (e.g., pass the same "
                "sequence_length/start/end via **interp_kwargs or snap down to whole windows inside interp_fn)."
            )
        out[k] = yk

    if transform:
        out = np.log(out)
    return out.astype(dtype, copy=False)


def interpolate_tmrca_per_window_spanavg(
    lefts: np.ndarray,
    rights: np.ndarray,
    values: np.ndarray,
    *,
    interval_start: int = 0,
    interval_end: Optional[int] = None,
    interval_size: int = 2000,
) -> np.ndarray:
    """
    Exact length-weighted averages of a piecewise-constant signal (values over [lefts, rights))
    into fixed windows [interval_start + k*interval_size, ...).
    """
    assert lefts.ndim == rights.ndim == values.ndim == 1
    assert len(lefts) == len(rights) == len(values)
    assert np.all(rights[1:] >= rights[:-1]) and np.all(lefts[1:] >= lefts[:-1])

    if interval_end is None:
        interval_end = int(rights[-1])

    # Build window edges
    starts = np.arange(interval_start, interval_end, interval_size, dtype=np.int64)
    ends   = starts + interval_size
    nW = len(starts)

    numer = np.zeros(nW, dtype=np.float64)  # overlap * value
    # Two-pointer sweep over segments and windows: O(S + W)
    i = 0  # segment index
    j = 0  # window index
    S = len(lefts)

    while i < S and j < nW:
        a = max(lefts[i], starts[j])
        b = min(rights[i], ends[j])
        if b > a:
            numer[j] += (b - a) * values[i]
        # advance the pointer that ends first
        if rights[i] <= ends[j]:
            i += 1
        else:
            j += 1

    return numer / interval_size


def interpolate_tmrcas(
    ts: tskit.TreeSequence,
    window_size: int,
    sequence_length: Optional[int] = None,
    sample_a: int = 0,
    sample_b: int = 1,
    interval_start: int = 0,
) -> np.ndarray:
    """
    Exact windowed averages of TMRCA for a given pair of samples across a tree sequence.

    When the tree sequence covers a sub-region of a chromosome (e.g. simulated
    with left/right coordinates), set *interval_start* to the genomic start
    position so that windows align with the actual data.
    """
    if sequence_length is None:
        interval_end = int(ts.sequence_length)
    else:
        interval_end = interval_start + int(sequence_length)

    lefts, rights, tmrcas = [], [], []
    for tree in ts.trees():
        left, right = tree.interval
        m = tree.mrca(sample_a, sample_b)
        if m == tskit.NULL:
            tmrca = np.nan
        else:
            try:
                tmrca = tree.time(m)
            except (ValueError, IndexError):
                tmrca = np.nan
        lefts.append(left)
        rights.append(right)
        tmrcas.append(tmrca)

    lefts  = np.asarray(lefts, dtype=np.int64)
    rights = np.asarray(rights, dtype=np.int64)
    vals   = np.asarray(tmrcas, dtype=np.float64)

    return interpolate_tmrca_per_window_spanavg(
        lefts, rights, vals,
        interval_start=interval_start,
        interval_end=interval_end,
        interval_size=window_size,
    )


# ---- simple deterministic helpers ----
def find_ts_files(root: pathlib.Path) -> List[pathlib.Path]:
    exts = {".trees", ".ts", ".tsk", ".tskit"}
    out = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            p = pathlib.Path(dirpath) / fn
            if p.suffix.lower() in exts:
                out.append(p)
    out.sort(key=lambda p: str(p.relative_to(root)))
    return out

#def deterministic_split(files: List[pathlib.Path], train_ratio=0.9, seed=12345):
#    N = len(files)
#    idx = np.arange(N)
#    rng = np.random.default_rng(seed)
#    rng.shuffle(idx)                 # deterministic shuffle of FILES
#    cut = int(np.floor(train_ratio * N))
#    train_idx = set(idx[:cut].tolist())
#    return train_idx  # membership test is O(1)

def deterministic_split_grouped(files: List[pathlib.Path], base: pathlib.Path,
                                train_ratio=0.9, seed=12345) -> set[int]:
    rng = np.random.default_rng(seed)
    # group files by scenario
    by_scn = {}
    for i, f in enumerate(files):
        scn = scenario_from_path(f, base)
        by_scn.setdefault(scn, []).append(i)

    train_idx = set()
    for scn, idxs in by_scn.items():
        idxs = np.array(idxs, dtype=int)
        rng.shuffle(idxs)
        n = len(idxs)
        cut = int(np.floor(train_ratio * n))
        # ensure at least 1 test if there are ≥2 files in the scenario
        if n >= 2 and cut == n:
            cut = n - 1
        train_idx.update(idxs[:cut].tolist())
    return train_idx

def choose_pairs(ts: tskit.TreeSequence, num_pairs: int, seed: int) -> np.ndarray:
    n = ts.num_samples
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    if len(all_pairs) < num_pairs:
        raise ValueError(f"TS has only {len(all_pairs)} unique pairs; requested {num_pairs}.")
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(all_pairs), size=num_pairs, replace=False)
    return np.asarray([all_pairs[k] for k in sel], dtype=np.int32)

def scenario_from_path(p: pathlib.Path, base: pathlib.Path) -> str:
    # keep the folder structure under the base (excluding the filename)
    rel = p.relative_to(base).parent
    return str(rel) if str(rel) != "." else "."

def _percent_missing_per_window(missing_mask, window_size, step_size, sequence_length):
    """
    Fast rolling window sums using cumulative sum. Returns fraction in [0,1].
    missing_mask: length=sequence_length, 1=missing, 0=present
    Windows: [k*step_size, k*step_size + window_size)
    """
    n_steps = int(np.ceil(sequence_length / step_size))
    window_starts = np.arange(n_steps, dtype=int) * step_size
    window_ends = np.minimum(window_starts + window_size, sequence_length)

    # cumulative sum with a leading zero for easy range sums
    c = np.zeros(sequence_length + 1, dtype=np.int64)
    c[1:] = np.cumsum(missing_mask, dtype=np.int64)

    # sum of missing within each window = c[end] - c[start]
    sums = c[window_ends] - c[window_starts]
    lens = (window_ends - window_starts).astype(np.int64)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = sums / np.maximum(lens, 1)
    return frac
"""
def missingness_by_window_scales(
    missing_mask: np.ndarray,
    base_window: int,
    step_size: int,
    multipliers: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    n_steps = int(np.ceil(sequence_length / step_size))
    out = np.zeros((len(multipliers), n_steps), dtype=np.float32)
    for i, m in enumerate(multipliers):
        w_m = int(m) * int(base_window)
        out[i] = _percent_missing_per_window(
            missing_mask, window_size=w_m, step_size=step_size, sequence_length=sequence_length
        ).astype(np.float32)
    return out
"""
"""
def bitmask_to_intervals(bitmask):
    changepoints = np.flatnonzero(bitmask[1:] != bitmask[:-1])
    changepoints = np.append(np.append(0, changepoints + 1), bitmask.size)
    bedmask = []
    for s, e in zip(changepoints[:-1], changepoints[1:]):
        if bitmask[s]: bedmask.append([s, e]) 
    return np.stack(bedmask)
"""
    
# -fast vesions
def missingness_by_window_scales(
    missing_mask: np.ndarray,
    base_window: int,
    step_size: int,
    multipliers: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    """
    Vectorized over all multipliers: one cumsum, all window sums via fancy indexing.
    Returns shape (len(multipliers), n_steps).
    """
    # Ensure compact dtypes for faster cumsum/indexing
    mm = np.ascontiguousarray(missing_mask, dtype=np.uint8)
    N = int(sequence_length)

    n_steps = int(np.ceil(N / step_size))
    starts = (np.arange(n_steps, dtype=np.int64) * step_size)

    # Single cumulative sum for all queries
    c = np.empty(N + 1, dtype=np.int64)
    c[0] = 0
    c[1:] = np.cumsum(mm, dtype=np.int64)

    # All window sizes at once
    w = (np.asarray(multipliers, dtype=np.int64) * int(base_window))
    ends = np.minimum(starts[None, :] + w[:, None], N)

    # Broadcasted range sums and lengths -> fractions
    sums = c[ends] - c[starts][None, :]
    lens = (ends - starts[None, :]).astype(np.float32)
    out = (sums / np.maximum(lens, 1)).astype(np.float32)
    return out

def bitmask_to_intervals(bitmask: np.ndarray) -> np.ndarray:
    """
    Vectorized run-length extraction of True segments [start, end) from a boolean bitmask.
    Returns shape (K, 2). If no True runs, returns (0,2) array.
    """
    b = np.asarray(bitmask, dtype=bool)
    if b.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    # Indices where value changes, add boundaries
    changes = np.flatnonzero(b[1:] != b[:-1]) + 1
    bounds = np.r_[0, changes, b.size]

    # Select only runs that start with True
    start_flags = b[bounds[:-1]]
    if not np.any(start_flags):
        return np.empty((0, 2), dtype=np.int64)

    starts = bounds[:-1][start_flags]
    ends   = bounds[1:][start_flags]
    return np.column_stack((starts, ends)).astype(np.int64)



def process_X_with_bitmask(ts, pairs, window_size, sequence_length, dtype, unaccessible_bitmask_subset):
    ts_masked = ts.delete_intervals(bitmask_to_intervals(unaccessible_bitmask_subset))
    X = process_X(ts_masked, pairs=pairs, window_size=window_size, sequence_length=sequence_length, dtype=dtype)

    w_multipliers = np.array([2, 8, 32, 64])
    missing_mask = unaccessible_bitmask_subset.astype(int)
    missing_by_mult = missingness_by_window_scales(
        missing_mask=missing_mask,
        base_window=window_size,
        step_size=window_size,           
        multipliers=w_multipliers,
        sequence_length=sequence_length,
    )

    missing_by_mult = np.expand_dims(missing_by_mult, axis=0)
    missing_by_mult = np.tile(missing_by_mult, (len(pairs), 1, 1))
    X[:, 0, :, :, 0] = np.exp(missing_by_mult)
    X[:, 1, :, :, 0] = np.exp(1 - missing_by_mult)
    return X.astype(dtype)



# ---- worker ----
def _worker(args_tuple: Tuple[int, str, str, str, str, int, int, int, bool, int, str]) -> str:
    (
        idx, f_str, base_str, out_root_str, split, window_size, sequence_length,
        num_pairs, global_seed, skip_existing, simplify_first_n_samples, bitmask
    ) = args_tuple

    f = pathlib.Path(f_str)
    base = pathlib.Path(base_str)
    out_root = pathlib.Path(out_root_str)

    scenario = scenario_from_path(f, base)
    short_id = f"{f.stem}_i{idx}"
    out_dir = out_root / split / scenario / short_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # fast skip
    if skip_existing and (out_dir / "X.npy").exists() and (out_dir / "y.npy").exists():
        return f"[skip] {split}/{scenario}/{short_id}"

    # load ts
    try:
        ts = tskit.load(str(f))
        if simplify_first_n_samples > 0 and ts.num_samples > simplify_first_n_samples:
            samples = list(range(simplify_first_n_samples))
            ts = ts.simplify(samples=samples)
    except Exception as e:
        return f"[error] load failed {f}: {e}"

    # deterministic per-file seed (your original: depends on index)
    pair_seed = int(np.int64(global_seed) + np.int64(idx) * 10007)

    if bitmask is not None:
        rng = np.random.default_rng(pair_seed)  
        start_index = rng.integers(0, bitmask.size - sequence_length)
        unaccessible_bitmask_subset = bitmask[start_index:start_index + sequence_length]

    try:
        pairs = choose_pairs(ts, num_pairs, seed=pair_seed)
        if bitmask is not None:
            X = process_X_with_bitmask(ts, pairs, window_size=window_size, sequence_length=sequence_length,
                                      dtype=np.float16, unaccessible_bitmask_subset=unaccessible_bitmask_subset)
        else:
            X = process_X(ts, pairs, window_size=window_size, sequence_length=sequence_length, dtype=np.float16)
        y = process_y(ts, pairs, window_size=window_size, sequence_length=sequence_length, transform=np.log, dtype=np.float16)

        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)
        np.save(out_dir / "pairs.npy", pairs)

        meta = {
            "source_file": str(f.relative_to(base)),
            "id": short_id,
            "split": split,
            "scenario": scenario,
            "num_pairs": int(num_pairs),
            "window_size": int(window_size),
            "sequence_length": int(sequence_length),
            "num_samples": int(ts.num_samples),
            "dtype": "float16",
            "y_transform": "log",
            "global_seed": int(global_seed),
            "file_index": int(idx),
        }
        with open(out_dir / "meta.json", "w") as fp:
            json.dump(meta, fp, indent=2)

        return f"[ok] {split}/{scenario}/{short_id}  X{X.shape}  y{y.shape}"
    except Exception as e:
        return f"[error] processing {f}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Root where your simulated TSs live")
    ap.add_argument("--out_subdir", default="processed", help="Output directory name under base_dir")
    ap.add_argument("--window_size", type=int, default=2000)
    ap.add_argument("--sequence_length", type=int, default=1e6)  # not used currently
    ap.add_argument("--num_pairs", type=int, default=200)
    ap.add_argument("--simplify_first_n_samples", type=int, default=50,
                    help="If >0, simplify TSs to this many samples first")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--global_seed", type=int, default=12345)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--num_workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--bitmask", type=str, default=None)
    args = ap.parse_args()

    base = pathlib.Path(args.base_dir)
    out_root = base / args.out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    files = find_ts_files(base)
    if not files:
        print(f"No TS files found under {base}")
        return

    #train_membership = deterministic_split(files, args.train_ratio, seed=args.global_seed)
    train_membership = deterministic_split_grouped(files, base, args.train_ratio, seed=args.global_seed)

    unaccessible_bitmask = None
    if args.bitmask is not None:
        bitmask = np.load(args.bitmask)['access_2L']
        unaccessible_bitmask = ~bitmask

    # Build job list (freeze split/index to keep determinism with multiprocessing)
    jobs = []
    for idx, f in enumerate(files):
        split = "train" if idx in train_membership else "test"
        jobs.append((
            idx,
            str(f),
            str(base),
            str(out_root),
            split,
            int(args.window_size),
            int(args.sequence_length),
            int(args.num_pairs),
            int(args.global_seed),
            bool(args.skip_existing),
            int(args.simplify_first_n_samples),
            unaccessible_bitmask
        ))

    # Run in parallel
    with mp.Pool(processes=args.num_workers) as pool:
        for msg in pool.imap_unordered(_worker, jobs, chunksize=1):
            print(msg, flush=True)

if __name__ == "__main__":
    main()
