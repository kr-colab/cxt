### deterministic ###
import os, torch
import traceback

# cuBLAS determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # ":16:8"  # or 

# Torch deterministic mode
torch.use_deterministic_algorithms(True, warn_only=False)

# cuDNN / TF32 off for determinism
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Make SDPA deterministic (disable flash & mem-effic; force math)
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

# Global seeds (covers any stray torch/np/random usage)
import random, numpy as np
BASE = 1234
random.seed(BASE)
np.random.seed(BASE)
torch.manual_seed(BASE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE)

# ============================================================
# Multi-GPU pipeline + tqdm + MULTIPROCESS SOURCE BUILD
# (same behavior as your working version, but memory-fixed)
# ============================================================

import gc
import copy
import threading
from typing import Callable, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm, trange
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------------------------
# SFS + source building
# --------------------------

def calculate_window_sfs_vectorized(
        positions, pivot_frequencies,
        window_size=2000, sequence_length=1e6, num_samples=50, step_size=2000, availability_mask=None,):
    n_windows = int(np.ceil(sequence_length / step_size))
    window_starts = np.arange(n_windows) * step_size
    window_ends = np.minimum(window_starts + window_size, sequence_length)
    site_in_window = (positions[:, np.newaxis] >= window_starts) & \
                     (positions[:, np.newaxis] < window_ends)
    sfs_array = np.zeros((n_windows, num_samples), dtype=int)
    for i in range(n_windows):
        window_freqs = pivot_frequencies[site_in_window[:, i]]
        if len(window_freqs) > 0:
            sfs_array[i] = np.bincount(window_freqs, minlength=num_samples)
    return sfs_array


def calculate_window_sfs_vectorized(
    positions,
    pivot_frequencies,
    window_size=2000,
    sequence_length=1_000_000,
    num_samples=50,
    step_size=2000,
    availability_mask=None,
):
    # Number of windows is fixed to 500 to match your current setup
    n_windows = 500

    # Default: all available if no mask is provided
    if availability_mask is None:
        availability_mask = np.ones((n_windows,), dtype=float)
    else:
        availability_mask = np.asarray(availability_mask, dtype=float)
        assert availability_mask.shape == (500,), "availability_mask must be length 500"

    # Window coordinates
    window_starts = np.arange(n_windows, dtype=np.int64) * step_size
    window_ends   = np.minimum(window_starts + window_size, sequence_length).astype(np.int64)

    # Step (mask) coordinates — 500 bins by construction
    step_starts = np.arange(500, dtype=np.int64) * step_size
    step_ends   = np.minimum(step_starts + step_size, sequence_length).astype(np.int64)

    # Vectorized site->window membership
    # positions: (m,), window_starts/ends: (n_windows,)
    # site_in_window: (m, n_windows)
    site_in_window = (positions[:, None] >= window_starts[None, :]) & \
                     (positions[:, None] <  window_ends[None,   :])

    # Output (float: we’ll scale by availability)
    sfs_array = np.zeros((n_windows, num_samples), dtype=int)

    # Precompute per-window available bp via overlap with each step bin
    # For window i: overlap with all steps j is:
    #   overlap_ij = max(0, min(window_end[i], step_end[j]) - max(window_start[i], step_start[j]))
    # available_bp[i] = sum_j overlap_ij * availability_mask[j]
    # We'll compute this window-by-window to keep memory light (500x500 is fine anyway).
    for i in range(n_windows):
        ws, we = window_starts[i], window_ends[i]
        if we <= ws:  # empty window at the tail, if any
            continue

        # Overlap with all 500 steps (vectorized)
        left  = np.maximum(ws, step_starts)
        right = np.minimum(we, step_ends)
        overlaps = np.clip(right - left, 0, None).astype(float)  # (500,)

        available_bp = float(np.dot(overlaps, availability_mask))  # sum_j overlap_ij * avail[j]

        # Collect SFS for observed sites in this window
        if site_in_window[:, i].any():
            window_freqs = pivot_frequencies[site_in_window[:, i]]
            # Raw counts of sites per frequency bin (observed, possibly with missing)
            counts = np.bincount(window_freqs, minlength=num_samples).astype(float)
        else:
            counts = np.zeros((num_samples,), dtype=float)

        # Missing-data correction:
        # scale = window_size / available_bp (if available_bp > 0), else leave zeros
        if available_bp > 0:
            scale = window_size / available_bp
            sfs_array[i, :] = counts * scale
        else:
            # No available sequence: keep zeros (or you could set np.nan if preferred)
            sfs_array[i, :] = 0.0

    return sfs_array

import numpy as np

def calculate_window_sfs_vectorized(
    positions,
    pivot_frequencies,
    window_size=2000,
    sequence_length=1_000_000,
    num_samples=50,
    step_size=2000,
    availability_mask=None,
):
    # --- coerce inputs (and use integer comparison domain) ---
    positions = np.asarray(positions, dtype=np.int64)
    pivot_frequencies = np.asarray(pivot_frequencies)

    # Windows identical to the first implementation
    n_windows = int(np.ceil(sequence_length / step_size))
    window_starts = np.arange(n_windows, dtype=np.int64) * step_size
    window_ends   = np.minimum(window_starts + window_size, int(sequence_length)).astype(np.int64)

    # Site->window membership (same boolean logic as the first version)
    site_in_window = (positions[:, None] >= window_starts[None, :]) & \
                     (positions[:, None] <  window_ends[None,   :])

    # --- Fast path: no availability -> EXACT behavior match to the first function ---
    if availability_mask is None:
        sfs_array = np.zeros((n_windows, num_samples), dtype=int)
        for i in range(n_windows):
            window_freqs = pivot_frequencies[site_in_window[:, i]]
            if len(window_freqs) > 0:
                sfs_array[i] = np.bincount(window_freqs, minlength=num_samples)
        return sfs_array

    # --- Masked path (scaled counts; float output) ---
    availability_mask = np.asarray(availability_mask, dtype=float)

    # Steps are aligned to step_size; require matching length
    n_steps = int(np.ceil(sequence_length / step_size))
    if availability_mask.shape != (n_steps,):
        raise ValueError(f"availability_mask must be length {n_steps}, got {availability_mask.shape}")

    step_starts = np.arange(n_steps, dtype=np.int64) * step_size
    step_ends   = np.minimum(step_starts + step_size, int(sequence_length)).astype(np.int64)

    sfs_array = np.zeros((n_windows, num_samples), dtype=float)

    for i in range(n_windows):
        ws, we = window_starts[i], window_ends[i]
        if we <= ws:
            continue

        # Overlap of window i with all steps (integer arithmetic; cast for dot)
        left  = np.maximum(ws, step_starts)
        right = np.minimum(we, step_ends)
        overlaps = np.clip(right - left, 0, None).astype(float)  # (n_steps,)

        available_bp = float(np.dot(overlaps, availability_mask))

        if site_in_window[:, i].any():
            counts = np.bincount(pivot_frequencies[site_in_window[:, i]], minlength=num_samples).astype(float)
        else:
            counts = np.zeros((num_samples,), dtype=float)

        if available_bp > 0.0:
            scale = (we - ws) / available_bp  # window_size / available_bp
            sfs_array[i, :] = counts * scale
        else:
            sfs_array[i, :] = np.nan

    # --- Adjacent-window interpolation (linear, per SFS bin) ---
    if np.isnan(sfs_array).any():
        for k in range(num_samples):
            y = sfs_array[:, k]
            mask = np.isnan(y)
            if mask.any():
                if (~mask).any():
                    x_good = np.flatnonzero(~mask)
                    y_good = y[~mask]
                    y[mask] = np.interp(np.flatnonzero(mask), x_good, y_good)  # edge values are held constant
                else:
                    # all missing in this column -> fall back to zeros
                    y[:] = 0.0
            sfs_array[:, k] = y

    return sfs_array


#def check_blocks(blocks):
#    for b in blocks:
#        assert isinstance(b, tuple) and len(b) == 2 and b[0] < b[1], \
#            "Each block must be a tuple (start, end) with start < end"
#        assert b[1] - b[0] == 1e6, "Each block must be 1e6 bp"


def basic_filtering(block_gm, block_positions, num_samples=50):
    mask = np.logical_or(
        np.any(block_gm >= 2, axis=0),     # filters out non-biallelic
        block_gm.sum(0) >= num_samples)             # filters out fixed sites
    block_gm = block_gm[:, ~mask]
    block_positions = block_positions[~mask]
    return block_gm, block_positions


# --------- multiprocessing helpers for building sources ---------
def _build_one_src_task(task):
    """
    Worker function running in a separate process.
    Args:
      task = (b_idx, p_idx, block_pos, block_gm, pivot_A, pivot_B)
    Returns:
      (b_idx, p_idx, X)
    """
    b_idx, p_idx, block_pos, block_gm, pivot_A, pivot_B, sequence_length, step_size, availability_mask = task
    X = build_src(block_pos, block_gm, pivot_A, pivot_B, sequence_length, step_size, availability_mask)
    return b_idx, p_idx, X


def _build_sources_parallel(tasks, progress=True, max_workers: Optional[int] = None):
    """
    Run build_src over tasks via processes.
    Returns X_base stacked in original order [N, 2, 4, 500, 50].
    """
    N = len(tasks)
    results_buf = [None] * N
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, task in enumerate(tasks):
            fut = ex.submit(_build_one_src_task, task)
            fut._order = i  # attach order index
            futures.append(fut)

        it = as_completed(futures)
        if progress:
            it = tqdm(it, total=N, desc="Building sources (mp)", leave=False)
        for fut in it:
            i = getattr(fut, "_order", None)
            b_idx, p_idx, X = fut.result()
            results_buf[i] = X

    X_base = np.stack(results_buf, axis=0)
    return X_base


def _build_sources_serial(tasks, progress=True):
    it = tasks
    if progress:
        it = tqdm(it, total=len(tasks), desc="Building sources", leave=False)
    Xs = []
    for (b_idx, p_idx, block_pos, block_gm, pivot_A, pivot_B, sequence_length, step_size, availability_mask) in it:
        Xs.append(build_src(block_pos, block_gm, pivot_A, pivot_B, sequence_length, step_size, availability_mask))
    return np.stack(Xs, axis=0)

def build_src(block_positions, block_gm, pivot_id_A, pivot_id_B, sequence_length=1e6, step_size=2000, availability_mask=None):

    num_samples, num_sites = block_gm.shape

    # masks per site
    xor_mask  = (block_gm[pivot_id_A] ^ block_gm[pivot_id_B]).astype(bool)
    xnor_mask = ~xor_mask

    # site frequencies among samples
    freqs = block_gm.sum(0).astype(np.int32)

    # SUBSET positions AND freqs consistently
    pos_xor,  freqs_xor  = block_positions[xor_mask],  freqs[xor_mask]
    pos_xnor, freqs_xnor = block_positions[xnor_mask], freqs[xnor_mask]

    # helper that tolerates empty sets
    def sfs_for(pos, f, win_mult):
        if pos.size == 0:
            return np.zeros((int(np.ceil(1e6/2000)), num_samples), dtype=np.int32)
        return calculate_window_sfs_vectorized(
            positions=pos.astype(np.float32),
            pivot_frequencies=f.astype(np.int32),
            window_size=step_size * win_mult,
            sequence_length=sequence_length,
            num_samples=num_samples,
            step_size=step_size,
            availability_mask=availability_mask
        )

    w_multipliers = (2, 8, 32, 64)
    n_w = int(np.ceil(sequence_length/step_size))  # 500
    Xs_xor  = np.zeros((len(w_multipliers), n_w, num_samples), dtype=np.int32)
    Xs_xnor = np.zeros((len(w_multipliers), n_w, num_samples), dtype=np.int32)

    for i, w in enumerate(w_multipliers):
        Xs_xor[i]  = sfs_for(pos_xor,  freqs_xor,  w)
        Xs_xnor[i] = sfs_for(pos_xnor, freqs_xnor, w)

    X = np.stack([Xs_xor, Xs_xnor], axis=0).astype(np.float16)
    return np.log1p(X)


def basic_filtering(block_gm, block_positions):
    non_bial = np.any(block_gm > 1, axis=0)
    freq = block_gm.sum(0)
    fixed = (freq == 0) | (freq == block_gm.shape[0])
    mask = non_bial | fixed
    return block_gm[:, ~mask], block_positions[~mask]



# --------------------------
# Model cache resize (no buffers)
# --------------------------
@torch.no_grad()
def resize_model_cache_no_buffers(model, B, T=1001):
    """
    Resizes per-layer KV caches without register_buffer.
    Shapes: [B, H_kv, T, head_size]
    """
    ref = next(model.parameters())
    device, dtype = ref.device, ref.dtype
    attn0 = model.transformer.h[0].attn
    H_kv = getattr(attn0, "n_kv_head", getattr(attn0, "n_head"))
    D = attn0.head_size
    if T is None:
        T = getattr(getattr(model, "config", object()), "block_size",
                    getattr(attn0, "max_seq_len", 1001))
    for blk in model.transformer.h:
        attn = blk.attn
        for name in ("cache_k", "cache_v"):
            t = attn._buffers.pop(name, None)
            if hasattr(attn, name):
                obj = getattr(attn, name)
                setattr(attn, name, None)
                if isinstance(obj, torch.Tensor):
                    del obj
            if t is not None:
                del t
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shape = (B, H_kv, T, D)
    for blk in model.transformer.h:
        attn = blk.attn
        attn.cache_k = torch.zeros(shape, device=device, dtype=dtype, requires_grad=False)
        attn.cache_v = torch.zeros(shape, device=device, dtype=dtype, requires_grad=False)


# --------------------------
# Utils
# --------------------------
def _iter_chunks(N, B):
    for s in range(0, N, B):
        e = min(N, s + B)
        yield s, e

from cxt.utils import LOG_RESIDUAL_GRID, TIMES

def to_log_times(yhat, rep_mode=False, residual_model=False):
    if residual_model:
        time = LOG_RESIDUAL_GRID
    else:
        time = TIMES
    if rep_mode:
        return time[yhat[:, :, 1:].cpu().numpy() - 2].transpose(1, 0, 2)
    return time[yhat[:, 1:].cpu().numpy() - 2]


def generate_causal_mask(seq_len, full_attention_n=None, device="cpu"):
    full_attention_n = full_attention_n if full_attention_n is not None else 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask[:full_attention_n, :full_attention_n] = 1  # Full attention for first n tokens
    return mask.bool().unsqueeze(0).unsqueeze(0)


# --------------------------
# Sampling helpers
# --------------------------
def _make_generators(N, device, base_seed):
    gens = []
    for i in range(N):
        g = torch.Generator(device=device)
        g.manual_seed(int(base_seed) + i)
        gens.append(g)
    return gens


def _sample_per_row(probs, generators, row_ids):
    """
    Robust categorical sampling per row, deterministic via per-row generators.
    Guarantees indices in [0, V-1] even under FP rounding.
    """
    # Guard: zero-out tiny negatives, renormalize
    probs = torch.clamp(probs, min=0)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    # If any row sums to 0 (top_k removed all mass), fallback to uniform
    zero_mask = (probs_sum == 0)
    if zero_mask.any():
        V = probs.size(-1)
        probs = probs.masked_fill(zero_mask, 1.0 / V)
        probs_sum = probs_sum.masked_fill(zero_mask, 1.0)  # avoid div-by-0
    probs = probs / probs_sum

    # CDF with last element forced to exactly 1 to avoid searchsorted==V
    cdf = probs.cumsum(dim=-1)
    # Numerical safety: ensure last column is one
    cdf[:, -1] = 1.0

    B, V = probs.size(0), probs.size(1)
    # Draw u in [0, 1) deterministically per row
    u = torch.empty(B, 1, device=probs.device)
    for i in range(B):
        g = generators[int(row_ids[i])]
        # strictly less than 1.0 to avoid edges
        u[i, 0] = torch.rand((), device=probs.device, generator=g)
        if u[i, 0] == 1:  # ultra-rare, but clamp
            u[i, 0] = torch.nextafter(torch.tensor(1.0, device=probs.device), torch.tensor(0.0, device=probs.device))

    idx = torch.searchsorted(cdf, u, right=True)
    # clamp to [0, V-1]
    idx = torch.clamp(idx, 0, V - 1).long()
    return idx



# --------------------------
# Core single-device generate() with tqdm
# (attention mask cached & broadcast; no per-batch repeat)
# --------------------------
_ATTENTION_MASK_CACHE = {}  # key: (device_str, seq_len, full_n) -> [1,1,T,T]

def ensure_cache_B(model, B, T=1001):
    cur = getattr(model, "_cache_B", None)
    if cur == B:  # already the right size
        return
    resize_model_cache_no_buffers(model, B=B, T=T)
    model._cache_B = B

@torch.no_grad()
def generate(model, src, B=20, device="cuda", top_k=50, base_seed=1234,
             cache_matching=False, progress: bool = True, decode_bar: bool = False):
    curB = B
    if cache_matching:
        ensure_cache_B(model, curB)   
    model.eval()
    N = src.size(0)
    gens = _make_generators(N, device, base_seed)
    outs = []

    # build/reuse a single broadcastable mask [1,1,1001,1001]
    mask_key = (str(device), 1001, 501)
    attn_mask = _ATTENTION_MASK_CACHE.get(mask_key)
    if attn_mask is None:
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device=device)
        _ATTENTION_MASK_CACHE[mask_key] = attn_mask

    chunk_iter = range(0, N, B)
    if progress:
        chunk_iter = tqdm(chunk_iter, total=(N + B - 1)//B,
                          desc=f"Generate @ {device}", leave=False)

    with torch.inference_mode():
        for start in chunk_iter:
            end = min(start + B, N)
            batch_src = src[start:end].to(device, non_blocking=True)
            curB = batch_src.size(0)
            row_ids = torch.arange(start, end, device=device)

            # broadcasted mask (no .repeat)
            _ = model(batch_src, None, attn_mask, calculate_loss=False, use_cache=True, position=0)
            idx = torch.ones(curB, 1, dtype=torch.long, device=device)

            token_range = range(500, 1000)
            if decode_bar:
                token_range = trange(500, 1000, desc=f" decode {start}:{end}", leave=False)

            for i in token_range:
                logits = model(batch_src, idx[:, -1:], attn_mask, calculate_loss=False, use_cache=True, position=i)
                logits = logits[:, -1, :]
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = _sample_per_row(probs, gens, row_ids)
                idx = torch.cat([idx, next_token], dim=1)

            outs.append(idx)
            if hasattr(model, "clear_cache"):
                model.clear_cache()

    return torch.cat(outs, dim=0)

@torch.no_grad()
def generate(
    model,
    src,
    B: int = 20,
    device: str = "cuda",
    top_k: int | None = 50,
    base_seed: int = 1234,
    cache_matching: bool = False,
    progress: bool = True,
    decode_bar: bool = False,
    adapter=None,
):
    """
    Deterministic, cache-safe generate():
      - Resizes KV cache per *actual* microbatch (curB) when cache_matching=True
      - Uses a single broadcastable attention mask (no .repeat())
      - Clamps top_k to vocab size and uses robust sampler (_sample_per_row)
      - Validates vocab size stays constant during decode
    """
    model.eval()

    N = src.size(0)
    gens = _make_generators(N, device, base_seed)
    outs = []

    # Infer (or fix) sequence lengths used by your model
    T_total = 1001   # 500 src + 1 BOS + up to 500 decode
    full_n  = 501    # allow full attention for first 501 tokens

    # Build/reuse a single broadcastable mask [1, 1, T_total, T_total]
    mask_key = (str(device), T_total, full_n)
    attn_mask = _ATTENTION_MASK_CACHE.get(mask_key)
    if attn_mask is None:
        attn_mask = generate_causal_mask(T_total, full_attention_n=full_n, device=device)
        _ATTENTION_MASK_CACHE[mask_key] = attn_mask

    # Chunk over the batch
    chunk_iter = range(0, N, B)
    if progress:
        chunk_iter = tqdm(chunk_iter, total=(N + B - 1)//B, desc=f"Generate @ {device}", leave=False)

    with torch.inference_mode():

        for start in chunk_iter:
            end = min(start + B, N)
            batch_src = src[start:end].to(device, non_blocking=True)
            curB = batch_src.size(0)
            row_ids = torch.arange(start, end, device=device)

            # Ensure KV cache matches *actual* microbatch size
            if cache_matching:
                ensure_cache_B(model, curB)  # idempotent; cheap if size unchanged

            # Prime cache before decode
            if adapter is not None:
                batch_src = adapter(batch_src)

            _ = model(batch_src, None, attn_mask, calculate_loss=False, use_cache=True, position=0)
            idx = torch.ones(curB, 1, dtype=torch.long, device=device)

            # Decode loop
            token_range = trange(500, 1000, desc=f" decode {start}:{end}", leave=False) if decode_bar else range(500, 1000)
            V0 = None  # track vocab size consistency

            for i in token_range:
                logits = model(batch_src, idx[:, -1:], attn_mask, calculate_loss=False, use_cache=True, position=i)
                logits = logits[:, -1, :]  # [B, V]
                V = logits.size(-1)

                # Vocab size should not change mid-decode
                if V0 is None:
                    V0 = V
                elif V != V0:
                    raise RuntimeError(f"Vocab size changed during decode: was {V0}, now {V}")

                # Safe top-k: clamp to V and avoid zeroing all mass
                if top_k is not None:
                    tk = int(min(top_k, V)) or V
                    v, _ = torch.topk(logits, tk)
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_token = _sample_per_row(probs, gens, row_ids)  # robust per-row sampler
                idx = torch.cat([idx, next_token], dim=1)

            outs.append(idx)

            # Clear per-chunk caches if available
            if hasattr(model, "clear_cache"):
                model.clear_cache()

    return torch.cat(outs, dim=0)


# ============================================================
# Multi-GPU helpers (threaded sharding; cache-safe)
# (CPU-based replication to avoid cuda:0 spikes)
# ============================================================
def list_cuda_devices(n_gpus: int | None = None) -> List[str]:
    if not torch.cuda.is_available():
        return []
    n = torch.cuda.device_count()
    if n_gpus is not None:
        n = min(n, int(n_gpus))
    return [f"cuda:{i}" for i in range(n)]


def _replicate_model_to_device_cpu_base(src_model: torch.nn.Module, device: str):
    """
    Clone on CPU, then move to `device` to avoid allocator peaks on cuda:0.
    """
    mdl_cpu = copy.deepcopy(src_model).to("cpu", non_blocking=True)
    mdl = mdl_cpu.to(device, non_blocking=True)
    del mdl_cpu
    if hasattr(mdl, "cache_to_device"):
        mdl.cache_to_device(device)
    mdl.eval()
    return mdl


def _shard_indices(N: int, k: int) -> List[Tuple[int, int]]:
    base = N // k
    rem = N % k
    offsets = [0]
    for i in range(k):
        step = base + (1 if i < rem else 0)
        offsets.append(offsets[-1] + step)
    return [(offsets[i], offsets[i + 1]) for i in range(k) if offsets[i] < offsets[i + 1]]


def _thread_worker_generate(
    model: torch.nn.Module,
    src_chunk: torch.Tensor,
    B_local: int,
    device: str,
    base_seed: int,
    top_k: int,
    cache_matching: bool,
    out_list: list,
    out_index: int,
    progress: bool,
    decode_bar: bool,
    adapter: torch.nn.Module | None = None
):
    # bind CUDA context to this thread/device
    try:
        # Bind CUDA to this thread
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.set_device(int(device.split(":")[1]))

        # Per-thread RNG seeding for determinism (covers stray usage)
        import random, numpy as np
        random.seed(base_seed + out_index)
        np.random.seed(base_seed + out_index)
        torch.manual_seed(base_seed + out_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + out_index)

        with torch.inference_mode():
            y = generate(
                model=model,
                src=src_chunk.to(device, non_blocking=True),
                B=min(B_local, src_chunk.size(0)),
                device=device,
                top_k=top_k,
                base_seed=base_seed,
                cache_matching=cache_matching,
                progress=progress,
                decode_bar=decode_bar,
                adapter=adapter,
            )
        out_list[out_index] = y.to("cpu", non_blocking=True)

    except Exception as e:
        # Store the exception object in the slot, with traceback attached
        e._traceback = traceback.format_exc()
        out_list[out_index] = e


def multi_gpu_generate(
    model: torch.nn.Module,
    src: torch.Tensor,
    devices: Sequence[str] | None = None,
    B_per_device: int = 24,
    top_k: int = 50,
    base_seed: int = 1234,
    cache_matching: bool = False,
    progress: bool = True,
    decode_bar: bool = False,
    adapter: torch.nn.Module | None = None,
):
    """
    Run `generate` concurrently across multiple GPUs.
    Returns a single tensor concatenated in the original order.
    """
    if devices is None:
        devices = list_cuda_devices()
    if not devices:
        return generate(
            model, src, B=B_per_device,
            device="cuda" if torch.cuda.is_available() else "cpu",
            top_k=top_k, base_seed=base_seed, cache_matching=cache_matching,
            progress=progress, decode_bar=decode_bar, adapter=adapter
        )

    N = src.size(0)
    shards = _shard_indices(N, len(devices))
    models = [_replicate_model_to_device_cpu_base(model, d) for d in devices]

    if cache_matching:
        for mdl in models:
            try:
                resize_model_cache_no_buffers(mdl, B=B_per_device)
            except Exception:
                pass

    results = [None] * len(shards)
    threads = []
    for i, ((s, e), device, mdl) in enumerate(zip(shards, devices, models)):
        src_chunk = src[s:e]
        t = threading.Thread(
            target=_thread_worker_generate,
            kwargs=dict(
                model=mdl,
                src_chunk=src_chunk,
                B_local=B_per_device,
                device=device,
                base_seed=base_seed + s,  # preserve row-level determinism
                top_k=top_k,
                cache_matching=cache_matching,
                out_list=results,
                out_index=i,
                progress=progress,
                decode_bar=decode_bar,
                adapter=adapter,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    if progress:
        with tqdm(total=len(threads), desc="Devices", leave=False) as pbar:
            alive = [True]*len(threads)
            import time
            while any(alive):
                updated = False
                for j, t in enumerate(threads):
                    if alive[j] and not t.is_alive():
                        alive[j] = False
                        pbar.update(1)
                        updated = True
                if not updated:
                    time.sleep(0.05)

    for t in threads:
        t.join()

    # NEW: verify all shards produced tensors, else aggregate errors
    bad = [(i, r) for i, r in enumerate(results) if not isinstance(r, torch.Tensor)]
    if bad:
        msgs = []
        for i, r in bad:
            dev = devices[i] if i < len(devices) else f"shard#{i}"
            if hasattr(r, "_traceback"):
                msgs.append(f"[{dev}] {type(r).__name__}: {r}\n{r._traceback}")
            else:
                msgs.append(f"[{dev}] Non-tensor result: {repr(r)}")
        raise RuntimeError("multi_gpu_generate failed on some shards:\n\n" + "\n".join(msgs))

    Y = torch.cat(results, dim=0)

    for mdl in models:
        try: mdl.clear_cache()
        except Exception: pass
        del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return Y



# --------------------------
# VCF / TS wrappers (unchanged API + devices + progress)
# --------------------------
def vcf_parser(path):
    vcf = pd.read_csv(path, comment='#', sep='\t', header=None)
    pos_col = 1
    positions = vcf.iloc[:, pos_col]
    assert pd.api.types.is_numeric_dtype(positions), "POS column must be numeric"
    assert positions.is_monotonic_increasing, "POS column must be sorted"
    positions = positions.to_numpy(dtype=np.float32)
    # find first genotype-like column
    for col in vcf.columns:
        val = vcf.iloc[0, col]
        if isinstance(val, str) and ('|' in val or '/' in val):
            sample_start_col = col
            break
    else:
        raise AssertionError("No genotype columns found! Expected '0|1' or '1/1'")
    vcf = vcf.loc[:, sample_start_col:]
    haplo = [vcf[c].str.split(r"[|/]", expand=True).astype(int) for c in vcf.columns]
    genotypes = pd.concat(haplo, axis=1).to_numpy(dtype=np.int32)
    return positions, genotypes.T


# --------------------------
# Small helpers
# --------------------------
def get_tmrca_for(index_map, tmrca, block, pivot):
    """Return the TMRCA row for a given (block, pivot)."""
    idx = np.flatnonzero((index_map == (block, pivot)).all(axis=1))
    if len(idx) == 0:
        raise ValueError(f"(block={block}, pivot={pivot}) not found in index_map")
    return tmrca[idx[0]]

# --------------------------
# Model loader (as you had)
# --------------------------
from cxt.train import LitTokenFreeDecoder
def load_model(config, model_path, device="cuda"):
    lit = LitTokenFreeDecoder(config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    lit.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt
    model = lit.model
    del lit
    model.to(device)
    if hasattr(model, "cache_to_device"):
        model.cache_to_device(device)
    model.eval()
    return model

# --------------------------
# Ground-truth utility
# --------------------------
from cxt.utils import interpolate_tmrcas
def ground_truth_tmrca(ts, block, pivot_A, pivot_B, window_size=2000):
    start, end = map(int, block)
    ts12 = ts.simplify(samples=[pivot_A, pivot_B])
    tmrca = interpolate_tmrcas(ts12, window_size=window_size, sequence_length=ts.sequence_length)
    s, e = start // window_size, end // window_size
    vals = tmrca[s:e]
    return np.log(np.clip(vals, 1e-12, None)) if len(vals) else np.full(((end - start) // window_size,), np.log(1e4))


def _prepare_build_tasks(gm_samples, positions, blocks, pivot_pairs, sequence_length, step_size, availability_mask):
    tasks = []
    index_map = []
    #for b_idx, (block_start, block_end) in enumerate(blocks):
    #    block_mask = (positions >= block_start) & (positions < block_end)
    #    block_pos_abs = positions[block_mask]
    #    block_gm = gm_samples[:, block_mask]
    for b_idx, (block_start, block_end) in enumerate(blocks):
        block_mask = (positions >= block_start) & (positions < block_end)
        block_pos_abs = positions[block_mask]
        block_gm = gm_samples[:, block_mask]
        if availability_mask is not None:
            block_availability = availability_mask[int(block_start):int(block_end)]
            n = len(block_availability) // step_size
            block_availability_mask = block_availability[:n * step_size].reshape(n, step_size).mean(axis=1)
        else:
            block_availability_mask = None

        # block-relative coordinates
        block_pos_rel = block_pos_abs - block_start

        # optional: stricter, symmetric filtering
        block_gm, block_pos_rel = basic_filtering(block_gm, block_pos_rel)

        for p_idx, (pivot_A, pivot_B) in enumerate(pivot_pairs):
            tasks.append((b_idx, p_idx, block_pos_rel, block_gm, pivot_A, pivot_B, sequence_length, step_size, block_availability_mask))
            index_map.append((b_idx, p_idx))
    return tasks, np.array(index_map, dtype=np.int32)


# ============================================================
# Fast process-per-GPU helpers + wrappers (explicit spawn ctx)
# ============================================================

def _get_spawn_ctx():
    """
    Always return a multiprocessing context configured for 'spawn'.
    Using the context's Process/Queue avoids accidental 'fork' in notebooks.
    """
    try:
        import multiprocessing as mp
    except Exception:
        import torch.multiprocessing as mp
    try:
        return mp.get_context("spawn")
    except Exception:
        # Older Python may not support get_context on torch.multiprocessing
        # Fall back to stdlib mp with forced spawn start method.
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
        return mp

# ─────────────────────────────────────────────────────────────
# Worker: always quiet (no tqdm, no decode bars)
# ─────────────────────────────────────────────────────────────
def _proc_worker_generate_fast_mainloop(
    rank: int,
    device: str,
    model_cpu_or_none,
    model_factory_or_none,
    X_base_np,
    ids_np,
    id_chunks,
    B_local: int,
    base_seed: int,
    cache_matching: bool,
    decode_bar: bool,     # ignored in worker (forced False)
    out_queue,
    progress: bool,       # ignored in worker (forced False)
    adapter: torch.nn.Module | None = None
):
    import traceback as _tb
    try:
        import random, copy, torch, numpy as _np
        if torch.cuda.is_available() and str(device).startswith("cuda:"):
            torch.cuda.set_device(int(str(device).split(":")[1]))

        # Determinism per worker
        random.seed(base_seed + rank)
        _np.random.seed(base_seed + rank)
        torch.manual_seed(base_seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed + rank)

        # Build/clone model inside child
        if callable(model_factory_or_none):
            try:
                if adapter is not None:
                    mdl = model_factory_or_none("broad+adapter")
                else:
                    mdl = model_factory_or_none("broad")
            except TypeError:
                mdl = model_factory_or_none()
        else:
            mdl = copy.deepcopy(model_cpu_or_none)

        mdl = mdl.to(device, non_blocking=True)
        if hasattr(mdl, "cache_to_device"):
            mdl.cache_to_device(device)
        mdl.eval()
        p_dtype = next(mdl.parameters()).dtype

        # Iterate assigned chunks (no tqdm in worker)
        for (s, e) in id_chunks:
            sel = ids_np[s:e]
            X_chunk_np = X_base_np[sel]
            X_cpu = torch.as_tensor(X_chunk_np, dtype=p_dtype, device="cpu")
            if torch.cuda.is_available():
                X_cpu = X_cpu.pin_memory()
            with torch.inference_mode():
                Y_part = generate(
                    model=mdl,
                    src=X_cpu,
                    B=min(B_local, X_cpu.size(0)),
                    device=device,
                    top_k=50,
                    base_seed=base_seed + s,
                    cache_matching=cache_matching,
                    progress=decode_bar,         # ← force quiet. Update: not anymore
                    decode_bar=False,       # ← force quiet
                    adapter=adapter
                )
                out_queue.put((s, Y_part.cpu().numpy()))

        try:
            if hasattr(mdl, "clear_cache"):
                mdl.clear_cache()
        except Exception:
            pass

    except Exception as e:
        e._traceback = _tb.format_exc()
        out_queue.put(("__error__", rank, str(e), e._traceback))

# ─────────────────────────────────────────────────────────────
# Parent: explicit spawn ctx + single notebook progress bar
# ─────────────────────────────────────────────────────────────
def _fast_process_per_gpu_generate(
    X_base,
    devices,
    B_per_device: int,
    n_reps: int,
    base_seed: int,
    cache_matching: bool,
    decode_bar: bool,          # parent-wide decode flag (workers are quiet)
    model_cpu,
    model_factory_or_none,
    progress: bool = True,     # parent progress bar
    adapter: torch.nn.Module | None = None
):
    """
    Rep-major, process-per-GPU fast path (explicit spawn context).
    Workers are silent; parent shows a single Jupyter tqdm over chunks.
    """
    import numpy as _np, torch
    # explicit spawn context avoids accidental fork in notebooks
    try:
        import multiprocessing as mp
    except Exception:
        import torch.multiprocessing as mp
    try:
        ctx = mp.get_context("spawn")
    except Exception:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
        ctx = mp

    N = X_base.shape[0]
    world = len(devices)

    # Rep-major ids: [0..N-1] tiled n_reps times
    ids = _np.tile(_np.arange(N, dtype=_np.int64), n_reps)

    # Chunk stream; round-robin to devices
    chunk_size = max(1, B_per_device * max(1, world))
    ranges = [(s, min(s + chunk_size, ids.size)) for s in range(0, ids.size, chunk_size)]
    shards = [[] for _ in range(world)]
    for i, r in enumerate(ranges):
        shards[i % world].append(r)

    # Use context’s SimpleQueue/Process (no Manager, no fork)
    out_q = ctx.SimpleQueue()
    procs = []
    for rank, dev in enumerate(devices):
        p = ctx.Process(
            target=_proc_worker_generate_fast_mainloop,
            args=(
                rank, dev,
                model_cpu, model_factory_or_none,
                X_base, ids, shards[rank],
                B_per_device, base_seed, cache_matching, decode_bar,
                out_q, False,  # workers are always quiet
                adapter
            ),
        )
        p.start()
        procs.append(p)

    # Parent progress: one bar over number of chunks
    expected = sum(len(x) for x in shards)
    pieces, n_done = [], 0
    if progress:
        try:
            from tqdm.notebook import tqdm as _tqdm  # pretty blue bars
        except Exception:
            from tqdm import tqdm as _tqdm           # fallback ASCII
        pbar = _tqdm(total=expected, desc="Fast multi-GPU chunks", leave=True)
    else:
        pbar = None

    while n_done < expected:
        msg = out_q.get()
        if isinstance(msg, tuple) and len(msg) >= 2 and msg[0] == "__error__":
            _, rank, err, tb = msg
            for p in procs: p.join(timeout=0.1)
            if pbar: pbar.close()
            raise RuntimeError(f"Worker {rank} failed: {err}\n{tb}")
        s, y_np = msg
        pieces.append((s, y_np))
        n_done += 1
        if pbar: pbar.update(1)

    for p in procs:
        p.join()

    if pbar:
        pbar.close()

    pieces.sort(key=lambda t: t[0])
    Y_cpu = torch.from_numpy(_np.concatenate([p[1] for p in pieces], axis=0))
    return Y_cpu


from cxt.utils import stochastic_diversity_bias_correction
def apply_tmrca_bias_correction(tmrca, ts, index_map, blocks, pivot_pairs, mutation_rate, availability_mask=None):

    corrected_tmrca_all = np.zeros_like(tmrca)
    for b_idx, (block_start, block_end) in enumerate(blocks):

        index_map_block = np.where(index_map[:,0] == b_idx)[0]
        predictions = tmrca[:, index_map_block, :]

        corrected_tmrca_all[:, index_map_block, :] = stochastic_diversity_bias_correction(
            tree_sequence=ts.keep_intervals([[block_start, block_end]]),
            mutation_rate=mutation_rate,
            predictions=predictions, # log and in form (n_replicates, pairs, n_windows)
            availability_mask=availability_mask,
            pivot_pairs=np.array(pivot_pairs),
            rng=np.random.default_rng(1234),
        )
    return corrected_tmrca_all


from cxt.utils import stochastic_diversity_bias_correction_v2
def apply_tmrca_bias_correction_v2(tmrca, gm, positions, index_map, blocks, pivot_pairs, mutation_rate, availability_mask=None):

    corrected_tmrca_all = np.zeros_like(tmrca)
    for b_idx, (block_start, block_end) in enumerate(blocks):
        block_mask = (positions >= block_start) & (positions < block_end)
        block_pos_abs = positions[block_mask]

        step_size = int((block_end - block_start) // 500)
        if availability_mask is not None:
            block_availability = availability_mask[int(block_start):int(block_end)]
            n = len(block_availability) // step_size
            block_availability_mask = block_availability[:n * step_size].reshape(n, step_size).mean(axis=1)
        else:
            block_availability_mask = None

        
        block_gm = gm[:, block_mask]
        block_pos_rel = block_pos_abs - block_start

        index_map_block = np.where(index_map[:,0] == b_idx)[0]
        predictions = tmrca[:, index_map_block, :]

        corrected_tmrca_all[:, index_map_block, :] = stochastic_diversity_bias_correction_v2(
            genotype_matrix=block_gm, 
            mutation_rate=mutation_rate,
            predictions=predictions, # log and in form (n_replicates, pairs, n_windows)
            pivot_pairs=np.array(pivot_pairs),
            availability_mask=block_availability_mask,
            positions=block_pos_rel,
            window_size=step_size,
            rng=np.random.default_rng(1234),
        )
    return corrected_tmrca_all

def translate_from_genotype_matrix(
        gm,
        positions,
        model,  # CPU model object
        blocks=[(0e6, 1e6)],
        pivot_pairs=[(0, 1)],
        sample_ids=np.arange(0, 50), # not used anymore
        device="cuda", B=24,
        cache_matching=False,
        n_reps: int = 15,
        base_seed: int = 1234,
        top_k: int = 50,
        devices=None,
        B_per_device: int | None = None,
        progress: bool = True,
        decode_bar: bool = False,
        build_workers: int = 0,
        use_fast_process_per_gpu: bool = False,  # NEW
        adapter: torch.nn.Module | None = None,
        mutation_rate:float = None,
        residual_model: bool = False,
        availability_mask: np.ndarray | None = None
    ):
    """
    Returns yhat of shape (n_items, n_reps, ...).
    NOTE: requires existing helpers in your module:
      _prepare_build_tasks, _build_sources_parallel/_serial, _iter_chunks,
      multi_gpu_generate, generate, to_log_times, vcf_parser.
    """
    import numpy as _np, torch
    sample_ids = np.arange(gm.shape[0])
    gm_samples = gm[sample_ids, :]
    a, b = blocks[0]
    sequence_length = int(b - a)
    step_size = sequence_length // 500

    # Build sources (mp or serial)
    tasks, index_map = _prepare_build_tasks(
        gm_samples, positions, blocks, pivot_pairs, sequence_length, step_size, availability_mask
    )
    if build_workers and build_workers > 1:
        X_base = _build_sources_parallel(tasks, progress=progress, max_workers=build_workers)
    else:
        X_base = _build_sources_serial(tasks, progress=progress)

    N = X_base.shape[0]
    B_local = B if B_per_device is None else B_per_device

    # Keep source on CPU (pinned for H2D overlap inside generate)
    param_dtype = next(model.parameters()).dtype
    X_cpu_t = torch.as_tensor(X_base, dtype=param_dtype, device="cpu")
    if torch.cuda.is_available():
        X_cpu_t = X_cpu_t.pin_memory()

    # --- single-rep path ---
    if n_reps <= 1:
        if devices and len(devices) > 1:
            yhat = multi_gpu_generate(
                model=model, src=X_cpu_t, devices=devices, B_per_device=B_local,
                top_k=top_k, base_seed=base_seed, cache_matching=cache_matching,
                progress=progress, decode_bar=decode_bar, adapter=adapter
            )
        else:
            y_list = []
            loop = _iter_chunks(N, B)
            if progress:
                from tqdm.auto import tqdm as _tqdm
                loop = _tqdm(loop, total=(N + B - 1)//B, desc="Batches", leave=False)
            for s, e in loop:
                y_chunk = generate(
                    model.to(device), X_cpu_t[s:e], B=min(B, e - s), device=device,
                    top_k=top_k, base_seed=base_seed + s,
                    cache_matching=cache_matching, progress=False, decode_bar=decode_bar, adapter=adapter
                )
                y_list.append(y_chunk)
            yhat = torch.cat(y_list, dim=0)

        Y = to_log_times(yhat, rep_mode=False, residual_model=residual_model)
        if mutation_rate is not None:
            Y = apply_tmrca_bias_correction_v2(
                tmrca=Y.unsequeeze(0),
                gm=gm_samples,
                positions=positions,
                index_map=index_map,
                blocks=blocks,
                pivot_pairs=pivot_pairs,
                mutation_rate=mutation_rate,
                availability_mask=availability_mask)
        return Y, index_map

    # --- multi-rep paths ---
    world = len(devices) if (devices and len(devices) > 1) else 1

    # Prefer model factory if available (so child builds model)
    model_factory = globals().get("setup_cxt_model", None) if use_fast_process_per_gpu else None

    if devices and len(devices) > 1 and use_fast_process_per_gpu:
        X_np = X_base if isinstance(X_base, _np.ndarray) else X_cpu_t.cpu().numpy()
        Y_flat = _fast_process_per_gpu_generate(
            X_base=X_np,
            devices=devices,
            B_per_device=B_local,
            n_reps=n_reps,
            base_seed=base_seed,
            cache_matching=cache_matching,
            decode_bar=decode_bar,
            model_cpu=model,                 # fallback only
            model_factory_or_none=model_factory,
            progress=progress,
            adapter=adapter
        )
        Y = Y_flat.reshape(n_reps, N, *Y_flat.shape[1:]).transpose(0, 1).contiguous()
        Y = to_log_times(Y, rep_mode=True, residual_model=residual_model)
        if mutation_rate is not None:
            Y = apply_tmrca_bias_correction_v2(
                tmrca=Y,
                gm=gm_samples,
                positions=positions,
                index_map=index_map,
                blocks=blocks,
                pivot_pairs=pivot_pairs,
                availability_mask=availability_mask,
                mutation_rate=mutation_rate)
        return Y, index_map

    # Default safe path (threaded multi-GPU or single GPU)
    ids = torch.tile(torch.arange(N, dtype=torch.long), (n_reps,))
    Y_parts = []
    chunk_size = (B_local if B_per_device is not None else B) * world
    loop = _iter_chunks(ids.numel(), chunk_size)
    if progress:
        from tqdm.auto import tqdm as _tqdm
        loop = _tqdm(loop, total=(ids.numel() + chunk_size - 1)//chunk_size,
                     desc="Batches (reps fused)", leave=False)

    for s, e in loop:
        sel = ids[s:e]
        X_chunk = X_cpu_t.index_select(0, sel.to(dtype=torch.long))
        if devices and len(devices) > 1:
            Y_part = multi_gpu_generate(
                model=model, src=X_chunk, devices=devices, B_per_device=B_local,
                top_k=top_k, base_seed=base_seed + s, cache_matching=cache_matching,
                progress=True, decode_bar=decode_bar, adapter=adapter ## set progress True here
            )
        else:
            Y_part = generate(
                model.to(device), X_chunk, B=min(B, X_chunk.size(0)), device=device,
                top_k=top_k, base_seed=base_seed + s,
                cache_matching=cache_matching, progress=False, decode_bar=decode_bar, adapter=adapter
            )
        Y_parts.append(Y_part)

    Y = torch.cat(Y_parts, dim=0)
    Y = Y.reshape(n_reps, N, *Y.shape[1:]).transpose(0, 1)
    Y = to_log_times(Y.contiguous(), rep_mode=True, residual_model=residual_model)

    if mutation_rate is not None:
        Y = apply_tmrca_bias_correction_v2(
            tmrca=Y,
            gm=gm_samples,
            positions=positions,
            index_map=index_map,
            blocks=blocks,
            pivot_pairs=pivot_pairs,
            availability_mask=availability_mask,
            mutation_rate=mutation_rate)
    return Y, index_map

def translate_from_vcf(
    vcf_path,
    model,
    blocks=[(0e6, 1e6)],
    pivot_pairs=[(0, 1)],
    sample_ids=np.arange(0, 50),
    device="cuda",
    B=24,
    cache_matching=False,
    n_reps=15,
    base_seed=1234,
    top_k=50,
    devices=None,
    B_per_device: int | None = None,
    progress: bool = True,
    decode_bar: bool = False,
    build_workers: int = 0,
    use_fast_process_per_gpu: bool = False,  # NEW
    adapter: torch.nn.Module | None = None,
    mutation_rate: float | None = None,
    residual_model: bool = False,
    availability_mask: np.ndarray | None = None
):
    positions, gm = vcf_parser(vcf_path)
    return translate_from_genotype_matrix(
        gm=gm, positions=positions, model=model,
        blocks=blocks, pivot_pairs=pivot_pairs, sample_ids=sample_ids,
        device=device, B=B, cache_matching=cache_matching,
        n_reps=n_reps, base_seed=base_seed, top_k=top_k,
        devices=devices, B_per_device=B_per_device,
        progress=progress, decode_bar=decode_bar, build_workers=build_workers,
        use_fast_process_per_gpu=use_fast_process_per_gpu, adapter=adapter, mutation_rate=mutation_rate, residual_model=residual_model, availability_mask=availability_mask
    )


def translate_from_ts(
    ts,
    model,
    blocks=[(0e6, 1e6)],
    pivot_pairs=[(0, 1)],
    sample_ids=np.arange(0, 50),
    device="cuda",
    B=24,
    cache_matching=False,
    n_reps=15,
    base_seed=1234,
    top_k=50,
    devices=None,
    B_per_device: int | None = None,
    progress: bool = True,
    decode_bar: bool = False,
    build_workers: int = 0,
    use_fast_process_per_gpu: bool = False,  # NEW
    adapter: torch.nn.Module | None = None,
    mutation_rate: float | None = None,
    residual_model: bool = False,
    availability_mask: np.ndarray | None = None
):
    positions = ts.tables.sites.position
    gm = ts.genotype_matrix().T
    return translate_from_genotype_matrix(
        gm=gm, positions=positions, model=model,
        blocks=blocks, pivot_pairs=pivot_pairs, sample_ids=sample_ids,
        device=device, B=B, cache_matching=cache_matching,
        n_reps=n_reps, base_seed=base_seed, top_k=top_k,
        devices=devices, B_per_device=B_per_device,
        progress=progress, decode_bar=decode_bar, build_workers=build_workers,
        use_fast_process_per_gpu=use_fast_process_per_gpu, adapter=adapter, mutation_rate=mutation_rate, residual_model=residual_model, availability_mask=availability_mask
    )


def translate(
    input_data,
    data_type,
    model,
    blocks=[(0e6, 1e6)],
    pivot_pairs=[(0, 1)],
    sample_ids=np.arange(0, 50),
    device="cuda",
    B=128,
    cache_matching=True,
    n_reps=15,
    base_seed=1234,
    top_k=50,
    devices=None,
    B_per_device: int | None = None,
    progress: bool = True,
    decode_bar: bool = True,
    build_workers: int = 8,
    use_fast_process_per_gpu: bool = True,  
    adapter: torch.nn.Module | None = None,
    mutation_rate: float | None = None,
    residual_model: bool = False,
    availability_mask: np.ndarray | None = None
):
    if data_type == "vcf":
        return translate_from_vcf(
            input_data, model, blocks, pivot_pairs, sample_ids,
            device, B, cache_matching, n_reps, base_seed, top_k,
            devices=devices, B_per_device=B_per_device,
            progress=progress, decode_bar=decode_bar, build_workers=build_workers,
            use_fast_process_per_gpu=use_fast_process_per_gpu, adapter=adapter, mutation_rate=mutation_rate, residual_model=residual_model, availability_mask=availability_mask
        )
    elif data_type == "ts":
        return translate_from_ts(
            input_data, model, blocks, pivot_pairs, sample_ids,
            device, B, cache_matching, n_reps, base_seed, top_k,
            devices=devices, B_per_device=B_per_device,
            progress=progress, decode_bar=decode_bar, build_workers=build_workers,
            use_fast_process_per_gpu=use_fast_process_per_gpu, adapter=adapter, mutation_rate=mutation_rate, residual_model=residual_model, availability_mask=availability_mask
        )
    elif data_type == "gm":
        gm, positions = input_data
        return translate_from_genotype_matrix(
            gm=gm, positions=positions, model=model,
            blocks=blocks, pivot_pairs=pivot_pairs, sample_ids=sample_ids,
            device=device, B=B, cache_matching=cache_matching,
            n_reps=n_reps, base_seed=base_seed, top_k=top_k,
            devices=devices, B_per_device=B_per_device,
            progress=progress, decode_bar=decode_bar, build_workers=build_workers,
            use_fast_process_per_gpu=use_fast_process_per_gpu, adapter=adapter, mutation_rate=mutation_rate, residual_model=residual_model, availability_mask=availability_mask
        )
    else:
        raise ValueError("data_type must be one of 'vcf', 'ts', or 'gm'")
