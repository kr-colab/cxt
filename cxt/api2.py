# ============================================================
# Multi-GPU pipeline + tqdm + MULTIPROCESS SOURCE BUILD
# (same behavior as your working version, but memory-fixed)
# ============================================================

import gc
import copy
import threading
from typing import List, Sequence, Tuple, Optional

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
        window_size=2000, sequence_length=1e6, num_samples=50, step_size=2000):
    """Memory-efficient vectorized calculation of SFS"""
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


def check_blocks(blocks):
    for b in blocks:
        assert isinstance(b, tuple) and len(b) == 2 and b[0] < b[1], \
            "Each block must be a tuple (start, end) with start < end"
        assert b[1] - b[0] == 1e6, "Each block must be 1e6 bp"


def basic_filtering(block_gm, block_positions):
    mask = np.logical_or(
        np.any(block_gm >= 2, axis=0),     # filters out non-biallelic
        block_gm.sum(0) >= 50)             # filters out fixed sites
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
    b_idx, p_idx, block_pos, block_gm, pivot_A, pivot_B = task
    X = build_src(block_pos, block_gm, pivot_A, pivot_B)
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
    for (b_idx, p_idx, block_pos, block_gm, pivot_A, pivot_B) in it:
        Xs.append(build_src(block_pos, block_gm, pivot_A, pivot_B))
    return np.stack(Xs, axis=0)


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


def to_log_times(yhat, TIMES=np.linspace(3, 17, 324), rep_mode=False):
    if rep_mode:
        return TIMES[yhat[:, :, 1:].cpu().numpy() - 2]
    return TIMES[yhat[:, 1:].cpu().numpy() - 2]


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
    # probs: [B, V], row_ids: [B] global indices
    cdf = probs.cumsum(dim=-1)                         # [B, V]
    B = probs.size(0)
    u = torch.empty(B, 1, device=probs.device)
    for i in range(B):
        u[i, 0] = torch.rand((), device=probs.device,
                             generator=generators[int(row_ids[i])])
    idx = torch.searchsorted(cdf, u, right=True)       # [B, 1]
    return idx.long()


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
    if cache_matching:
        ensure_cache_B(model, B)
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
):
    # bind CUDA context to this thread/device
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":")[1]))
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
        )
    out_list[out_index] = y.to("cpu", non_blocking=True)


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
            progress=progress, decode_bar=decode_bar
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
# Core translate (multi-GPU + mp build)
# --------------------------
def translate_from_genotype_matrix(
        gm,
        positions,
        model,
        one_mb_blocks = [(0e6, 1e6)],
        pivot_ids = [(0, 1)],
        sample_ids = np.arange(0, 50),
        device="cuda", B=24,
        cache_matching=False,
        n_reps: int = 15,
        base_seed: int = 1234,
        top_k: int = 50,
        devices: Sequence[str] | None = None,
        B_per_device: int | None = None,
        progress: bool = True,
        decode_bar: bool = False,
        build_workers: int = 0  # 0/1 => serial; >1 => ProcessPoolExecutor
    ):
    """
    Returns yhat of shape (n_items, n_reps, ...).
    n_items = len(one_mb_blocks) * len(pivot_ids)
    """
    check_blocks(one_mb_blocks)
    gm_samples = gm[sample_ids, :]

    # ---------- build sources ----------
    tasks, index_map = _prepare_build_tasks(gm_samples, positions, one_mb_blocks, pivot_ids)

    if build_workers and build_workers > 1:
        X_base = _build_sources_parallel(tasks, progress=progress, max_workers=build_workers)
    else:
        X_base = _build_sources_serial(tasks, progress=progress)

    N = X_base.shape[0]
    B_local = B if B_per_device is None else B_per_device
    tensor_device = device if not (devices and len(devices) > 1) else "cpu"

    if n_reps <= 1:
        X = torch.tensor(X_base, dtype=torch.float32, device=tensor_device)
        if devices and len(devices) > 1:
            yhat = multi_gpu_generate(
                model=model, src=X, devices=devices, B_per_device=B_local,
                top_k=top_k, base_seed=base_seed, cache_matching=cache_matching,
                progress=progress, decode_bar=decode_bar
            )
        else:
            y_list = []
            loop = _iter_chunks(N, B)
            if progress:
                loop = tqdm(loop, total=(N + B - 1)//B, desc="Batches", leave=False)
            for s, e in loop:
                y_chunk = generate(model, X[s:e], B=min(B, e - s), device=device,
                                   top_k=top_k, base_seed=base_seed + s,
                                   cache_matching=cache_matching,
                                   progress=False, decode_bar=decode_bar)
                y_list.append(y_chunk)
            yhat = torch.cat(y_list, dim=0)
        return to_log_times(yhat, rep_mode=False), index_map

    # replicate for reps
    X_rep = np.repeat(X_base, repeats=n_reps, axis=0)
    X_rep = X_rep.reshape(N, n_reps, *X_base.shape[1:])
    X_rep = X_rep.transpose(1, 0, *range(2, X_rep.ndim)).reshape(N * n_reps, *X_base.shape[1:])

    X_t = torch.tensor(X_rep, dtype=torch.float32, device=tensor_device)
    if devices and len(devices) > 1:
        Y = multi_gpu_generate(
            model=model, src=X_t, devices=devices, B_per_device=B_local,
            top_k=top_k, base_seed=base_seed, cache_matching=cache_matching,
            progress=progress, decode_bar=decode_bar
        )
    else:
        y_chunks = []
        M = X_t.shape[0]
        loop = _iter_chunks(M, B)
        if progress:
            loop = tqdm(loop, total=(M + B - 1)//B, desc="Batches", leave=False)
        for s, e in loop:
            y_chunk = generate(model, X_t[s:e], B=min(B, e - s), device=device,
                               top_k=top_k, base_seed=base_seed + s,
                               cache_matching=cache_matching,
                               progress=False, decode_bar=decode_bar)
            y_chunks.append(y_chunk)
        Y = torch.cat(y_chunks, dim=0)

    # undo interleave → [N, n_reps, ...]
    Y = Y.reshape(n_reps, N, *Y.shape[1:]).transpose(0, 1).contiguous()
    return to_log_times(Y, rep_mode=True), index_map


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


def translate_from_vcf(vcf_path, model, one_mb_blocks=[(0e6,1e6)], pivot_ids=[(0,1)],
                       sample_ids=np.arange(0,50), device="cuda", B=24,
                       cache_matching=False, n_reps=15, base_seed=1234, top_k=50,
                       devices: Sequence[str] | None = None, B_per_device: int | None = None,
                       progress: bool = True, decode_bar: bool = False, build_workers: int = 0):
    positions, gm = vcf_parser(vcf_path)
    return translate_from_genotype_matrix(
        gm=gm, positions=positions, model=model,
        one_mb_blocks=one_mb_blocks, pivot_ids=pivot_ids, sample_ids=sample_ids,
        device=device, B=B, cache_matching=cache_matching,
        n_reps=n_reps, base_seed=base_seed, top_k=top_k,
        devices=devices, B_per_device=B_per_device,
        progress=progress, decode_bar=decode_bar, build_workers=build_workers)


def translate_from_ts(ts, model, one_mb_blocks=[(0e6,1e6)], pivot_ids=[(0,1)],
                      sample_ids=np.arange(0,50), device="cuda", B=24,
                      cache_matching=False, n_reps=15, base_seed=1234, top_k=50,
                      devices: Sequence[str] | None = None, B_per_device: int | None = None,
                      progress: bool = True, decode_bar: bool = False, build_workers: int = 0):
    positions = ts.tables.sites.position
    gm = ts.genotype_matrix().T
    return translate_from_genotype_matrix(
        gm=gm, positions=positions, model=model,
        one_mb_blocks=one_mb_blocks, pivot_ids=pivot_ids, sample_ids=sample_ids,
        device=device, B=B, cache_matching=cache_matching,
        n_reps=n_reps, base_seed=base_seed, top_k=top_k,
        devices=devices, B_per_device=B_per_device,
        progress=progress, decode_bar=decode_bar, build_workers=build_workers)


def translate(input_data, data_type, model, one_mb_blocks=[(0e6,1e6)], pivot_ids=[(0,1)],
              sample_ids=np.arange(0,50), device="cuda", B=24, cache_matching=False,
              n_reps=15, base_seed=1234, top_k=50,
              devices: Sequence[str] | None = None, B_per_device: int | None = None,
              progress: bool = True, decode_bar: bool = False, build_workers: int = 0):
    if data_type == "vcf":
        return translate_from_vcf(input_data, model, one_mb_blocks, pivot_ids,
                                  sample_ids, device, B, cache_matching,
                                  n_reps, base_seed, top_k,
                                  devices=devices, B_per_device=B_per_device,
                                  progress=progress, decode_bar=decode_bar, build_workers=build_workers)
    elif data_type == "ts":
        return translate_from_ts(input_data, model, one_mb_blocks, pivot_ids,
                                 sample_ids, device, B, cache_matching,
                                 n_reps, base_seed, top_k,
                                 devices=devices, B_per_device=B_per_device,
                                 progress=progress, decode_bar=decode_bar, build_workers=build_workers)
    elif data_type == "gm":
        gm, positions = input_data
        return translate_from_genotype_matrix(
            gm=gm, positions=positions, model=model,
            one_mb_blocks=one_mb_blocks, pivot_ids=pivot_ids, sample_ids=sample_ids,
            device=device, B=B, cache_matching=cache_matching,
            n_reps=n_reps, base_seed=base_seed, top_k=top_k,
            devices=devices, B_per_device=B_per_device,
            progress=progress, decode_bar=decode_bar, build_workers=build_workers)
    else:
        raise ValueError("data_type must be one of 'vcf', 'ts', or 'gm'")


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




def _prepare_build_tasks(gm_samples, positions, one_mb_blocks, pivot_ids):
    tasks = []
    index_map = []
    for b_idx, (block_start, block_end) in enumerate(one_mb_blocks):
        block_mask = (positions >= block_start) & (positions < block_end)
        block_pos_abs = positions[block_mask]
        block_gm = gm_samples[:, block_mask]

        # block-relative coordinates
        block_pos_rel = block_pos_abs - block_start

        # optional: stricter, symmetric filtering
        block_gm, block_pos_rel = basic_filtering(block_gm, block_pos_rel)

        for p_idx, (pivot_A, pivot_B) in enumerate(pivot_ids):
            tasks.append((b_idx, p_idx, block_pos_rel, block_gm, pivot_A, pivot_B))
            index_map.append((b_idx, p_idx))
    return tasks, np.array(index_map, dtype=np.int32)

def build_src(block_positions, block_gm, pivot_id_A, pivot_id_B):
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
            return np.zeros((int(np.ceil(1e6/2000)), 50), dtype=np.int32)
        return calculate_window_sfs_vectorized(
            positions=pos.astype(np.float32),
            pivot_frequencies=f.astype(np.int32),
            window_size=2000 * win_mult,
            sequence_length=1e6,
            num_samples=50,
            step_size=2000,
        )

    w_multipliers = (2, 8, 32, 64)
    n_w = int(np.ceil(1e6/2000))  # 500
    Xs_xor  = np.zeros((len(w_multipliers), n_w, 50), dtype=np.int32)
    Xs_xnor = np.zeros((len(w_multipliers), n_w, 50), dtype=np.int32)

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

