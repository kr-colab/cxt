"""Unified inference API for cxt.

Replaces ``api.py`` and ``api2.py`` with a single entry point:

    translate(input_data, model, blocks, pivot_pairs, ...)

Accepts tree sequences, VCF paths, or (genotype_matrix, positions) tuples.
"""

from __future__ import annotations

import gc
import copy
import threading
from typing import List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm, trange

from cxt.sfs import build_src, basic_filtering


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 324
TIMES = np.linspace(3, 17, GRID_SIZE)


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

_MASK_CACHE: dict = {}


def generate_causal_mask(seq_len: int, full_attention_n: int = 0, device="cpu"):
    """Build a boolean causal attention mask with an optional fully-visible prefix.

    Parameters
    ----------
    seq_len : int
        Total sequence length.
    full_attention_n : int
        The first *full_attention_n* tokens attend to each other
        bidirectionally (used for the encoder portion of the
        source-target concatenation).
    device : str
        Torch device for the mask tensor.

    Returns
    -------
    mask : Tensor
        Boolean mask of shape ``(1, 1, seq_len, seq_len)``.
    """
    key = (str(device), seq_len, full_attention_n)
    if key not in _MASK_CACHE:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask[:full_attention_n, :full_attention_n] = 1
        _MASK_CACHE[key] = mask.bool().unsqueeze(0).unsqueeze(0)
    return _MASK_CACHE[key]


# ---------------------------------------------------------------------------
# VCF parser
# ---------------------------------------------------------------------------

def vcf_parser(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a VCF file into (positions, genotype_matrix)."""
    vcf = pd.read_csv(path, comment="#", sep="\t", header=None)
    positions = vcf.iloc[:, 1].to_numpy(dtype=np.float32)
    assert pd.api.types.is_numeric_dtype(vcf.iloc[:, 1])

    for col in vcf.columns:
        val = vcf.iloc[0, col]
        if isinstance(val, str) and ("|" in val or "/" in val):
            sample_start = col
            break
    else:
        raise ValueError("No genotype columns found")

    haplo = [
        vcf[c].str.split(r"[|/]", expand=True).astype(int)
        for c in vcf.columns[sample_start:]
    ]
    gm = pd.concat(haplo, axis=1).to_numpy(dtype=np.int32)
    return positions, gm.T


# ---------------------------------------------------------------------------
# Token-index to log-TMRCA conversion
# ---------------------------------------------------------------------------

def to_log_times(yhat, rep_mode=False):
    """Convert discrete token indices to log-TMRCA values on the output grid.

    Parameters
    ----------
    yhat : Tensor
        Token index predictions from :func:`generate`.
    rep_mode : bool
        If True, *yhat* has shape ``(N, n_reps, T)``; the returned
        array is transposed to ``(n_reps, N, T-1)``.

    Returns
    -------
    log_tmrca : ndarray
        Continuous log-TMRCA values looked up from the discretisation grid.
    """
    if rep_mode:
        return TIMES[yhat[:, :, 1:].cpu().numpy() - 2].transpose(1, 0, 2)
    return TIMES[yhat[:, 1:].cpu().numpy() - 2]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _make_generators(N, device, base_seed):
    return [
        torch.Generator(device=device).manual_seed(int(base_seed) + i)
        for i in range(N)
    ]


def _sample_per_row(probs, generators, row_ids):
    """Robust per-row categorical sampling with deterministic generators."""
    probs = torch.clamp(probs, min=0)
    s = probs.sum(dim=-1, keepdim=True)
    zero = s == 0
    if zero.any():
        probs = probs.masked_fill(zero, 1.0 / probs.size(-1))
        s = s.masked_fill(zero, 1.0)
    probs = probs / s

    cdf = probs.cumsum(dim=-1)
    cdf[:, -1] = 1.0
    B = probs.size(0)
    u = torch.empty(B, 1, device=probs.device)
    for i in range(B):
        u[i, 0] = torch.rand((), device=probs.device, generator=generators[int(row_ids[i])])
    idx = torch.searchsorted(cdf, u, right=True)
    return torch.clamp(idx, 0, probs.size(-1) - 1).long()


# ---------------------------------------------------------------------------
# KV cache management
# ---------------------------------------------------------------------------

def _resize_kv_cache(model, B, T=1001):
    """Resize KV caches without using register_buffer (avoids state_dict issues)."""
    for block in model.transformer.h:
        attn = block.attn
        device = next(model.parameters()).device
        shape = (B, attn.n_head, T, attn.head_size)
        attn.cache_k = torch.zeros(shape, device=device)
        attn.cache_v = torch.zeros(shape, device=device)


def _ensure_cache(model, B, T=1001):
    cur = getattr(model, "_cache_B", None)
    if cur != B:
        _resize_kv_cache(model, B, T)
        model._cache_B = B


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model,
    src: torch.Tensor,
    B: int = 20,
    device: str = "cuda",
    top_k: int | None = 50,
    base_seed: int = 1234,
    cache_matching: bool = False,
    progress: bool = True,
    decode_bar: bool = False,
    adapter=None,
) -> torch.Tensor:
    """Autoregressive generation with KV cache.

    Parameters
    ----------
    model : TokenFreeDecoder
        Model in eval mode.
    src : Tensor
        Source features of shape ``(N, 2, W_scales, 500, num_samples)``.
    B : int
        Micro-batch size (number of sequences decoded at once).
    device : str
        Device for generation (e.g. ``"cuda"`` or ``"cpu"``).
    top_k : int or None
        If set, restrict sampling to the *top_k* most likely tokens.
    base_seed : int
        Base seed for per-row deterministic generators.
    cache_matching : bool
        Dynamically resize KV caches to match batch size.
    progress : bool
        Show a tqdm progress bar over micro-batches.
    decode_bar : bool
        Show a per-token progress bar inside each micro-batch.
    adapter : nn.Module or None
        Optional :class:`~cxt.train.IEAdapter` applied to *src* before
        the backbone forward pass.

    Returns
    -------
    tokens : Tensor
        Generated token indices of shape ``(N, 501)``
        (includes the start token).
    """
    model.eval()
    N = src.size(0)
    gens = _make_generators(N, device, base_seed)
    outs = []
    attn_mask = generate_causal_mask(1001, full_attention_n=501, device=device)

    chunk_iter = range(0, N, B)
    if progress:
        chunk_iter = tqdm(chunk_iter, total=(N + B - 1) // B,
                          desc=f"Generate @ {device}", leave=False)

    with torch.inference_mode():
        for start in chunk_iter:
            end = min(start + B, N)
            batch_src = src[start:end].to(device, non_blocking=True)
            curB = batch_src.size(0)
            row_ids = torch.arange(start, end, device=device)

            if cache_matching:
                _ensure_cache(model, curB)

            if adapter is not None:
                batch_src = adapter(batch_src)

            model(batch_src, None, attn_mask, calculate_loss=False, use_cache=True, position=0)
            idx = torch.ones(curB, 1, dtype=torch.long, device=device)

            token_range = trange(500, 1000, leave=False) if decode_bar else range(500, 1000)
            for i in token_range:
                logits = model(batch_src, idx[:, -1:], attn_mask,
                               calculate_loss=False, use_cache=True, position=i)
                logits = logits[:, -1, :]
                if top_k is not None:
                    tk = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, tk)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_token = _sample_per_row(probs, gens, row_ids)
                idx = torch.cat([idx, next_token], dim=1)

            outs.append(idx)
            if hasattr(model, "clear_cache"):
                model.clear_cache()

    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Multi-GPU generation (threaded, one model replica per GPU)
# ---------------------------------------------------------------------------

def _replicate_to_device(model, device):
    replica = copy.deepcopy(model)
    replica.to(device)
    replica.eval()
    return replica


def _thread_worker(replica, src_chunk, device, B, top_k, base_seed,
                   cache_matching, decode_bar, adapter, result_list, idx):
    try:
        y = generate(
            replica, src_chunk, B=B, device=device, top_k=top_k,
            base_seed=base_seed, cache_matching=cache_matching,
            progress=False, decode_bar=decode_bar, adapter=adapter,
        )
        result_list[idx] = y.cpu()
    except Exception as e:
        result_list[idx] = e


def multi_gpu_generate(
    model,
    src: torch.Tensor,
    devices: List[str],
    B_per_device: int = 20,
    top_k: int = 50,
    base_seed: int = 1234,
    cache_matching: bool = False,
    progress: bool = True,
    decode_bar: bool = False,
    adapter=None,
) -> torch.Tensor:
    """Shard source across GPUs and generate in parallel threads.

    Creates one deep-copy of *model* (and *adapter*) per device, splits
    *src* evenly, and runs :func:`generate` in parallel Python threads.

    Parameters
    ----------
    model : TokenFreeDecoder
        Model on CPU (will be replicated per device).
    src : Tensor
        Source features of shape ``(N, ...)``.
    devices : list[str]
        List of CUDA device strings, e.g. ``["cuda:0", "cuda:1"]``.
    B_per_device : int
        Micro-batch size per device.
    top_k : int
        Top-k sampling cutoff.
    base_seed : int
        Base seed forwarded to :func:`generate`.
    cache_matching : bool
        Resize KV caches per micro-batch.
    progress : bool
        Show progress bars.
    decode_bar : bool
        Per-token progress bars.
    adapter : nn.Module or None
        Optional adapter replicated per device.

    Returns
    -------
    tokens : Tensor
        Concatenated token indices on CPU, shape ``(N, 501)``.
    """
    K = len(devices)
    N = src.size(0)
    shards = []
    for i in range(K):
        lo = (N * i) // K
        hi = (N * (i + 1)) // K
        shards.append(src[lo:hi])

    replicas = [_replicate_to_device(model, d) for d in devices]
    adapter_replicas = (
        [_replicate_to_device(adapter, d) for d in devices]
        if adapter is not None else [None] * K
    )
    results = [None] * K
    threads = []
    for i in range(K):
        t = threading.Thread(
            target=_thread_worker,
            args=(replicas[i], shards[i], devices[i], B_per_device, top_k,
                  base_seed, cache_matching, decode_bar, adapter_replicas[i], results, i),
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for r in results:
        if isinstance(r, Exception):
            raise r

    Y = torch.cat(results, dim=0)
    del replicas, adapter_replicas
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return Y


# ---------------------------------------------------------------------------
# Source building (multiprocess)
# ---------------------------------------------------------------------------

def _build_one_task(task):
    """Worker: build source tensor for one (block, pivot_pair)."""
    return build_src(**task)


def _build_sources(tasks, workers: int = 0, progress: bool = True):
    if workers > 1:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_build_one_task, t): i for i, t in enumerate(tasks)}
            it = as_completed(futs)
            if progress:
                it = tqdm(it, total=len(tasks), desc="Building sources", leave=False)
            for fut in it:
                results[futs[fut]] = fut.result()
        return np.stack(results)
    else:
        it = tasks
        if progress:
            it = tqdm(it, total=len(tasks), desc="Building sources", leave=False)
        return np.stack([build_src(**t) for t in it])


def _prepare_tasks(gm, positions, blocks, pivot_pairs, step_size,
                   availability_mask, use_interpolation, missingness_bitmask):
    tasks = []
    index_map = []
    for b_idx, (bstart, bend) in enumerate(blocks):
        seq_len = int(bend - bstart)
        mask_b = (positions >= bstart) & (positions < bend)
        bpos = positions[mask_b] - bstart
        bgm = gm[:, mask_b]

        if availability_mask is not None:
            ba = availability_mask[int(bstart):int(bend)]
            n = len(ba) // step_size
            ba = ba[: n * step_size].reshape(n, step_size).mean(axis=1)
        else:
            ba = None

        bm = (
            missingness_bitmask[int(bstart):int(bend)]
            if missingness_bitmask is not None
            else None
        )

        bgm, bpos = basic_filtering(bgm, bpos)

        for p_idx, (pA, pB) in enumerate(pivot_pairs):
            tasks.append(dict(
                block_positions=bpos, block_gm=bgm,
                pivot_id_A=pA, pivot_id_B=pB,
                sequence_length=float(seq_len), step_size=step_size,
                availability_mask=ba, use_interpolation=use_interpolation,
                missingness_bitmask=bm,
            ))
            index_map.append([b_idx, p_idx])

    return tasks, np.array(index_map)


# ---------------------------------------------------------------------------
# Bias correction wrapper
# ---------------------------------------------------------------------------

def _apply_bias_correction(Y, gm, positions, index_map, blocks, pivot_pairs,
                           mutation_rate, availability_mask, missingness_bitmask):
    """Apply per-block stochastic bias correction.

    Y has shape (n_reps, N, 500) when n_reps > 1, or (N, 500) when n_reps == 1.
    The correction function expects predictions with ndim == 3, i.e.
    (n_reps, n_items_in_block, 500).
    """
    from cxt.correction import stochastic_diversity_bias_correction_v2

    is_3d = Y.ndim == 3
    if not is_3d:
        Y = Y[np.newaxis]

    rng = np.random.default_rng(1234)

    corrected = np.empty_like(Y)
    for b_idx, (bstart, bend) in enumerate(blocks):
        seq_len = int(bend - bstart)
        step_size = seq_len // 500
        mask_b = (positions >= bstart) & (positions < bend)
        bgm = gm[:, mask_b]

        bm = None
        if missingness_bitmask is not None:
            bm_raw = missingness_bitmask[int(bstart):int(bend)]
            n = len(bm_raw) // step_size
            bm = bm_raw[: n * step_size].reshape(n, step_size).mean(axis=1)

        rows = np.flatnonzero(index_map[:, 0] == b_idx)
        if rows.size == 0:
            continue

        tmrca_block = Y[:, rows]
        pp = np.array(pivot_pairs)[index_map[rows, 1]]

        corrected_block = stochastic_diversity_bias_correction_v2(
            genotype_matrix=bgm,
            mutation_rate=mutation_rate,
            predictions=tmrca_block,
            pivot_pairs=pp,
            sequence_length=float(seq_len),
            window_size=step_size,
            mask_missingness=bm,
            rng=rng,
        )
        corrected[:, rows] = corrected_block

    if not is_3d:
        corrected = corrected[0]
    return corrected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_from_genotype_matrix(
    gm: np.ndarray,
    positions: np.ndarray,
    model,
    blocks: list[tuple] = [(0, 1_000_000)],
    pivot_pairs: list[tuple] = [(0, 1)],
    devices: list[str] | None = None,
    B: int = 128,
    B_per_device: int | None = None,
    n_reps: int = 15,
    base_seed: int = 1234,
    top_k: int = 50,
    cache_matching: bool = True,
    progress: bool = True,
    decode_bar: bool = False,
    build_workers: int = 8,
    adapter=None,
    mutation_rate: float | None = None,
    availability_mask: np.ndarray | None = None,
    use_interpolation: bool = False,
    missingness_bitmask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer pairwise TMRCA from a genotype matrix.

    Returns (tmrca, index_map) where tmrca has shape
    ``(n_items, [n_reps,] n_windows)`` and index_map ``(n_items, 2)``
    maps each row to ``[block_idx, pivot_idx]``.
    """
    a, b = blocks[0]
    seq_len = int(b - a)
    step_size = seq_len // 500

    tasks, index_map = _prepare_tasks(
        gm, positions, blocks, pivot_pairs, step_size,
        availability_mask, use_interpolation, missingness_bitmask,
    )
    X_base = _build_sources(tasks, workers=build_workers, progress=progress)

    N = X_base.shape[0]
    B_local = B_per_device or B
    device = devices[0] if devices else "cuda"

    param_dtype = next(model.parameters()).dtype
    X_cpu = torch.as_tensor(X_base, dtype=param_dtype, device="cpu")
    if torch.cuda.is_available():
        X_cpu = X_cpu.pin_memory()

    def _run_generate(src_t, seed):
        if devices and len(devices) > 1:
            return multi_gpu_generate(
                model, src_t, devices=devices, B_per_device=B_local,
                top_k=top_k, base_seed=seed, cache_matching=cache_matching,
                progress=progress, decode_bar=decode_bar, adapter=adapter,
            )
        _adapter = adapter.to(device) if adapter is not None else None
        return generate(
            model.to(device), src_t, B=min(B, src_t.size(0)), device=device,
            top_k=top_k, base_seed=seed, cache_matching=cache_matching,
            progress=progress, decode_bar=decode_bar, adapter=_adapter,
        )

    if n_reps <= 1:
        yhat = _run_generate(X_cpu, base_seed)
        Y = to_log_times(yhat, rep_mode=False)
    else:
        ids = torch.tile(torch.arange(N, dtype=torch.long), (n_reps,))
        world = len(devices) if devices and len(devices) > 1 else 1
        chunk_size = B_local * world
        parts = []
        loop = range(0, ids.numel(), chunk_size)
        if progress:
            loop = tqdm(loop, total=(ids.numel() + chunk_size - 1) // chunk_size,
                        desc="Batches (reps)", leave=False)
        for s in loop:
            e = min(s + chunk_size, ids.numel())
            sel = ids[s:e]
            X_chunk = X_cpu.index_select(0, sel)
            parts.append(_run_generate(X_chunk, base_seed + s))
        yhat = torch.cat(parts, dim=0)
        yhat = yhat.reshape(n_reps, N, *yhat.shape[1:]).transpose(0, 1)
        Y = to_log_times(yhat.contiguous(), rep_mode=True)

    if mutation_rate is not None:
        Y = _apply_bias_correction(
            Y, gm, positions, index_map, blocks, pivot_pairs,
            mutation_rate, availability_mask, missingness_bitmask,
        )

    return Y, index_map


def translate_from_vcf(vcf_path: str, model, **kwargs):
    """Infer TMRCA from a VCF file. See ``translate_from_genotype_matrix``."""
    positions, gm = vcf_parser(vcf_path)
    return translate_from_genotype_matrix(gm=gm, positions=positions, model=model, **kwargs)


def translate_from_ts(ts, model, **kwargs):
    """Infer TMRCA from a tree sequence. See ``translate_from_genotype_matrix``."""
    positions = ts.tables.sites.position
    gm = ts.genotype_matrix().T
    return translate_from_genotype_matrix(gm=gm, positions=positions, model=model, **kwargs)


def translate(
    input_data,
    model,
    blocks: list[tuple] = [(0, 1_000_000)],
    pivot_pairs: list[tuple] = [(0, 1)],
    data_type: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Unified inference entry point.

    Parameters
    ----------
    input_data : tree sequence, VCF path (str), or (gm, positions) tuple.
    model : loaded cxt model (use ``cxt.checkpoint.load_model``).
    blocks : list of (start, end) genomic intervals in bp.
    pivot_pairs : list of (sample_A, sample_B) haploid indices.
    data_type : "ts", "vcf", or "gm". Auto-detected if None.
    **kwargs : forwarded to ``translate_from_genotype_matrix``.

    Returns
    -------
    tmrca : ndarray of log-TMRCA values.
    index_map : ndarray mapping rows to (block_idx, pivot_idx).
    """
    if data_type is None:
        if isinstance(input_data, str):
            data_type = "vcf"
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            data_type = "gm"
        else:
            data_type = "ts"

    if data_type == "vcf":
        return translate_from_vcf(input_data, model, blocks=blocks, pivot_pairs=pivot_pairs, **kwargs)
    elif data_type == "ts":
        return translate_from_ts(input_data, model, blocks=blocks, pivot_pairs=pivot_pairs, **kwargs)
    elif data_type == "gm":
        gm, positions = input_data
        return translate_from_genotype_matrix(
            gm=gm, positions=positions, model=model,
            blocks=blocks, pivot_pairs=pivot_pairs, **kwargs,
        )
    else:
        raise ValueError(f"data_type must be 'ts', 'vcf', or 'gm', got {data_type!r}")
