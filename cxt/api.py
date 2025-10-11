import numpy as np
import torch
import torch.multiprocessing as mp
from cxt.train import LitTokenFreeDecoder

from cxt.config import BroadModelConfig 
TokenFreeDecoderConfig = BroadModelConfig
config = TokenFreeDecoderConfig()
from torch.serialization import add_safe_globals

import sys
import torch
from torch.serialization import add_safe_globals

def load_model(config, model_path: str, device: str = "cuda"):
    # ——— 0) Prep the unpickler so that if it DOES need to unpickle
    #        TokenFreeDecoderConfig, it can find it safely:
    add_safe_globals([TokenFreeDecoderConfig])
    # also inject it into __main__, so standard pickle lookups succeed
    sys.modules["__main__"].TokenFreeDecoderConfig = TokenFreeDecoderConfig

    # ——— 1) Build your Lightning wrapper
    lit_model = LitTokenFreeDecoder(config)

    # ——— 2) Try loading ONLY the tensor weights (no Python objects)
    try:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True  # PL 2.x feature: skip all pickled objects
        )
    except Exception as err:
        # if it still complains about unsafe globals, load full checkpoint
        if "Unsupported global" in str(err):
            
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False
            )
        else:
            # any other error, re-raise
            raise

    # ——— 3) Normalize to a raw state_dict
    state_dict = checkpoint.get("state_dict", checkpoint)
    lit_model.load_state_dict(state_dict, strict=False)

    # ——— 4) Peel off, move, cache, and eval
    model = lit_model.model
    model.to(device)
    model.cache_to_device(device)
    model.eval()
    return model



from collections import defaultdict
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import torch
from cxt.inference import generate
from cxt.utils import TIMES
import os
import subprocess


config.device = 'cpu'

import pandas as pd
import numpy as np

def vcf_parser(path):
    vcf = pd.read_csv(path, comment='#', sep='\t', header=None)

    # Standard VCF: column 1 (0-based index) is POS
    pos_col = 1
    positions = vcf.iloc[:, pos_col]

    assert pd.api.types.is_numeric_dtype(positions), "POS column must be numeric"
    assert positions.is_monotonic_increasing, "POS column must be sorted"
    positions = positions.to_numpy(dtype=np.float32)

    # Find first genotype-like column
    for col in vcf.columns:
        val = vcf.iloc[0, col]
        if isinstance(val, str) and ('|' in val or '/' in val):
            sample_start_col = col
            break
    else:
        raise AssertionError("No genotype columns found! Expected values like '0|1' or '1/1'")

    vcf = vcf.loc[:, sample_start_col:]
    haplo = [vcf[col].str.split(r"[|/]", expand=True).astype(int) for col in vcf.columns]
    genotypes = pd.concat(haplo, axis=1).to_numpy(dtype=np.int32)

    return positions, genotypes.T


from cxt.utils import calculate_window_sfs_vectorized


def build_X(
        gm, positions, window_size=4000, step_size=2000,
        xor_ops=None, pivot_A=0, pivot_B=1):
    assert xor_ops is not None, "xor_ops must be provided"
    sequence_length = 1e6
    num_samples = gm.shape[0]
    assert num_samples == 50, "Number of samples must be 50"
    mask = np.logical_or(np.any(gm >= 2, axis=0), gm.sum(0) >= num_samples)
    gm = gm[:, ~mask]
    positions = positions[~mask]
    frequencies = gm.sum(0)
    xor_freqs = frequencies * xor_ops(gm[pivot_A], gm[pivot_B])    
    # Pre-allocate output array
    w_multipliers = np.array([2, 8, 32, 64])
    Xs = np.zeros((len(w_multipliers), 
                   int(np.ceil(sequence_length / step_size)), 
                   num_samples), dtype=int)
    # Calculate for each window size
    for i, w in enumerate(w_multipliers):
        Xs[i] = calculate_window_sfs_vectorized(
            site_positions=positions,
            pivot_frequencies=xor_freqs,
            window_size=window_size * w,
            step_size=step_size,
            sequence_length=sequence_length,
            num_samples=num_samples
        )
    return Xs

xor = lambda a, b: (a^b).astype(int)
xnor = lambda a, b: (1 - xor(a, b)).astype(int)




def to_input(gm, positions, pivot_A, pivot_B):
    Xor = build_X(
        gm=gm, positions=positions,
        window_size=2000, step_size=2000,
        xor_ops=xor,
        pivot_A=pivot_A, pivot_B=pivot_B
    )
    Xnor = build_X(
        gm=gm, positions=positions,
        window_size=2000, step_size=2000,
        xor_ops=xnor,
        pivot_A=pivot_A, pivot_B=pivot_B
    )
    X = np.stack([Xor, Xnor], axis=0).astype(np.float16)
    X = np.log1p(X)
    return X

def process_pair(args):
    gm, positions, pivot_A, pivot_B = args
    return to_input(gm, positions, pivot_A, pivot_B)

def totensorlist(l): return [torch.tensor(a) for a in l]

def prepare_data(
        gm_list, positions_list,
        pivot_combinations,
        device='cuda', num_processes=50):
    
    args = []
    for gm, positions in zip(gm_list, positions_list):
        args.extend([(gm, positions, a, b) for a, b in pivot_combinations])

    with Pool(num_processes) as pool: 
        src_list = list(tqdm(pool.imap(process_pair, args), total=len(args)))

    src_list = totensorlist(src_list)
    src = torch.stack(src_list, dim=0)
    src = src.to(device).to(torch.float32)
    
    return src

def multi_gpu_inference(rank, models, Xs, devices, queue, num_replicates):
    seed = 0xC0FFEE00 + rank  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = devices[rank]
    model = models[rank]
    model.to(device)
    model.cache_to_device(device)
    results = []
    for idx, X in enumerate(Xs):
        yhat_list = []
        progress_bar = tqdm(range(num_replicates),
                            desc=f"[GPU {rank}] Index {idx} ⏳ Processing", position=rank, leave=True)
        for _ in progress_bar:
            yhat = generate(model, X.to(device), B=X.shape[0], device=device)
            yhat = yhat[:, 1:].cpu().numpy() - 2
            yhat = TIMES[yhat]
            yhat_list.append(yhat)
        results.append((idx, np.stack(yhat_list)))
    queue.put((rank, results))

def check_and_download_model(cache_dir, repo_url, rel_path):
    """
    - cache_dir: where to clone (and cache) the repo
    - repo_url: e.g. "https://github.com/kevinkorfmann/models.git"
    - rel_path: path *inside* the repo to your .ckpt, 
                e.g. "cxt/broad-model/epoch=0-step=11296.ckpt"
    """
    dest = os.path.join(cache_dir, rel_path)
    if not os.path.exists(dest):
        os.makedirs(cache_dir, exist_ok=True)
        print("Cloning repo…")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, cache_dir], check=True)
        print("Pulling LFS file…")
        subprocess.run(["git", "-C", cache_dir, "lfs", "pull", "-I", rel_path], check=True)
    else:
        print("Using cached model at", dest)
    return dest

def translate(
    vcf,
    pivot_combinations=[(0, 1)],
    num_replicates=15,
    mutation_rate=None,
    devices=None,
    return_uncorrected: bool = False,
):

    # ——— Parsing VCF ———
    positions, gm = vcf_parser(vcf)
    bs = 10**6
    # make sure your block‐indices are ints
    bi = (positions // bs).astype(int)
    # bucket up positions & column‐indices
    pos = defaultdict(list)
    idx = defaultdict(list)
    for i, p in enumerate(positions):
        b = bi[i]
        pos[b].append(int(p) % bs)
        idx[b].append(i)
    # total number of blocks as a pure Python int
    n_blocks = int(bi.max()) + 1
    # now real lists
    positions_list = [np.array(pos[b], dtype=np.float32) for b in range(n_blocks)]
    gm_list        = [gm[:, idx[b]] for b in range(n_blocks)]

    # ——— Device setup ———
    if devices is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPUs required for this model. No GPUs found.")
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    num_devices = len(devices)
    if num_replicates % num_devices != 0:
        raise ValueError("num_replicates must be divisible by number of GPUs")
    replicates_per_gpu = num_replicates // num_devices
    print(f"→ Using GPUs: {devices}")

    # ——— Download checkpoint if needed ———
    model_path = check_and_download_model(
        "model_cache",
        "https://github.com/kevinkorfmann/models.git",
        "cxt/broad-model/epoch=0-step=11296.ckpt"
    )

    # ——— Prepare multiprocessing ———
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    queue = manager.Queue()

    # ——— Load one model per GPU (in half-precision) ———
    models = []
    for device in devices:
        model = load_model(config=config, model_path=model_path, device='cpu')
        models.append(model)

    # ——— Parse VCF and prepare input chunks ———
    X = prepare_data(
        gm_list=gm_list,
        positions_list=positions_list,
        pivot_combinations=pivot_combinations,
        device="cpu",
        num_processes=1
    )
    num_pivots = len(pivot_combinations)
    Xs = [X[i : i + num_pivots] for i in range(0, len(X), num_pivots)]
    print(f"→ Prepared {len(Xs)} samples with {num_pivots} pivots each.")

    # ——— Run one process per GPU ———
    mp.spawn(
        multi_gpu_inference,
        args=(models, Xs, devices, queue, replicates_per_gpu),
        nprocs=num_devices,
        join=True
    )

    # ——— Collect & stitch predictions ———
    results = [queue.get() for _ in range(num_devices)]
    vcf_results = defaultdict(list)
    for _, res_list in results:
        for idx, yhat in res_list:
            vcf_results[idx].append(yhat)

    all_preds = []
    for idx in sorted(vcf_results):
        all_preds.append(np.concatenate(vcf_results[idx], axis=0))

    tmrca = np.concatenate(all_preds, axis=0)
    n_groups = tmrca.shape[0] // num_replicates
    tmrca = tmrca.reshape(n_groups, num_replicates, *tmrca.shape[1:]) 

    if mutation_rate != None:
        uncorrected_tmrca = tmrca.copy()
        for i, gm in enumerate(gm_list):
            tmrca[i] = stochastic_diversity_bias_correction_v2(
                genotype_matrix=gm,
                mutation_rate=mutation_rate,
                predictions=tmrca[i],
                pivot_pairs=np.array(pivot_combinations),
                rng=np.random.default_rng(1337),
            )
        if return_uncorrected:
            return uncorrected_tmrca, tmrca
    return tmrca





