import torch
from cxt.train import LitTokenFreeDecoder
import torch.nn.functional as F
#from cxt.utils import generate_causal_mask
from cxt.train import generate_causal_mask

from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

from cxt.utils import post_process, accumulating_mses, mse
from cxt.utils import simulate_parameterized_tree_sequence, TIMES


from cxt.utils import process_pair
def totensorlist(l): return [torch.tensor(a) for a in l]

import torch.multiprocessing as mp
#from cxt.config import TokenFreeDecoderConfig
#from cxt.config import NarrowModelConfig, BroadModelConfig

from collections import defaultdict


BATCH = 190#1225#45#190#1225#190#1225
NUM_SAMPLES = 20#50#10#20#50#20#50

def generate_nokv(model, src, top_k=None, temperature=1, max_len=500, device='cuda'):
    with torch.no_grad():
        B = src.size(0)
        idx = torch.ones(B, 1).long().to(device)
        for _ in range(max_len):
            #with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(src, idx, calculate_loss=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx[:, 1:].cpu().numpy() - 2




def generate_causal_mask(seq_len, full_attention_n=None, device="cpu"):
    full_attention_n = full_attention_n if full_attention_n is not None else 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask[:full_attention_n, :full_attention_n] = 1  # Full attention for first n tokens
    return mask.bool().unsqueeze(0).unsqueeze(0)

#def generate_causal_mask(seq_len, device):
#    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
#    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

def generate(model, src, B=20, device="cuda", clear_cache=True, use_ddp=False):
    #attn_mask = generate_causal_mask(1001, device=device)
    attn_mask = generate_causal_mask(1001, full_attention_n=501, device=device)
    attn_mask = attn_mask.repeat(B, 1, 1, 1)
    idx = torch.ones(B, 1).long().to(device)
    
    with torch.no_grad():
        #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(src, None, attn_mask, calculate_loss=False, use_cache=True, position=0)
            idx = torch.ones(B, 1).long().to(device)
            top_k = 50
            for i in range(500, 1000):
                logits = model(src, idx[:, -1:], attn_mask, calculate_loss=False, use_cache=True, position=i)
                logits = logits[:, -1, :]
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)
        
    if use_ddp and clear_cache: model.module.clear_cache()
    elif clear_cache: model.clear_cache()
    return idx

"""
def load_model(config, model_path=None, device='cuda'):
    model = LitTokenFreeDecoder(config)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #model.load_state_dict(torch.load(model_path,
    #    weights_only=False)['state_dict'], strict=False)
    model = model.model
    model.to(device=device)
    model.cache_to_device(device)
    model.eval()
    return model
"""

def load_model(config, model_path, device="cuda"):
    """
    Lazy loader to avoid circular imports during multiprocessing spawn.
    Imports cxt.train only when called.
    """
    import importlib, torch
    train_mod = importlib.import_module("cxt.train")     # lazy
    LitTokenFreeDecoder = getattr(train_mod, "LitTokenFreeDecoder")

    lit = LitTokenFreeDecoder(config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    lit.load_state_dict(ckpt["state_dict"], strict=False)
    model = lit.model
    del lit, ckpt

    model.to(device)
    if hasattr(model, "cache_to_device"):
        model.cache_to_device(device)
    model.eval()
    return model




def prepare_ts_data(ts: object, num_samples: int, B: int, device='cuda', num_processes=50, offset=0, ignore_target=False) -> tuple:
    """
    Prepares the data for the model by processing pairs of samples from the tree sequence.

    Parameters:
    - ts: The tree sequence object.
    - num_samples: The number of samples to generate.
    - B: The batch size.

    Returns:
    - src: Tensor containing the source data.
    - tgt: Tensor containing the target data.
    """
    args = [(ts, a, b, offset, ignore_target) for a, b in combinations(range(num_samples), 2)]
    with Pool(num_processes) as pool: 
        results = list(tqdm(pool.imap(process_pair, args), total=B))
    src_list, tgt_list = zip(*results)
    src_list = totensorlist(src_list)
    tgt_list = totensorlist(tgt_list)
    src = torch.stack(src_list, dim=0)  
    tgt = torch.stack(tgt_list, dim=0) 
    src = src.to(device).to(torch.float32)
    #src = src.to(torch.bfloat16)
    return src, tgt

def prepare_tss_data(ts_list: list, num_samples: int, B: int, device='cuda', num_processes=50, offset=0, ignore_target=False) -> tuple:
    """
    Prepares the data for the model by processing pairs of samples from multiple tree sequences.

    Parameters:
    - ts_list: A list of tree sequence objects.
    - num_samples: The number of samples to generate.
    - B: The batch size.
    - device: The computing device ('cuda' or 'cpu').
    - num_processes: Number of parallel processes to use.

    Returns:
    - src: Tensor containing the source data.
    - tgt: Tensor containing the target data.
    """
    args = []
    for ts in ts_list:
        args.extend([(ts, a, b, offset, ignore_target) for a, b in combinations(range(num_samples), 2)])

    with Pool(num_processes) as pool: 
        results = list(tqdm(pool.imap(process_pair, args), total=B * len(ts_list)))

    src_list, tgt_list = zip(*results)
    src_list = totensorlist(src_list)
    tgt_list = totensorlist(tgt_list)

    src = torch.stack(src_list, dim=0)
    tgt = torch.stack(tgt_list, dim=0)

    src = src.to(device).to(torch.float32)
    #tgt = tgt.to(device).to(torch.float32)

    return src, tgt


def translate_from_ts(
        ts,
        max_replicates = 20, use_early_stopping = True,
        model_config=None,
        model_path=None,
        device='cuda',
        offset=0,
        ignore_target=False
    ):
    """Assumes sample size of 50 for now and large enough GPU to fit 1225 batch."""

    model = load_model(
        config=model_config, 
        model_path=model_path,
        device=device
    )

    src, tgt = prepare_ts_data(ts, num_samples=50, B=1225, device=device, offset=offset, ignore_target=ignore_target)
    yhats, ytrues = [], []
    for i in range(max_replicates):
        sequence = generate(model, src, B=1225, device=device)
        yhat, ytrue = post_process(tgt, sequence, TIMES)
        yhats.append(yhat)
        ytrues.append(ytrue)
        if not ignore_target:
            if use_early_stopping:
                # early stopping criteria
                if i > 1:
                    mses = accumulating_mses(yhats, ytrues)
                    derivatives = np.diff(mses)
                    if abs(derivatives[-1]) < 0.001:
                        print(f"Stopping at {i} because derivative is {derivatives[-1]}.")
                        break
    yhats = np.stack(yhats)
    ytrues = np.stack(ytrues)  
    return yhats, ytrues


def run_inference(rank, model_path, model_config, src, tgt, devices, queue, num_replicates):
    """Runs inference on a single GPU, processing a subset of replicates."""

    seed = 0xC0FFEE00 + rank  # Ensure unique seeds across runs but stable per GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = devices[rank]

    model = load_model(
        config=model_config(device=device),
        model_path=model_path,
        device=device
    )


    
    yhat_list, ytrue_list = [], []
    progress_bar = tqdm(range(num_replicates),
                         desc=f"[GPU {rank}] ⏳ Processing", position=rank, leave=True)

    for _ in progress_bar:
        sequence = generate(model, src.to(device), B=1225, device=device)
        yhat, ytrue = post_process(tgt, sequence, TIMES)
        yhat_list.append(yhat)
        ytrue_list.append(ytrue)

    queue.put((rank, np.stack(yhat_list), np.stack(ytrue_list)))




def translate_from_ts_multigpu(
        ts,
        max_replicates = 15,
        model_config=None, 
        model_path=None,
        devices=['cuda:0', 'cuda:1', 'cuda:2'],
        num_processes_ts_preproc=30
    ):

    mp.set_start_method('spawn', force=True)
    num_processes = len(devices)
    assert max_replicates % num_processes == 0
    replicates_per_process = max_replicates // num_processes

    src, tgt = prepare_ts_data(
        ts, num_samples=50, B=1225, device='cpu',
        num_processes=num_processes_ts_preproc)

    queue = mp.Manager().Queue()

    mp.spawn(run_inference, args=(model_path, model_config, src, tgt, devices, queue, replicates_per_process), 
             nprocs=num_processes, join=True)

    results = []
    for _ in range(num_processes):
        results.append(queue.get())  # Blocks until an item is available

    results.sort(key=lambda x: x[0]) 
    _, yhat_list, ytrue_list = zip(*results)
    yhat_array = np.concatenate(yhat_list, axis=0)
    ytrue_array = np.concatenate(ytrue_list, axis=0)
    return yhat_array, ytrue_array



def run_inference_jupyter(rank, models, src, tgt, devices, queue, num_replicates):
    """Runs inference on a single GPU, processing a subset of replicates."""

    seed = 0xC0FFEE00 + rank  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = devices[rank]
    model = models[rank]

    model.to(device)
    model.cache_to_device(device)

    yhat_list, ytrue_list = [], []
    progress_bar = tqdm(range(num_replicates),
                         desc=f"[GPU {rank}] ⏳ Processing", position=rank, leave=True)

    for _ in progress_bar:
        sequence = generate(model, src.to(device), B=1225, device=device)
        yhat, ytrue = post_process(tgt, sequence, TIMES)
        yhat_list.append(yhat)
        ytrue_list.append(ytrue)

    queue.put((rank, np.stack(yhat_list), np.stack(ytrue_list)))


def translate_from_ts_multigpu_jupyter(
        ts,
        max_replicates = 15,
        model_config=None, 
        model_path=None,
        devices=['cuda:0', 'cuda:1', 'cuda:2'],
        num_processes_ts_preproc=30
    ):

    mp.set_start_method('spawn', force=True)
    num_processes = len(devices)
    assert max_replicates % num_processes == 0
    replicates_per_process = max_replicates // num_processes

    src, tgt = prepare_ts_data(
        ts, num_samples=50, B=1225, device='cpu',
        num_processes=num_processes_ts_preproc)

    queue = mp.Manager().Queue()

    model_config.device = 'cpu'
    models = []
    for i in range(num_processes):
        model = load_model(
            config=model_config,
            model_path=model_path,
            device='cpu'
        )
        models.append(model)


    mp.spawn(run_inference_jupyter, args=(models, src, tgt, devices, queue, replicates_per_process), 
             nprocs=num_processes, join=True)

    results = []
    for _ in range(num_processes):
        results.append(queue.get())  # Blocks until an item is available

    results.sort(key=lambda x: x[0]) 
    _, yhat_list, ytrue_list = zip(*results)
    yhat_array = np.concatenate(yhat_list, axis=0)
    ytrue_array = np.concatenate(ytrue_list, axis=0)
    return yhat_array, ytrue_array




def run_inference_multi_gpu_jupyter_multi_ts(rank, models, ts_data_list, devices, queue, num_replicates):
    """Runs inference on a single GPU, processing subsets of replicates from multiple TS files."""
    
    seed = 0xC0FFEE00 + rank  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = devices[rank]
    model = models[rank]

    model.to(device)
    model.cache_to_device(device)

    results = []
    
    for ts_idx, (src, tgt) in enumerate(ts_data_list):
        yhat_list, ytrue_list = [], []
        progress_bar = tqdm(range(num_replicates),
                            desc=f"[GPU {rank}] TS {ts_idx} ⏳ Processing", position=rank, leave=True)

        for _ in progress_bar:
            sequence = generate(model, src.to(device), B=BATCH, device=device)
            yhat, ytrue = post_process(tgt, sequence, TIMES)
            yhat_list.append(yhat)
            ytrue_list.append(ytrue)

        results.append((ts_idx, np.stack(yhat_list), np.stack(ytrue_list)))

    queue.put((rank, results))


def translate_from_multi_ts_multi_gpu(
        ts_list,
        max_replicates=15,
        model_config=None, 
        model_path=None,
        devices=['cuda:0', 'cuda:1', 'cuda:2'],
        num_processes_ts_preproc=30,
        offset=0,
        ignore_target=False,
    ):

    mp.set_start_method('spawn', force=True)
    num_processes = len(devices)
    assert max_replicates % num_processes == 0
    replicates_per_process = max_replicates // num_processes

    queue = mp.Manager().Queue()
    models = []
    model_config.device = 'cpu'
    for i in range(num_processes):
        model = load_model(
            config=model_config,
            model_path=model_path,
            device='cpu'
        )
        models.append(model)

    #ts_data_list = [
    #    prepare_ts_data(
    #        ts,
    #        num_samples=NUM_SAMPLES, B=BATCH,
    #        device='cpu',
    #        num_processes=num_processes_ts_preproc,
    #        offset=offset,
    #        ignore_target=ignore_target
    #        )
    #    for ts in ts_list
    #]
    ts_data_list = prepare_tss_data(
            ts_list,
            num_samples=NUM_SAMPLES, B=BATCH,
            device='cpu',
            num_processes=num_processes_ts_preproc,
            offset=offset,
            ignore_target=ignore_target
        )

    new_ts_data_list = []
    src, tgt = ts_data_list
    for i in range(len(src) // BATCH):
        new_ts_data_list.append((src[i*BATCH:(i+1)*BATCH], tgt[i*BATCH:(i+1)*BATCH]))
    ts_data_list = new_ts_data_list
    

    mp.spawn(run_inference_multi_gpu_jupyter_multi_ts, args=(models, ts_data_list, devices, queue, replicates_per_process), 
             nprocs=num_processes, join=True)

    results = []
    for _ in range(num_processes):
        results.append(queue.get())  # Blocks until an item is available

     # Group results by TS index
    ts_results = defaultdict(list)  # Maps ts_idx → [(yhat_array, ytrue_array)]

    for _, ts_list_results in results:
        for ts_idx, yhat, ytrue in ts_list_results:
            ts_results[ts_idx].append((yhat, ytrue))

    # Ensure ordering matches ts_list and concatenate results
    final_yhat, final_ytrue = [], []

    for ts_idx in sorted(ts_results.keys()):  # Sorting ensures correct TS order
        yhat_list, ytrue_list = zip(*ts_results[ts_idx])  # Extract from tuples
        final_yhat.append(np.concatenate(yhat_list, axis=0))  # Stack along replicate axis
        final_ytrue.append(np.concatenate(ytrue_list, axis=0))

    # Convert to final arrays
    yhat_array = np.concatenate(final_yhat, axis=0)
    ytrue_array = np.concatenate(final_ytrue, axis=0)

    return yhat_array, ytrue_array


