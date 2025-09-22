import numpy as np
import os
import stdpopsim
import pickle
import ray
import argparse
import matplotlib.pyplot as plt
import torch
import pickle

from matplotlib.colors import Normalize
from torch import Tensor
from cxt.config import BroadModelConfig 
import cxt.utils as utils
import cxt.inference as inference

TokenFreeDecoderConfig = BroadModelConfig

# --- lib

def stochastic_bias_correction(
    mutation_rate: np.ndarray,
    sequence_length: np.ndarray,
    mutation_count: np.ndarray,
    predictions: np.ndarray,
    rng: np.random.Generator = None,
) -> (np.ndarray, np.ndarray):
    r"""
    Correct the predicted TMRCAs such that expected diversity matches
    observed diversity, for a given mutation rate. This is done stochastically,
    by using the fact that under the model,

        mutation_count ~ Poisson(2 * correction * mu * \sum_i TMRCA_i * window_size_i)
    
    the posterior (given improper constant prior) is,

        correction ~ Gamma(mutation_count + 1, 2 * mu * \sum_i TMRCA_i * window_size_i)

    and sampling accordingly (e.g. iid for each TMRCA sample, pivot pair).
    
    The input predictions are assumed to have dimensions 
    `(replicates, pairs, windows)`.
    """
    assert predictions.ndim == 3
    assert mutation_count.ndim == 1
    assert sequence_length.ndim == 1
    assert mutation_rate.ndim == 1
    assert mutation_count.size == predictions.shape[1]
    assert sequence_length.size == predictions.shape[1]
    assert mutation_rate.size == predictions.shape[1]
    if rng is None: rng = np.random.default_rng()
    corrected = []
    for log_tmrca in predictions:
        rate = 2 * np.exp(log_tmrca).mean(axis=-1) * \
            mutation_rate * sequence_length
        correction = rng.gamma(shape=mutation_count + 1, scale=1 / rate)
        corrected.append(log_tmrca + np.log(correction)[:, np.newaxis])
    corrected = np.stack(corrected)
    return corrected


# --- impl

parser = argparse.ArgumentParser(
    "Calculate CI coverage from stochastic cxt predictions along a grid of "
    "window positions. The goal is to see if uncertainty estimates are "
    "well-calibrated."
)
parser.add_argument(
    "--random-seed", help="Global random seed", 
    type=int, default=1,
)
parser.add_argument(
    "--num-cpus", help="Number of parallel processes", 
    type=int, default=50,
)
parser.add_argument(
    "--output-path", help="Where to save plots", type=str, 
    default="/sietch_colab/data_share/cxt/experiments/coverage-rice/",
)
parser.add_argument(
    "--model-path", help="Lightning checkpoint to load model from", type=str,
    default="/sietch_colab/data_share/cxt/broad_model/epoch=0-step=11296.ckpt",
)
parser.add_argument(
    "--overwrite-cache", action="store_true",
    help="Overwrite saved results",
)
parser.add_argument(
    "--device", help="Device to use for tensor computations", type=str,
    default="cuda:0",
)
args = parser.parse_args()

if not os.path.exists(args.output_path): 
    os.makedirs(args.output_path)

demography = args.demography
rng = np.random.default_rng(args.random_seed)
ray.init(num_cpus=args.num_cpus)
torch.manual_seed(1024)

# get contig from which to base parameter values around
num_diploids = 25
species = stdpopsim.get_species("OrySat")
demogr = species.get_demographic_model("BottleneckMigration_3C07")
base_contig = species.get_contig("chr1")
mut_rate = base_contig.mutation_rate
rec_rate = base_contig.recombination_map.mean_rate
base_m = mut_rate
base_r = rec_rate
windows = np.linspace(0, 1e6, 501)


@ray.remote
def simulate_parallel(seed, mut_rate, rec_rate):
    contig = species.get_contig(length=1e6, mutation_rate=mut_rate, recombination_rate=rec_rate)
    sample = {demogr.populations[0].name: num_diploids}
    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(
        contig=contig, 
        samples=sample, 
        demographic_model=demogr, 
        seed=seed,
    )
    ts_pivot = ts.simplify(samples=[0, 1])
    div = ts_pivot.diversity(mode='site', span_normalise=False)
    tmr = ts_pivot.divergence(
        sample_sets=[[0],[1]],
        mode='branch', 
        windows=windows, 
        span_normalise=True,
    ) / 2
    obs = ts_pivot.divergence(
        sample_sets=[[0],[1]],
        mode='site', 
        windows=windows, 
        span_normalise=True,
    ) / 2
    sln = ts_pivot.sequence_length
    args = (ts, 0, 1, 0.0, False)   # (TreeSequence, pivot_A, pivot_B, offset, ignore_target)
    observed, target = utils.process_pair(args) 
    return observed, target, np.array([div, sln, *tmr, *obs])


model_path = args.model_path
device = args.device
config = TokenFreeDecoderConfig(device=device)
model = inference.load_model(
    config=config,
    model_path=model_path,
    device=device,
)

num_pairs = 100
batch_size = 10
num_batch = 10
num_epoch = 10

results = {}
for m in [0.5, 1, 2]:
    cache_file = f"{args.output_path}/cache.{str(m)}.pkl"
    stats_file = f"{args.output_path}/stats.{str(m)}.pkl"
    if not os.path.exists(cache_file) or not os.path.exists(stats_file) or args.overwrite_cache:
        ycorr_store = []
        ytrue_store = []
        param_store = []
        exp_div_store = []
        obs_div_store = []
        for epoch in range(num_epoch):
            print(f"Epoch {epoch}", flush=True)
            seed_array = rng.integers(2 ** 32 - 1, size=num_pairs)
            job_list = [simulate_parallel.remote(s, base_m * m, base_r) for s in seed_array]
            src, tgt, stats = zip(*ray.get(job_list))
            pair_id = np.repeat(np.arange(num_pairs), batch_size)

            # repeat each pair some number of times
            src = np.stack(src)[pair_id]
            tgt = np.stack(tgt)[pair_id]
            src = Tensor(src).to(device)
            tgt = Tensor(tgt).to(torch.int32)

            stats = np.stack(stats)
            mut_count, seq_length = stats[:, 0], stats[:, 1]
            exp_div = stats[:, 2:502]
            obs_div = stats[:, 502:1002]
            mut_rate = np.repeat(base_m * m, mut_count.size)
            exp_div_store.append(exp_div)
            obs_div_store.append(obs_div)
            
            # now for each dataset, generate samples for confidence intervals
            ypred_store = []
            for rep in range(num_batch):
                print(f"Running sample {rep}", flush=True)
                sequence = inference.generate(model, src, B=src.shape[0], device=device)
                ypred, ytrue = inference.post_process(tgt, sequence, utils.TIMES)
                ypred = ypred.reshape(num_pairs, batch_size, -1).transpose(1, 0, 2) # to [rep, pair, window]
                ytrue = ytrue.reshape(num_pairs, batch_size, -1)[:, 0, :]
                ypred_store.append(ypred)
            ycorr = stochastic_bias_correction(mut_rate, seq_length, mut_count, np.concatenate(ypred_store, axis=0), rng)
            ycorr = ycorr.transpose(1, 0, 2) # back to [pair, rep, window]
            ycorr_store.append(ycorr)
            ytrue_store.append(ytrue)
            param_store.append(mut_rate)

        # save stuff
        ycorr_store = np.concatenate(ycorr_store, axis=0)
        ytrue_store = np.concatenate(ytrue_store, axis=0)
        param_store = np.concatenate(param_store, axis=0)
        exp_div_store = np.concatenate(exp_div_store, axis=0)
        obs_div_store = np.concatenate(obs_div_store, axis=0)
        cache = {"ycorr_store": ycorr_store, "ytrue_store": ytrue_store, "param_store": param_store}
        stats = {"exp_div": exp_div_store, "obs_div": obs_div_store}
        pickle.dump(stats, open(stats_file, "wb"))
        pickle.dump(cache, open(cache_file, "wb"))
    else:
        cache = pickle.load(open(cache_file, "rb"))
        stats = pickle.load(open(stats_file, "rb"))
    results[m] = (cache, stats)



