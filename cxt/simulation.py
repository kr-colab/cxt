import os
import msprime
import numpy as np
from tqdm import tqdm
from functools import partial
from cxt.utils import xor, xnor
from multiprocessing import Pool
from cxt.utils import ts2X_vectorized
from cxt.utils import simulate_parameterized_tree_sequence, interpolate_tmrcas
import argparse
import random
import stdpopsim
from typing import List, Dict, Optional

import warnings
warnings.filterwarnings("ignore")

def create_sawtooth_demogaphy_object(Ne = 2*10**4, magnitue=4):
    demography = msprime.Demography()
    demography.add_population(initial_size=(Ne))
    demography.add_population_parameters_change(time=20, population=None,
    growth_rate=6437.7516497364/(4*10**4))
    demography.add_population_parameters_change(time=30, growth_rate=-378.691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=200, growth_rate=-643.77516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=300, growth_rate=37.8691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=2000, growth_rate=64.377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=3000, growth_rate=-3.78691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=20000, growth_rate=-6.4377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=30000, growth_rate=0.378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=200000, growth_rate=0.64377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=300000, growth_rate=-0.0378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=2000000, growth_rate=-0.064377516497364/(magnitue*10**4))
    demography.add_population_parameters_change(time=3000000, growth_rate=0.00378691273513906/(magnitue*10**4))
    demography.add_population_parameters_change(time=20000000, growth_rate=0,initial_size=Ne)
    return demography



def random_sample_counts(
    sampling_populations: List[int], num_samples: int = 25, seed: Optional[int] = None
) -> Dict[str, int]:
    """
    Randomly distributes `n` samples across the given populations.

    :param list sampling_populations: Populations to sample from, each with a `.name` attribute.
    :param int num_samples: Number of samples to draw.
    :param int seed: Optional seed for reproducibility.

    :return: Dictionary mapping population names to sample counts.
    :rtype: dict
    """
    rng = random.Random(seed)
    sampled_counts = {pop.name: 0 for pop in sampling_populations}  
    for pop in rng.choices(sampling_populations, k=num_samples):
        sampled_counts[pop.name] += 1  
    return sampled_counts


#def sampling_populations(demography):
#    return [pop for pop in demography.populations if pop.allow_samples]

def sampling_populations(model):
    populations = []
    for pop in model.populations:
        if hasattr(pop, 'default_sampling_time'):
            if isinstance(pop.default_sampling_time, float):
                if pop.default_sampling_time > 0:
                    pass
            elif pop.allow_samples:
                populations.append(pop)
    return populations


def is_any_numeric_or_roman_numeral(item):
    # includes C. elegans chromosomes
    for char in item:
        if char.isdigit() or char in ['I', 'II', 'III', 'IV', 'V']: # removes X
            if char == 'CM009947.2':
                return False
            else: return True#False
    if item == 'Mt' or item == 'Pt':
        return False
    return False
                

def simulate_random_segment(
    seed, num_samples=25, segment_length=1e6, species_name="HomSap", genetic_map=None, population_size=None
):
    """Simulates a random human genomic segment using stdpopsim and msprime."""

    np.random.seed(seed)
    seed = np.random.randint(1, 2**32)

    species = stdpopsim.get_species(species_name)
    chromosomes = [chrom for chrom in species.genome.chromosomes if is_any_numeric_or_roman_numeral(chrom.id)]

    # good enough for now, misses last two chromosomes of rice
    chromosome = species.genome.chromosomes[np.random.randint(0, len(chromosomes))]
    while chromosome.id in ['Mt', 'Pt']:
        chromosome = species.genome.chromosomes[np.random.randint(0, len(chromosomes))]

    left = np.random.uniform(chromosome.length - segment_length)
    right = left + segment_length

    demographic_models = species.demographic_models
    if len(demographic_models) > 0:
        valid_models = [
            model for model in species.demographic_models 
            # excluded due to sampling points which are not in the present
            if model.description not in {
                'Multi-population model of ancient Eurasia',
                'Out-of-Africa with archaic admixture into Papuans',
                'Multi-population model of ancient Europe'
            }
        ]
    else:
        if population_size is None:
            valid_models = [stdpopsim.PiecewiseConstantSize(np.random.choice([10_000, 20_000, 40_000]))]
        else:
            valid_models = [stdpopsim.PiecewiseConstantSize(population_size)]


    demography = np.random.choice(valid_models)
    populations = sampling_populations(demography)
    samples = random_sample_counts(populations, num_samples=num_samples, seed=seed)
    
    if genetic_map is not None:
        whole_contig = species.get_contig(
            chromosome.id, mutation_rate=demography.mutation_rate, genetic_map=genetic_map
        )
        
        # valid genetic map region sampling
        while True:
            
            left = np.random.uniform(0, chromosome.length - segment_length)
            right = left + segment_length
            left_mask = whole_contig.recombination_map.left >= left
            right_mask = whole_contig.recombination_map.right < right
            #print("Sampling valid genetic map region...")
            #print(f"Chromosome: {chromosome.id}, left: {left}, right: {right}")
            if not left_mask.any() or not right_mask.any():
                continue  
            contig = species.get_contig(
                chromosome.id, left=left, right=right, 
                mutation_rate=demography.mutation_rate, genetic_map=genetic_map)
            if not np.isnan(contig.recombination_map.rate).any():
                break  

        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(demography, contig, samples, seed=seed)

    else:
        engine = stdpopsim.get_engine("msprime")
        contig = species.get_contig(
            #chromosome.synonyms[0], left=left, right=right, 
            chromosome.id, left=left, right=right, 
            mutation_rate=demography.mutation_rate
        )
        ts = engine.simulate(demography, contig, samples, seed=seed).trim()

        
        #ts = rescale_sequence(ts.keep_intervals([[left, right]]).simplify(), left).trim()
        #print(ts.num_trees, ts.num_sites, ts.sequence_length, ts.num_samples)

    return ts


def generate_data(args) -> tuple:
    """
    Generate data for simulation.
    Parameters
    ----------
    i : int
        A seed integer parameter used by the ts_simulation_func.
    pivot_A : int
        The first pivot index for processing, by default 0.
    pivot_B : int
        The second pivot index for processing, by default 1.
    ts_simulation_func : callable
        A function to simulate tree sequence data, by default None.
    Returns
    -------
    tuple
        A tuple containing:
        - X : np.ndarray
            A 3D array with processed data using XOR and XNOR operations.
        - y : np.ndarray
            A 1D array with log-interpolated TMRCA values.
    """
    i, pivot_A, pivot_B, ts_simulation_func, randomize_pivots, n_individuals = args
    ts = ts_simulation_func(i)
    #print(i)
    
    if randomize_pivots:
        pivots = np.arange(0, int(2*n_individuals))
        np.random.shuffle(pivots)
        pivot_A = pivots[0]
        pivot_B = pivots[1]

    # processing
    #print(ts.num_trees, ts.num_sites)
    Xxor = ts2X_vectorized(ts, window_size=2000, xor_ops=xor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    Xxnor = ts2X_vectorized(ts, window_size=2000, xor_ops=xnor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    X = np.stack([Xxor, Xxnor], axis=0).astype(np.float16)
        
    y = np.log(interpolate_tmrcas(ts.simplify(samples=[pivot_A, pivot_B]), window_size=2000)).astype(np.float16)
    return X, y

def save_batch(data, idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(f'{output_dir}/X_{idx}.npy', np.stack([X for X, _ in data]))
    np.save(f'{output_dir}/y_{idx}.npy', np.stack([y for _, y in data]))

def process_batches(num_samples: int, start_batch: int,
                    batch_size: int, num_processes: int,
                    output_dir: str, pivot_A: int, pivot_B: int,
                    ts_simulation_func: callable, randomize_pivots: bool, n_individuals: int) -> None:
    """
    Process data in batches using multiprocessing and save the results.

    Parameters
    ----------
    num_samples : int
        Total number of samples to process.
    start_batch : int
        The starting batch index.
    batch_size : int
        The number of samples per batch.
    num_processes : int
        The number of processes to use for multiprocessing.
    output_dir : str
        The directory where the output batches will be saved.
    pivot_A : int, optional
            The first pivot index for processing, by default 0.
    pivot_B : int, optional
            The second pivot index for processing, by default 1.
    ts_simulation_func : callable
        The simulation function to generate tree sequence data.
    randomize_pivots : bool
        Randomize pivot indices, by default False.

    Returns
    -------
    None
    """
    with Pool(num_processes) as pool:
        batch = []
        batch_idx = 0 + start_batch
        start_sample = batch_idx * batch_size
        args = []
        for i in range(start_sample, num_samples):
            args.append((i, pivot_A, pivot_B, ts_simulation_func, randomize_pivots, n_individuals))

        for i, result in enumerate(tqdm(pool.imap(generate_data, args), total=num_samples-start_sample)):
            batch.append(result)
            #print(len(result))
            if len(batch) == batch_size or i == num_samples - 1:
                save_batch(batch, batch_idx, output_dir)
                batch = []
                batch_idx += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process simulation parameters.')
    parser.add_argument('--num_processes', type=int, default=30, help='Number of processes to use')
    parser.add_argument('--num_samples', type=int, default=2_000_000, help='Number of samples to generate')
    parser.add_argument('--n_individuals', type=int, default=25, help='Number of diploid individuals to simulate')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for saving data')
    parser.add_argument('--start_batch', type=int, default=0, help='Starting batch index')
    parser.add_argument('--pivot_A', type=int, default=0, help='Pivot A index')
    parser.add_argument('--pivot_B', type=int, default=1, help='Pivot B index')
    parser.add_argument('--data_dir', type=str, default='/sietch_colab/kkor/base_dataset', help='Directory to save data')
    parser.add_argument('--scenario', type=str, choices=[
        'constant', 'sawtooth','stdpopsim_homsap', 'stdpopsim_homsap_map',
          'stdpopsim_bostau', 'stdpopsim_canfam', 'stdpopsim_canfam_map', 'stdpopsim_pantro',
            'stdpopsim_papanu', 'stdpopsim_papanu_map', 'stdpopsim_ponabe', 'stdpopsim_ponabe_map','stdpopsim_aedaeg',
            'stdpopsim_anapla', 'stdpopsim_anocar','stdpopsim_anogam', 'stdpopsim_apimel', 'stdpopsim_aratha', 'stdpopsim_aratha_map',
            'stdpopsim_caeele', 'stdpopsim_caeele_map', 'stdpopsim_dromel', 'stdpopsim_dromel_map', 'stdpopsim_drosec',
            'stdpopsim_gasacu', 'stdpopsim_helann', 'stdpopsim_helmel',
            'stdpopsim_musmus','stdpopsim_ratnor','stdpopsim_gorgor', 'stdpopsim_orysat','stdpopsim_susscr','stdpopsim_phosin',
              'island','llm_ne_constant','llm_ne_sawtooth','llm_island_3pop','llm_island_5pop','llm_hard_sweeps','random', 'llm_ne_constant_2', 'llm_ne_constant_3','llm_ne_constant_4','llm_ne_constant_5',
    ], default='constant', help='Scenario type')
    parser.add_argument('--randomize_pivots', default=False, help='Randomize pivot indices')
    args = parser.parse_args()

    num_processes = args.num_processes
    num_samples = args.num_samples
    batch_size = args.batch_size
    start_batch = args.start_batch
    pivot_A = args.pivot_A
    pivot_B = args.pivot_B
    data_dir = args.data_dir
    scenario = args.scenario
    randomize_pivots = args.randomize_pivots
    n_individuals = args.n_individuals


    if scenario == "constant":
        simulate_parameterized_tree_sequence = partial(simulate_parameterized_tree_sequence, samples=n_individuals)
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B, simulate_parameterized_tree_sequence, randomize_pivots, n_individuals)

    elif scenario == "sawtooth":
        simulate_parameterized_tree_sequence_sawtooth = partial(simulate_parameterized_tree_sequence, demography=create_sawtooth_demogaphy_object(Ne=20e3, magnitue=3), samples=n_individuals)
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B, simulate_parameterized_tree_sequence_sawtooth, randomize_pivots, n_individuals)

    elif scenario == "island":
        if n_individuals == 5:
            samples = {0: 3, 1: 1, 2: 1}
        elif n_individuals == 25:
            samples = {0: 15, 1: 5, 2: 5}
        else:
            samples = {0: int(n_individuals*0.6), 1: int(n_individuals*0.2), 2: int(n_individuals*0.2)}
            total_samples = sum(samples.values())
            if total_samples != n_individuals:
                diff = n_individuals - total_samples
                samples[0] += diff  # Adjust the first population to match n_individuals
        island_demography = msprime.Demography.island_model([10000, 5000, 5000], migration_rate=0.1)
        simulate_parameterized_tree_sequence_island = partial(simulate_parameterized_tree_sequence, island_demography=island_demography, samples=samples)
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B, simulate_parameterized_tree_sequence_island, randomize_pivots, n_individuals)

    # not used
    elif scenario == "random":
        random_scenario = np.random.choice(np.arange(1, 66))
        simulate_parameterized_tree_sequence_random = partial(simulate_parameterized_tree_sequence, random_scenario=random_scenario, samples=n_individuals)
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B, simulate_parameterized_tree_sequence_random, randomize_pivots, n_individuals)



    # mammals
    elif scenario == "stdpopsim_homsap":
        species_name = "HomSap"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_homsap_map":
        species_name = "HomSap"
        genetic_map = 'HapMapII_GRCh38'
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_bostau":
        species_name = "BosTau"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_canfam":
        species_name = "CanFam"
        population_size = 13000
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_canfam_map":
        species_name = "CanFam"
        population_size = 13000
        genetic_map = 'Campbell2016_CanFam3_1'
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_pantro":
        species_name = "PanTro"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_papanu":
        species_name = "PapAnu"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_papanu_map":
        species_name = "PapAnu"
        genetic_map = 'Pyrho_PAnubis1_0'
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_ponabe":
        species_name = "PonAbe"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_ponabe_map":
        species_name = "PonAbe"
        genetic_map = 'NaterPA_PonAbe3'
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals), randomize_pivots, n_individuals)
    
    # rest of stdpopsim scenarios
    elif scenario == "stdpopsim_aedaeg":
        species_name = "AedAeg"
        population_size = 1_000_000
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)


    elif scenario == "stdpopsim_anapla":
        species_name = "AnaPla"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                            partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_anocar":
        species_name = "AnoCar"
        population_size = 3_050_000
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_anogam":
        species_name = "AnoGam"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                            partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)
    
    elif scenario == "stdpopsim_apimel":
        species_name = "ApiMel"
        population_size = 200_000
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_aratha":
        species_name = "AraTha"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)
        
    elif scenario == "stdpopsim_aratha_map":
        species_name = "AraTha"
        genetic_map = 'SalomeAveraged_TAIR10'
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                            partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals), randomize_pivots, n_individuals)
        
    elif scenario == "stdpopsim_caeele":
        species_name = "CaeEle"
        population_size = 10000
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_caeele_map":
        species_name = "CaeEle"
        population_size = 10000
        genetic_map = 'RockmanRIAIL_ce11'
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, genetic_map=genetic_map, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_dromel":
        species_name = "DroMel"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)
        
    elif scenario == "stdpopsim_dromel_map":
        species_name = "DroMel"
        genetic_map = 'ComeronCrossoverV2_dm6'
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals), randomize_pivots, n_individuals)
        
    elif scenario == "stdpopsim_drosec":
        species_name = "DroSec"
        population_size = 100000
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_gasacu":
        species_name = "GasAcu"
        population_size = 10000
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_helann":
        species_name = "HelAnn"
        population_size = 673_968
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)


    elif scenario == "stdpopsim_helmel":
        species_name = "HelMel"
        population_size = 2_111_109
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)


    # stdpopsim update
    elif scenario == "stdpopsim_musmus":
        species_name = "MusMus"
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)
        
    elif scenario == "stdpopsim_ratnor":
        species_name = "RatNor"
        population_size = 1.24e5
        process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                         partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_gorgor":
            species_name = "GorGor"
            process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                            partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_orysat":
            species_name = "OrySat"
            process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                            partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)

    elif scenario == "stdpopsim_susscr":
            species_name = "SusScr"
            population_size = 270_000
            process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                            partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals), randomize_pivots, n_individuals)


    elif scenario == "stdpopsim_phosin":
            species_name = "PhoSin"
            process_batches(num_samples, start_batch, batch_size, num_processes, data_dir, pivot_A, pivot_B,
                            partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals), randomize_pivots, n_individuals)



    # broad dataset v0
    
    elif scenario == "llm_ne_constant":
        for population_size in [1e4, 2e4, 4e4]: # 1e4, 2e4, 4e4, 8e4
            for mutation_rate in [1e-8, 5e-8]:
                for recombination_rate in [1e-8, 5e-8]:
                    if mutation_rate == 5e-8 and recombination_rate == 5e-8:
                        continue
                    sim_func = partial(simulate_parameterized_tree_sequence, population_size=population_size, mutation_rate=mutation_rate, recombination_rate=recombination_rate, samples=n_individuals)
                    sub_data_dir = f"ne_constant_{population_size:.0e}_{mutation_rate:.1e}_{recombination_rate:.1e}"
                    save_dir = f"{data_dir}/{sub_data_dir}"
                    process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)

    # more constant data below (not used in the end)
    elif scenario == "llm_ne_constant_2":
       for population_size in [0.25e4, 0.5e4, 5e4,6e4,7e4, 8e4,9e4, 10e4, 16e4, 20e4, 32e4, 48e4, 64e4]: 
            for mutation_rate in [1e-8]:
                for recombination_rate in [1e-8]:
                    sim_func = partial(simulate_parameterized_tree_sequence, population_size=population_size, mutation_rate=mutation_rate, recombination_rate=recombination_rate, samples=n_individuals)
                    sub_data_dir = f"ne_constant_{population_size:.4e}_{mutation_rate:.1e}_{recombination_rate:.1e}" # popsize precision to .4e from .0e for few sizes
                    save_dir = f"{data_dir}/{sub_data_dir}"
                    process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)

    elif scenario == "llm_ne_constant_3":
       for population_size in [10e4, 15e4, 20e4, 25e4, 30e4, 35e4, 40e4, 45e4, 50e4]: 
            for mutation_rate in [1e-9]:
                for recombination_rate in [1e-9]:
                    sim_func = partial(simulate_parameterized_tree_sequence, population_size=population_size, mutation_rate=mutation_rate, recombination_rate=recombination_rate, samples=n_individuals)
                    sub_data_dir = f"ne_constant_{population_size:.4e}_{mutation_rate:.1e}_{recombination_rate:.1e}" # popsize precision to .4e from .0e for few sizes
                    save_dir = f"{data_dir}/{sub_data_dir}"
                    process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)

    elif scenario == "llm_ne_constant_4":
       for population_size in [10e4, 15e4]: 
            for mutation_rate in [5e-9]:
                for recombination_rate in [1e-9]:
                    sim_func = partial(simulate_parameterized_tree_sequence, population_size=population_size, mutation_rate=mutation_rate, recombination_rate=recombination_rate, samples=n_individuals)
                    sub_data_dir = f"ne_constant_{population_size:.4e}_{mutation_rate:.1e}_{recombination_rate:.1e}" # popsize precision to .4e from .0e for few sizes
                    save_dir = f"{data_dir}/{sub_data_dir}"
                    process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)


    elif scenario == "llm_ne_constant_5":
       for population_size in [1e6, 1.5e6]: 
            for mutation_rate in [1e-9, 5e-9]:
                for recombination_rate in [1e-9]:
                    sim_func = partial(simulate_parameterized_tree_sequence, population_size=population_size, mutation_rate=mutation_rate, recombination_rate=recombination_rate, samples=n_individuals)
                    sub_data_dir = f"ne_constant_{population_size:.4e}_{mutation_rate:.1e}_{recombination_rate:.1e}" # popsize precision to .4e from .0e for few sizes
                    save_dir = f"{data_dir}/{sub_data_dir}"
                    process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)



    elif scenario ==  "llm_ne_sawtooth":
        for magnitude in [3, 4, 5]:
            for population_size in [1e4, 2e4, 4e4]:
                for mutation_rate in [1e-8, 5e-8]:
                    for recombination_rate in [1e-8, 5e-8]:
                        if mutation_rate == 5e-8 and recombination_rate == 5e-8:
                            continue
                        sim_func = partial(simulate_parameterized_tree_sequence, demography=create_sawtooth_demogaphy_object(Ne=population_size,magnitue=magnitude), mutation_rate=mutation_rate, recombination_rate=recombination_rate, samples=n_individuals)
                        
                        sub_data_dir = f"ne_sawtooth_{magnitude}_{population_size:.0e}_{mutation_rate:.1e}_{recombination_rate:.1e}"
                        save_dir = f"{data_dir}/{sub_data_dir}"
                        process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)

    elif scenario == "llm_island_3pop":
        for migration_rate in [0.05, 0.2]:
            for population_size in [1e4, 2e4, 4e4]: # 1e4, 2e4, 4e4
                for mutation_rate in [1e-8, 5e-8]:
                    for recombination_rate in [1e-8, 5e-8]:
                        if mutation_rate == 5e-8 and recombination_rate == 5e-8:
                            continue
                        if n_individuals == 5:
                            samples = {0: 3, 1: 1, 2: 1}
                        elif n_individuals == 25:
                            samples = {0: 15, 1: 5, 2: 5}
                        else:
                            samples = {0: int(n_individuals*0.6), 1: int(n_individuals*0.2), 2: int(n_individuals*0.2)}
                            total_samples = sum(samples.values())
                            if total_samples != n_individuals:
                                diff = n_individuals - total_samples
                                samples[0] += diff  # Adjust the first population to match n_individuals
                            
                        island_demography = msprime.Demography.island_model([population_size, population_size/2, population_size/2], migration_rate=migration_rate)
                        sim_func = partial(simulate_parameterized_tree_sequence, island_demography=island_demography, samples=samples, mutation_rate=mutation_rate, recombination_rate=recombination_rate)
                        sub_data_dir = f"island_3pop_{migration_rate}_{population_size:.0e}_{mutation_rate:.1e}_{recombination_rate:.1e}"
                        save_dir = f"{data_dir}/{sub_data_dir}"
                        process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)

    
    # not this one, takes too long
    elif scenario == "llm_island_5pop":
        print("Starting 5-population island model simulations...")
        if n_individuals != 25:
            print("Skipping llm_island_5pop scenario as n_individuals is not 25.")
            exit(1)
        for migration_rate in [0.05, 0.2]:
            for population_size in [1e4, 2e4, 4e4]:
                for mutation_rate in [1e-8, 5e-8]:
                    for recombination_rate in [1e-8, 5e-8]:
                        if mutation_rate == 5e-8 and recombination_rate == 5e-8:
                            continue
                        samples = {0: 5, 1: 5, 2: 5, 3: 5, 4: 5}
                        island_demography = msprime.Demography.island_model([population_size, population_size/4, population_size/4, population_size/4, population_size/4], migration_rate=migration_rate)
                        sim_func = partial(simulate_parameterized_tree_sequence, island_demography=island_demography, samples=samples, mutation_rate=mutation_rate, recombination_rate=recombination_rate)
                        sub_data_dir = f"island_5pop_{migration_rate}_{population_size:.0e}_{mutation_rate:.1e}_{recombination_rate:.1e}"
                        save_dir = f"{data_dir}/{sub_data_dir}"
                        process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)
    

    elif scenario == "llm_hard_sweeps":
        np.random.seed(42)
        for population_size in [1e4, 2e4, 4e4]:
            for mutation_rate in [1e-8, 5e-8]:
                for recombination_rate in [1e-8, 5e-8]:
                    for selection_coefficient in [0.01, 0.1, 1]:
                        selection_position = np.random.choice([0.25e6, 0.5e6, 0.75e6])
                        if mutation_rate == 5e-8 and recombination_rate == 5e-8:
                            continue
                        sim_func = partial(
                            simulate_parameterized_tree_sequence, population_size=population_size,
                            mutation_rate=mutation_rate, recombination_rate=recombination_rate,
                            hard_sweep=True, selection_coefficient=selection_coefficient, selection_position=selection_position, samples=n_individuals)
                        sub_data_dir = f"hard_sweeps_{selection_coefficient}_{selection_position:.1e}_{population_size:.0e}_{mutation_rate:.1e}_{recombination_rate:.1e}"
                        save_dir = f"{data_dir}/{sub_data_dir}"
                        process_batches(num_samples, start_batch, batch_size, num_processes, save_dir, pivot_A, pivot_B, sim_func, randomize_pivots, n_individuals)
    

