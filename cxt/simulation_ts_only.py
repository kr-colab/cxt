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
import sys
from multiprocessing import get_context

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
    # includes C. elegans chromosomes except X
    for char in item:
        if char.isdigit() or char in ['I', 'II', 'III', 'IV', 'V', ]: # removes X
            if char == 'CM009947.2':
                return False
            else: return True#False
    if item == 'Mt' or item == 'Pt':
        return False
    return False

def random_sample_counts(
    sampling_populations: List[int], num_samples: int = 25, seed: Optional[int] = None
    ) -> Dict[str, int]:
    """
    Randomly distributes `n` samples across the given populations.
    """
    rng = random.Random(seed)
    sampled_counts = {pop.name: 0 for pop in sampling_populations}  
    for pop in rng.choices(sampling_populations, k=num_samples):
        sampled_counts[pop.name] += 1  
    return sampled_counts

def sample_chromosome(species):
    chromosomes = [
        chrom for chrom in species.genome.chromosomes
        if is_any_numeric_or_roman_numeral(chrom.id)
    ]
    chromosome = species.genome.chromosomes[np.random.randint(0, len(chromosomes))]
    while chromosome.id in ['Mt', 'Pt']:
        chromosome = species.genome.chromosomes[np.random.randint(0, len(chromosomes))]
    return chromosome
                

def simulate_random_segment(
    seed,
    num_samples=25,
    segment_length=1e6, 
    species_name="HomSap",
    genetic_map=None,
    population_size=None
):

    np.random.seed(seed)
    seed = np.random.randint(1, 2**32)
    species = stdpopsim.get_species(species_name)

    demographic_models = species.demographic_models
    if len(demographic_models) > 0:
        # Filter out models with non-present sampling points
        excluded_descriptions = {
            'Multi-population model of ancient Eurasia',
            'Out-of-Africa with archaic admixture into Papuans',
            'Multi-population model of ancient Europe'
        }
        valid_models = [model for model in demographic_models 
                        if model.description not in excluded_descriptions]
    else:
        valid_models = [stdpopsim.PiecewiseConstantSize(population_size)]

    demography = np.random.choice(valid_models)
    populations = sampling_populations(demography)
    samples = random_sample_counts(populations, num_samples=num_samples, seed=seed)

    engine = stdpopsim.get_engine("msprime")

    if genetic_map is None:
        chromosome = sample_chromosome(species)
        left = np.random.uniform(chromosome.length - segment_length)
        right = left + segment_length
        contig = species.get_contig(
            chromosome.id, left=left, right=right, 
            mutation_rate=demography.mutation_rate
        )
    else:
        while True:
            chromosome = sample_chromosome(species)
            left  = np.random.uniform(0, chromosome.length - segment_length)
            right = left + segment_length
            try:
                contig = species.get_contig(
                    chromosome.id, left=left, right=right,
                    mutation_rate=demography.mutation_rate, genetic_map=genetic_map
                )
            except ValueError:           # "All intervals are missing data"
                continue                 # try another window
            interior = contig.recombination_map.rate[1:-1]
            if len(interior) == 0 or np.isfinite(interior).all():
                break

    ts = engine.simulate(demography, contig, samples, seed=seed).trim()
    return ts





def dump_one_proc(i, outdir, sim_func):
    ts = sim_func(i)                  # top-level & picklable
    path = os.path.join(outdir, f"ts_{i:08d}.trees")
    ts.dump(path)
    return path

def generate_tree_sequences(num_samples, output_dir, ts_simulation_func, num_processes=8):
    os.makedirs(output_dir, exist_ok=True)
    worker = partial(dump_one_proc, outdir=output_dir, sim_func=ts_simulation_func)
    start_method = "fork"
    with get_context(start_method).Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(worker, range(num_samples)),
                      total=num_samples, desc="Simulating"):
            pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process simulation parameters.')
    parser.add_argument('--num_processes', type=int, default=30, help='Number of processes to use')
    parser.add_argument('--num_samples', type=int, default=2_000_000, help='Number of samples to generate')
    parser.add_argument('--n_individuals', type=int, default=25, help='Number of diploid individuals to simulate')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for saving data')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save data')
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
    args = parser.parse_args()

    num_processes = args.num_processes
    num_samples = args.num_samples
    batch_size = args.batch_size
    data_dir = args.data_dir
    scenario = args.scenario
    n_individuals = args.n_individuals


    if scenario == "constant":
        simulate_parameterized_tree_sequence = partial(simulate_parameterized_tree_sequence, samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=simulate_parameterized_tree_sequence, num_processes=num_processes)


    elif scenario == "sawtooth":
        simulate_parameterized_tree_sequence_sawtooth = partial(simulate_parameterized_tree_sequence, demography=create_sawtooth_demogaphy_object(Ne=20e3, magnitue=3), samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=simulate_parameterized_tree_sequence_sawtooth, num_processes=num_processes)


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
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=simulate_parameterized_tree_sequence_island, num_processes=num_processes)
        


    # mammals
    elif scenario == "stdpopsim_homsap":
        species_name = "HomSap"

        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)


    elif scenario == "stdpopsim_homsap_map":
        species_name = "HomSap"
        genetic_map = 'HapMapII_GRCh38'

        sim_func = partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)



    elif scenario == "stdpopsim_bostau":
        species_name = "BosTau"
        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)


    elif scenario == "stdpopsim_canfam":
        species_name = "CanFam"
        population_size = 13000
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)


    elif scenario == "stdpopsim_canfam_map":
        species_name = "CanFam"
        population_size = 13000
        genetic_map = 'Campbell2016_CanFam3_1'
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, genetic_map=genetic_map, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_pantro":
        species_name = "PanTro"
        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_papanu":
        species_name = "PapAnu"
        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_papanu_map":
        species_name = "PapAnu"
        genetic_map = 'Pyrho_PAnubis1_0'
        sim_func = partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_ponabe":
        species_name = "PonAbe"
        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_ponabe_map":
        species_name = "PonAbe"
        genetic_map = 'NaterPA_PonAbe3'
        sim_func = partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    
    # rest of stdpopsim scenarios
    elif scenario == "stdpopsim_aedaeg":
        species_name = "AedAeg"
        population_size = 1_000_000
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_anapla":
        species_name = "AnaPla"
        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)


    elif scenario == "stdpopsim_anocar":
        species_name = "AnoCar"
        population_size = 3_050_000
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_anogam":
        species_name = "AnoGam"
        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    
    elif scenario == "stdpopsim_apimel":
        species_name = "ApiMel"
        population_size = 200_000
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)


    elif scenario == "stdpopsim_aratha":
        species_name = "AraTha"
        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_aratha_map":
        species_name = "AraTha"
        genetic_map = 'SalomeAveraged_TAIR10'
        sim_func = partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_caeele":
        species_name = "CaeEle"
        population_size = 10000
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_caeele_map":
        species_name = "CaeEle"
        population_size = 10000
        genetic_map = 'RockmanRIAIL_ce11'
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, genetic_map=genetic_map, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_dromel":
        species_name = "DroMel"
        sim_func = partial(simulate_random_segment, species_name=species_name, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_dromel_map":
        species_name = "DroMel"
        genetic_map = 'ComeronCrossoverV2_dm6'
        sim_func = partial(simulate_random_segment, species_name=species_name, genetic_map=genetic_map, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_drosec":
        species_name = "DroSec"
        population_size = 100000
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_gasacu":
        species_name = "GasAcu"
        population_size = 10000
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)


    elif scenario == "stdpopsim_helann":
        species_name = "HelAnn"
        population_size = 673_968
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)

    elif scenario == "stdpopsim_helmel":
        species_name = "HelMel"
        population_size = 2_111_109
        sim_func = partial(simulate_random_segment, species_name=species_name, population_size=population_size, num_samples=n_individuals)
        generate_tree_sequences(
            num_samples=num_samples, output_dir=data_dir,
            ts_simulation_func=sim_func, num_processes=num_processes)



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
                    generate_tree_sequences(
                        num_samples=num_samples, output_dir=save_dir,
                        ts_simulation_func=sim_func, num_processes=num_processes)




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
                        generate_tree_sequences(
                            num_samples=num_samples, output_dir=save_dir,
                            ts_simulation_func=sim_func, num_processes=num_processes)
                        
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

                        generate_tree_sequences(
                            num_samples=num_samples, output_dir=save_dir,
                            ts_simulation_func=sim_func, num_processes=num_processes)



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

                        generate_tree_sequences(
                            num_samples=num_samples, output_dir=save_dir,
                            ts_simulation_func=sim_func, num_processes=num_processes)
    

