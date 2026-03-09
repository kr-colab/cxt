Simulation
==========

All training data for cxt is generated using ``python cxt/simulation_ts_only.py``,
which consolidates every simulation scenario into a single CLI. This page
documents the exact commands that produce the training data used in the paper.

The simulation script generates tree sequences using ``msprime`` and
``stdpopsim``, saving them as ``.trees`` files organized by scenario. These
tree sequences are then preprocessed into training pairs (see
:doc:`preprocessing`).

.. note::

   The simulation CLI is ``cxt/simulation_ts_only.py`` (not a Python module
   invocation). Core simulation functions such as
   ``simulate_parameterized_tree_sequence`` and ``create_sawtooth_demography``
   live in ``cxt.utils``.


Overview
--------

The training data consists of four categories:

1. **Base dataset** -- constant :math:`N_e`, sawtooth demography, island model
2. **LLM-style datasets** -- parameter sweeps over :math:`N_e`, mutation rate,
   recombination rate, and selection
3. **stdpopsim mammals** -- realistic demographic models for great apes and
   cattle, with and without genetic maps
4. **stdpopsim other species** -- 15+ additional species from the stdpopsim
   catalog

Each simulation produces 1 Mb tree sequences with 25 diploid individuals
(50 haploid samples) by default.


CLI reference
-------------

.. code-block:: text

   python cxt/simulation_ts_only.py \
       --scenario <scenario_name> \
       --data_dir <output_directory> \
       --num_samples <n_simulations> \
       --num_processes <n_parallel> \
       [--n_individuals 25] \
       [--batch_size 1000]

Key arguments:

- ``--scenario``: simulation scenario (see below)
- ``--data_dir``: output directory for ``.trees`` files
- ``--num_samples``: total number of simulations to generate
- ``--num_processes``: parallel worker count
- ``--n_individuals``: diploid sample count per simulation (default: 25)
- ``--batch_size``: number of simulations per batch (default: 1000)


Full pipeline (paper)
---------------------

The following script reproduces the complete training dataset. Adjust
``DATA_DIR`` to point to your storage location.

.. code-block:: bash

   #!/usr/bin/env bash
   set -euo pipefail

   DATA_DIR=/path/to/training_data
   SIM="python cxt/simulation_ts_only.py"

   mkdir -p ${DATA_DIR}
   mkdir -p ${DATA_DIR}/llm
   mkdir -p ${DATA_DIR}/stdpopsim/v0.2

   # ================================================================
   # 1. Base dataset
   # ================================================================

   # Constant Ne (10,000 simulations)
   ${SIM} --num_processes 50 --num_samples 10000 \
       --data_dir ${DATA_DIR}/base_dataset --scenario constant

   # Sawtooth demography (1,000 simulations)
   ${SIM} --num_processes 30 --num_samples 1000 \
       --data_dir ${DATA_DIR}/ssd --scenario sawtooth

   # Island model (1,000 simulations)
   ${SIM} --num_processes 30 --num_samples 1000 \
       --data_dir ${DATA_DIR}/idd --scenario island

   # ================================================================
   # 2. LLM-style datasets (parameter sweeps)
   # ================================================================

   ${SIM} --num_processes 100 --num_samples 125 \
       --data_dir ${DATA_DIR}/llm --scenario llm_ne_sawtooth

   ${SIM} --num_processes 100 --num_samples 50 \
       --data_dir ${DATA_DIR}/llm --scenario llm_hard_sweeps

   ${SIM} --num_processes 75 --num_samples 50 \
       --data_dir ${DATA_DIR}/llm --scenario llm_island_3pop

   ${SIM} --num_processes 100 --num_samples 500 \
       --data_dir ${DATA_DIR}/llm --scenario llm_ne_constant

   # ================================================================
   # 3. stdpopsim mammals (great apes, cattle)
   # ================================================================

   for scenario in stdpopsim_homsap stdpopsim_homsap_map \
       stdpopsim_bostau stdpopsim_canfam stdpopsim_canfam_map \
       stdpopsim_pantro stdpopsim_papanu stdpopsim_papanu_map \
       stdpopsim_ponabe stdpopsim_ponabe_map; do
       ${SIM} --num_processes 75 --num_samples 1000 \
           --data_dir ${DATA_DIR}/stdpopsim/v0.2/${scenario} \
           --scenario ${scenario}
   done

   # ================================================================
   # 4. stdpopsim other species
   # ================================================================

   ${SIM} --num_processes 100 --num_samples 300 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_aedaeg --scenario stdpopsim_aedaeg
   ${SIM} --num_processes 100 --num_samples 25 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_anapla --scenario stdpopsim_anapla
   ${SIM} --num_processes 100 --num_samples 5 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_anocar --scenario stdpopsim_anocar
   ${SIM} --num_processes 100 --num_samples 100 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_anogam --scenario stdpopsim_anogam
   ${SIM} --num_processes 100 --num_samples 5 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_apimel --scenario stdpopsim_apimel
   ${SIM} --num_processes 100 --num_samples 500 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_aratha --scenario stdpopsim_aratha
   ${SIM} --num_processes 100 --num_samples 500 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_aratha_map --scenario stdpopsim_aratha_map
   ${SIM} --num_processes 100 --num_samples 1000 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_caeele --scenario stdpopsim_caeele
   ${SIM} --num_processes 100 --num_samples 1000 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_caeele_map --scenario stdpopsim_caeele_map
   ${SIM} --num_processes 100 --num_samples 5 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_dromel --scenario stdpopsim_dromel
   ${SIM} --num_processes 100 --num_samples 300 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_drosec --scenario stdpopsim_drosec
   ${SIM} --num_processes 100 --num_samples 1000 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_gasacu --scenario stdpopsim_gasacu
   ${SIM} --num_processes 100 --num_samples 300 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_helann --scenario stdpopsim_helann
   ${SIM} --num_processes 100 --num_samples 5 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_helmel --scenario stdpopsim_helmel
   ${SIM} --num_processes 100 --num_samples 1000 \
       --data_dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_musmus --scenario stdpopsim_musmus


Available scenarios
-------------------

Run ``python cxt/simulation_ts_only.py --help`` for the full list. The main
categories are:

**Parametric models** (``msprime``):

- ``constant`` -- constant :math:`N_e = 20{,}000`
- ``sawtooth`` -- oscillating :math:`N_e` (Schiffels & Durbin 2014 zigzag)
- ``island`` -- 3-population island model with migration
- ``random`` -- 64 pre-drawn demographic trajectories with random
  :math:`\mu` and :math:`r`

**LLM-style parameter sweeps**:

- ``llm_ne_constant`` -- 3 :math:`N_e` × 2 :math:`\mu` × 2 :math:`r`
- ``llm_ne_sawtooth`` -- 3 magnitudes × 3 :math:`N_e` × 2 :math:`\mu` ×
  2 :math:`r`
- ``llm_island_3pop`` -- 2 migration rates × 3 :math:`N_e` × 2 :math:`\mu`
  × 2 :math:`r`
- ``llm_hard_sweeps`` -- 3 :math:`N_e` × 2 :math:`\mu` × 2 :math:`r` × 3
  selection coefficients

**stdpopsim species** (``stdpopsim``):

Simulations use species-specific demographic models from the ``stdpopsim``
catalog, sampling random chromosomal segments. Suffixes ``_map`` indicate
use of a species-specific genetic map.

.. list-table::
   :header-rows: 1
   :widths: 25 40 15

   * - Scenario
     - Species
     - Samples
   * - ``stdpopsim_homsap``
     - *H. sapiens*
     - 1,000
   * - ``stdpopsim_bostau``
     - *B. taurus*
     - 1,000
   * - ``stdpopsim_canfam``
     - *C. familiaris*
     - 1,000
   * - ``stdpopsim_pantro``
     - *P. troglodytes*
     - 1,000
   * - ``stdpopsim_papanu``
     - *P. anubis*
     - 1,000
   * - ``stdpopsim_ponabe``
     - *P. abelii*
     - 1,000
   * - ``stdpopsim_aedaeg``
     - *A. aegypti*
     - 300
   * - ``stdpopsim_anogam``
     - *A. gambiae*
     - 100
   * - ``stdpopsim_aratha``
     - *A. thaliana*
     - 500
   * - ``stdpopsim_caeele``
     - *C. elegans*
     - 1,000
   * - ``stdpopsim_dromel``
     - *D. melanogaster*
     - 5
   * - ``stdpopsim_gasacu``
     - *G. aculeatus*
     - 1,000
   * - ``stdpopsim_helann``
     - *H. annuus*
     - 300
   * - ``stdpopsim_helmel``
     - *H. melpomene*
     - 5
   * - ``stdpopsim_apimel``
     - *A. mellifera*
     - 5
   * - ``stdpopsim_musmus``
     - *M. musculus*
     - 1,000

Python API
----------

For programmatic use, the core simulation functions are available from
``cxt.utils``:

.. code-block:: python

   from cxt.utils import (
       simulate_parameterized_tree_sequence,
       create_sawtooth_demography,
   )

   # Constant Ne
   ts = simulate_parameterized_tree_sequence(seed=42, samples=25)

   # Sawtooth demography
   dem = create_sawtooth_demography(Ne=20_000, magnitude=3)
   ts = simulate_parameterized_tree_sequence(seed=42, demography=dem, samples=25)
