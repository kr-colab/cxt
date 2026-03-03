Simulation
==========

All training data for cxt is generated using ``python -m cxt.simulate``, which
consolidates every simulation scenario into a single CLI. This page documents
the exact commands that produce the training data used in the paper.

The simulation module generates tree sequences using ``msprime`` and
``stdpopsim``, saving them as ``.trees`` files organized by scenario. These
tree sequences are then preprocessed into training pairs (see
:doc:`preprocessing`).


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

   python -m cxt.simulate \
       --scenario <scenario_name> \
       --data-dir <output_directory> \
       --num-samples <n_simulations> \
       --num-processes <n_parallel> \
       [--n-individuals 25] \
       [--batch-size 1000] \
       [--randomize-pivots] \
       [--save-trees]

Key arguments:

- ``--scenario``: simulation scenario (see below)
- ``--data-dir``: output directory for data files
- ``--num-samples``: total number of simulations to generate
- ``--num-processes``: parallel worker count
- ``--n-individuals``: diploid sample count per simulation (default: 25)
- ``--randomize-pivots``: randomly pick pivot pairs instead of using (0, 1)
- ``--save-trees``: also save raw tree sequences (``.trees``) alongside
  the X/y arrays, enabling downstream preprocessing with
  ``python -m cxt.preprocess``


Full pipeline (paper)
---------------------

The following script reproduces the complete training dataset. Adjust
``DATA_DIR`` to point to your storage location.

.. code-block:: bash

   #!/usr/bin/env bash
   set -euo pipefail

   DATA_DIR=/path/to/training_data

   mkdir -p ${DATA_DIR}
   mkdir -p ${DATA_DIR}/llm
   mkdir -p ${DATA_DIR}/stdpopsim/v0.2

   # ================================================================
   # 1. Base dataset
   # ================================================================

   # Constant Ne (10,000 simulations)
   python -m cxt.simulate \
       --num-processes 50 \
       --num-samples 10000 \
       --data-dir ${DATA_DIR}/base_dataset \
       --scenario constant

   # Sawtooth demography (1,000 simulations)
   python -m cxt.simulate \
       --num-processes 30 \
       --num-samples 1000 \
       --data-dir ${DATA_DIR}/ssd \
       --scenario sawtooth

   # Island model (1,000 simulations)
   python -m cxt.simulate \
       --num-processes 30 \
       --num-samples 1000 \
       --data-dir ${DATA_DIR}/idd \
       --scenario island

   # ================================================================
   # 2. LLM-style datasets (parameter sweeps)
   # ================================================================

   # Sawtooth with varying Ne, mu, rec, magnitude
   python -m cxt.simulate \
       --num-processes 100 \
       --num-samples 125 \
       --data-dir ${DATA_DIR}/llm \
       --scenario llm_ne_sawtooth

   # Hard sweeps
   python -m cxt.simulate \
       --num-processes 100 \
       --num-samples 50 \
       --data-dir ${DATA_DIR}/llm \
       --scenario llm_hard_sweeps

   # 3-population island model
   python -m cxt.simulate \
       --num-processes 75 \
       --num-samples 50 \
       --data-dir ${DATA_DIR}/llm \
       --scenario llm_island_3pop

   # Constant Ne with varying mu and rec
   python -m cxt.simulate \
       --num-processes 100 \
       --num-samples 500 \
       --data-dir ${DATA_DIR}/llm \
       --scenario llm_ne_constant

   # ================================================================
   # 3. stdpopsim mammals (great apes, cattle)
   # ================================================================

   # H. sapiens (flat recombination)
   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_homsap \
       --scenario stdpopsim_homsap

   # H. sapiens (HapMapII genetic map)
   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_homsap_map \
       --scenario stdpopsim_homsap_map

   # B. taurus (cattle)
   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_bostau \
       --scenario stdpopsim_bostau

   # C. familiaris (dog)
   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_canfam \
       --scenario stdpopsim_canfam

   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_canfam_map \
       --scenario stdpopsim_canfam_map

   # P. troglodytes (chimpanzee)
   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_pantro \
       --scenario stdpopsim_pantro

   # P. anubis (olive baboon)
   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_papanu \
       --scenario stdpopsim_papanu

   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_papanu_map \
       --scenario stdpopsim_papanu_map

   # P. abelii (orangutan)
   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_ponabe \
       --scenario stdpopsim_ponabe

   python -m cxt.simulate --num-processes 75 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_ponabe_map \
       --scenario stdpopsim_ponabe_map

   # ================================================================
   # 4. stdpopsim other species
   # ================================================================

   # A. aegypti (yellow fever mosquito)
   python -m cxt.simulate --num-processes 100 --num-samples 300 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_aedaeg \
       --scenario stdpopsim_aedaeg

   # A. platyrhynchos (mallard)
   python -m cxt.simulate --num-processes 100 --num-samples 25 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_anapla \
       --scenario stdpopsim_anapla

   # A. carolinensis (green anole)
   python -m cxt.simulate --num-processes 100 --num-samples 5 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_anocar \
       --scenario stdpopsim_anocar

   # A. gambiae (malaria mosquito)
   python -m cxt.simulate --num-processes 100 --num-samples 100 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_anogam \
       --scenario stdpopsim_anogam

   # A. thaliana (thale cress)
   python -m cxt.simulate --num-processes 100 --num-samples 500 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_aratha \
       --scenario stdpopsim_aratha

   python -m cxt.simulate --num-processes 100 --num-samples 500 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_aratha_map \
       --scenario stdpopsim_aratha_map

   # C. elegans (nematode)
   python -m cxt.simulate --num-processes 100 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_caeele \
       --scenario stdpopsim_caeele

   python -m cxt.simulate --num-processes 100 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_caeele_map \
       --scenario stdpopsim_caeele_map

   # D. melanogaster (fruit fly)
   python -m cxt.simulate --num-processes 100 --num-samples 5 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_dromel \
       --scenario stdpopsim_dromel

   # D. sechellia
   python -m cxt.simulate --num-processes 100 --num-samples 300 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_drosec \
       --scenario stdpopsim_drosec

   # G. aculeatus (stickleback)
   python -m cxt.simulate --num-processes 100 --num-samples 1000 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_gasacu \
       --scenario stdpopsim_gasacu

   # H. annuus (sunflower)
   python -m cxt.simulate --num-processes 100 --num-samples 300 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_helann \
       --scenario stdpopsim_helann

   # H. melpomene (Heliconius butterfly)
   python -m cxt.simulate --num-processes 100 --num-samples 5 \
       --data-dir ${DATA_DIR}/stdpopsim/v0.2/stdpopsim_helmel \
       --scenario stdpopsim_helmel


Available scenarios
-------------------

Run ``python -m cxt.simulate --help`` for the full list. The main categories
are:

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

Python API
----------

For programmatic use, the core simulation functions are available directly:

.. code-block:: python

   from cxt.simulate import (
       simulate_parameterized_tree_sequence,
       simulate_random_segment,
       create_sawtooth_demography,
   )

   # Constant Ne
   ts = simulate_parameterized_tree_sequence(seed=42, samples=25)

   # Sawtooth demography
   dem = create_sawtooth_demography(Ne=20_000, magnitude=3)
   ts = simulate_parameterized_tree_sequence(seed=42, demography=dem, samples=25)

   # stdpopsim species
   ts = simulate_random_segment(seed=42, species_name="HomSap", num_samples=25)
