API Reference
=============


cxt (top-level)
---------------

.. automodule:: cxt
   :members:
   :undoc-members:
   :no-index:


Configuration
-------------

.. automodule:: cxt.config
   :members:
   :undoc-members:


Model Loading
-------------

.. automodule:: cxt.checkpoint
   :members: load_model, get_checkpoint_path, CHECKPOINT_REGISTRY, GITHUB_BASE
   :undoc-members:


Inference
---------

.. automodule:: cxt.translate
   :members: translate, translate_from_ts, translate_from_vcf, translate_from_genotype_matrix, vcf_parser, generate, multi_gpu_generate, to_log_times, generate_causal_mask
   :undoc-members:

.. note::

   ``translate()`` auto-detects the input type (tree sequence, VCF path,
   or genotype matrix tuple) and dispatches to the appropriate backend.
   When ``mutation_rate`` is provided, a per-block stochastic bias
   correction is applied using
   :func:`cxt.correction.stochastic_diversity_bias_correction_v2`.


SFS Computation
---------------

.. automodule:: cxt.sfs
   :members:
   :undoc-members:


Bias Correction
---------------

.. automodule:: cxt.correction
   :members: diversity_bias_correction, diversity_bias_correction_by_rep, stochastic_diversity_bias_correction, stochastic_diversity_bias_correction_v2
   :undoc-members:


Model Architecture
------------------

.. automodule:: cxt.model
   :members:
   :undoc-members:

.. automodule:: cxt.modules
   :members:
   :undoc-members:


Training
--------

.. automodule:: cxt.train
   :members: LitDecoder, LitAdapterDecoder, IEAdapter, FrozenDecoderWithAdapter
   :undoc-members:


Dataset
-------

.. automodule:: cxt.dataset
   :members:
   :undoc-members:


Simulation
----------

Simulation functions live in two modules:

- ``cxt.utils`` contains ``simulate_parameterized_tree_sequence``,
  ``create_sawtooth_demography``, ``sample_demography``,
  ``sample_population_size``, and ``DemographyStorage``.
- ``cxt.simulation_ts_only`` is the CLI entry point used by
  ``run_fresh.sh`` (invoked as ``python cxt/simulation_ts_only.py``).

.. autofunction:: cxt.utils.simulate_parameterized_tree_sequence

.. autofunction:: cxt.utils.create_sawtooth_demography

.. autofunction:: cxt.utils.sample_demography

.. autofunction:: cxt.utils.sample_population_size

.. autoclass:: cxt.utils.DemographyStorage
   :members:


Preprocessing
-------------

.. automodule:: cxt.preprocess
   :members: process_X, process_y, ts2X_vectorized_bichan, interpolate_tmrcas, interpolate_tmrca_per_window_spanavg, find_ts_files, deterministic_split_grouped, choose_pairs, scenario_from_path, missingness_by_window_scales, bitmask_to_intervals, process_X_with_bitmask
   :undoc-members:


Utilities
---------

.. automodule:: cxt.utils
   :members:
   :undoc-members:
