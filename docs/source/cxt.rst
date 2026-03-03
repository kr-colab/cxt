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


SFS Computation
---------------

.. automodule:: cxt.sfs
   :members:
   :undoc-members:


Bias Correction
---------------

.. automodule:: cxt.correction
   :members:
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

.. automodule:: cxt.simulate
   :members: simulate_parameterized_tree_sequence, simulate_random_segment, create_sawtooth_demography, sample_demography, sample_population_size, DemographyStorage
   :undoc-members:


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
