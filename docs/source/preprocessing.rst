Preprocessing
==========

Before training the ts outputs need to be preprocessed into the format required by ``cxt``. The following code shows how to preprocess the entire training and validation datasets, additional datasets for fine-tuning are subsetted from these through preprocessing scripts.

.. code-block:: bash

    
    # Vanilla preprocessing
    python preprocess.py   --base_dir /sietch_colab/kkor/cxt/ts   --out_subdir processed   --window_size 2000   --num_pairs 200   --train_ratio 0.9   --global_seed 12345   --num_workers 75   --skip_existing

    # creating adapter datasets
    python preprocess.py --base_dir /sietch_colab/kkor/cxt/ts --out_subdir processed_n10 --window_size 2000 --num_pairs 20 --simplify_first_n_samples 10 --train_ratio 0.9 --num_workers 75

    # fine-tuning for different window size
    python preprocess.py   --base_dir /sietch_colab/kkor/cxt/ts_large_pop   --out_subdir processed_small_window   --window_size 200 --sequence_length 100000  --num_pairs 200   --train_ratio 0.9   --global_seed 12345   --num_workers 75   --skip_existing

    # missing data in mind AND small window size
    python preprocess.py   --base_dir /sietch_colab/kkor/cxt/ts_large_pop   --out_subdir processed_small_window_missing_data   --window_size 200 --sequence_length 100000  --num_pairs 200   --train_ratio 0.9   --global_seed 12345   --num_workers 75   --skip_existing --bitmask /sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz

    # missing data and adapter for small sample size
    python preprocess.py   --base_dir /sietch_colab/kkor/cxt/ts_large_pop   --out_subdir processed_small_window_missing_data_n10   --window_size 200 --sequence_length 100000  --num_pairs 20   --train_ratio 0.9   --global_seed 12345   --num_workers 75   --skip_existing --bitmask /sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz --simplify_first_n_samples 10

Leading to the following preprocessed dataset ready for training below:   
processed (vanilla)   
processed_small_window (for the w200 window sizes)    
processed_n10 (for the adapter)   

processed_small_window_missing_data (... + missing data)   
processed_small_window_missing_data_n10 (all in one go: w200 + missing_data + n10)    
