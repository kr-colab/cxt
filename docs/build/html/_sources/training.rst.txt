Training
========

``cxt`` is trained using the Pytorch Lightning framework. The following code demonstrates how to train the model.

.. code-block:: bash
    #Training the narrow model:
    python train2.py --dataset_path  /sietch_colab/kkor/cxt/ts/processed --gpus 0 1 2  --num_epochs 6

    #Training the broad model:

    #Vanilla Broad Model:
    python train2.py --dataset_path  /sietch_colab/kkor/cxt/ts/processed --gpus 0 1 2  --num_epochs 2


    #Broad_w200 (fine-tuning for large populations): 
    python train2.py --dataset_path  /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window --gpus 0 1 2  --num_epochs 2 --learning_rate 3e-5 --checkpoint_path /home/kkor/cxt/cxt/lightning_logs/version_20/checkpoints/epoch=1-step=5280.ckpt

    #Broad_w200_missing
    # Make sure to deactivate singelton masking (if not done automatically)
    python train2.py --dataset_path  /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window_missing_data --gpus 0 1 2  --num_epochs 2 --learning_rate 3e-5 --checkpoint_path /home/kkor/cxt/cxt/lightning_logs/version_20/checkpoints/epoch=1-step=5280.ckpt

    #Broad_w200_missing with more epochs
    # Same as previous but with more epochs
    python train2.py --dataset_path  /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window_missing_data --gpus 0 1 2  --num_epochs 10 --learning_rate 3e-5 --checkpoint_path /home/kkor/cxt/cxt/lightning_logs/version_20/checkpoints/epoch=1-step=5280.ckpt

    # currently setup to automatically load the broad model
    python train2_n10.py --dataset_path  /sietch_colab/kkor/cxt/ts/processed_n10 --gpus 0 1 2  --num_epochs 3

    #Broad_w200_missing_n10 (all in one go)
    # uses broad+adapter checkpoint
    python train2_n10.py --dataset_path  /sietch_colab/kkor/cxt/ts_large_pop/processed_small_window_missing_data_n10 --gpus 0 1 2  --num_epochs 2/10 --learning_rate 3e-5 --checkpoint_path /sietch_colab/data_share/cxt/models/broad+adapter/version_26/checkpoints/epoch=2-step=792.ckpt
