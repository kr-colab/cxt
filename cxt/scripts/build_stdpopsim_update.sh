python simulation.py --num_processes 100 --num_samples 100000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir /sietch_colab/kkor/dataset/stdpopsim/v0.3/stdpopsim_musmus \
    --scenario stdpopsim_musmus; 

python simulation.py --num_processes 100 --num_samples 100000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir /sietch_colab/kkor/dataset/stdpopsim/v0.3/stdpopsim_ratnor \
    --scenario stdpopsim_ratnor; ### check again was stdpopsim_musmus

python simulation.py --num_processes 100 --num_samples 100000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir /sietch_colab/kkor/dataset/stdpopsim/v0.3/stdpopsim_gorgor \
    --scenario stdpopsim_gorgor; 

python simulation.py --num_processes 50 --num_samples 100000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir /sietch_colab/kkor/dataset/stdpopsim/v0.3/stdpopsim_orysat \
    --scenario stdpopsim_orysat; 

python simulation.py --num_processes 100 --num_samples 100000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir /sietch_colab/kkor/dataset/stdpopsim/v0.3/stdpopsim_susscr \
    --scenario stdpopsim_susscr; 

python simulation.py --num_processes 100 --num_samples 100000 --batch_size 1000 \
    --start_batch 0 --pivot_A 0 --pivot_B 1 --randomize_pivots True \
    --data_dir /sietch_colab/kkor/dataset/stdpopsim/v0.3/stdpopsim_phosin \
    --scenario stdpopsim_phosin; 


