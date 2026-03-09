
# base dataset

base_dir="${BASE_DIR:-./data/ts}"

mkdir -p $base_dir
mkdir -p ${base_dir}/llm
mkdir -p ${base_dir}/stdpopsim/v0.2

python simulation_ts_only.py \
    --num_processes 50 \
    --num_samples 10000 \
    --data_dir $base_dir/base_dataset \
    --scenario constant

# sawtooth
python simulation_ts_only.py \
    --num_processes 30 \
    --num_samples 1000 \
    --data_dir ${base_dir}/ssd \
    --scenario sawtooth

# island demography dataset

python simulation_ts_only.py \
    --num_processes 30 \
    --num_samples 1000 \
    --data_dir ${base_dir}/idd \
    --scenario island \

# auxillary dataset

python simulation_ts_only.py \
    --num_processes 100 \
    --num_samples 125 \
    --data_dir ${base_dir}/llm  \ 
    --scenario llm_ne_sawtooth   \

python simulation_ts_only.py \
    --num_processes 100 \
    --num_samples 50 \
    --data_dir ${base_dir}/llm  \
    --scenario llm_hard_sweeps   \

python simulation_ts_only.py \
    --num_processes 75 \
    --num_samples 50 \
    --data_dir ${base_dir}/llm \
    --scenario llm_island_3pop   \

python simulation_ts_only.py \
    --num_processes 100 \
    --num_samples 500 \
    --data_dir ${base_dir}/llm  \
    --scenario llm_ne_constant   \

# stdpopsim mammals

python simulation_ts_only.py --num_processes 75 --num_samples 1000  \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_homsap  \
    --scenario stdpopsim_homsap;

python simulation_ts_only.py --num_processes 75 --num_samples 1000  \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_homsap_map  \
    --scenario stdpopsim_homsap_map;   

python simulation_ts_only.py --num_processes 75 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_bostau  \
    --scenario stdpopsim_bostau;   

python simulation_ts_only.py --num_processes 75 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_canfam  \
    --scenario stdpopsim_canfam;   

python simulation_ts_only.py --num_processes 75 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_canfam_map  \
    --scenario stdpopsim_canfam_map;

python simulation_ts_only.py --num_processes 75 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_pantro  \
    --scenario stdpopsim_pantro;  

python simulation_ts_only.py --num_processes 75 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_papanu  \
    --scenario stdpopsim_papanu;  

python simulation_ts_only.py --num_processes 75 --num_samples 1000  \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_papanu_map  \
    --scenario stdpopsim_papanu_map;  

python simulation_ts_only.py --num_processes 75 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_ponabe  \
    --scenario stdpopsim_ponabe;   

python simulation_ts_only.py --num_processes 75 --num_samples 1000  \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_ponabe_map  \
    --scenario stdpopsim_ponabe_map;   


# stdpopsim other species

#1
python simulation_ts_only.py --num_processes 100 --num_samples 300 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_aedaeg  \
    --scenario stdpopsim_aedaeg;   
#2
python simulation_ts_only.py --num_processes 100 --num_samples 25 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_anapla \
    --scenario stdpopsim_anapla;   
#3
python simulation_ts_only.py --num_processes 100 --num_samples 5 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_anocar \
    --scenario stdpopsim_anocar;   
#4
python simulation_ts_only.py --num_processes 100 --num_samples 100 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_anogam \
    --scenario stdpopsim_anogam;   
#5
##python simulation_ts_only.py --num_processes 100 --num_samples 5 \
##    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_apimel \
##    --scenario stdpopsim_apimel; 
#6
python simulation_ts_only.py --num_processes 100 --num_samples 500 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_aratha \
    --scenario stdpopsim_aratha; 
#7
python simulation_ts_only.py --num_processes 100 --num_samples 500 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_aratha_map \
    --scenario stdpopsim_aratha_map; 

#8
python simulation_ts_only.py --num_processes 100 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_caeele \
    --scenario stdpopsim_caeele; 

#9
python simulation_ts_only.py --num_processes 100 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_caeele_map \
    --scenario stdpopsim_caeele_map; 

#10
# rerun due to error in Ne
python simulation_ts_only.py --num_processes 100 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_canfam \
    --scenario stdpopsim_canfam; 

#11
python simulation_ts_only.py --num_processes 100 --num_samples 5 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_dromel \
    --scenario stdpopsim_dromel; 

#12
##python simulation_ts_only.py --num_processes 100 --num_samples 5 \
##    --data_dir /sietch_colab/kkor/stdpopsim_dromel_map \
##    --scenario stdpopsim_dromel_map; 

#13
python simulation_ts_only.py --num_processes 100 --num_samples 300 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_drosec \
    --scenario stdpopsim_drosec; 

#14
python simulation_ts_only.py --num_processes 100 --num_samples 1000 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_gasacu \
    --scenario stdpopsim_gasacu; 

#15
python simulation_ts_only.py --num_processes 100 --num_samples 300 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_helann \
    --scenario stdpopsim_helann; 

#16
python simulation_ts_only.py --num_processes 100 --num_samples 5 \
    --data_dir ${base_dir}/stdpopsim/v0.2/stdpopsim_helmel \
    --scenario stdpopsim_helmel; 

