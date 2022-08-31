#!/bin/bash
#SBATCH --partition=train
#SBATCH --cpus-per-task=96 # 12 CPUs per GPU
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --time=70:00:00

cp /ontap_isolated/nicolashug/ /scratch/ -rv

python bench_data_loading_components.py --fs scratch
python bench_data_loading_components.py --fs ontap_isolated
python bench_data_loading_components.py --fs fsx_isolated

