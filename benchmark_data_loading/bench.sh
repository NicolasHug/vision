#!/bin/bash
#SBATCH --partition=train
#SBATCH --cpus-per-task=96 # 12 CPUs per GPU
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --time=70:00:00

mkdir -p /scratch/nicolashug

# cp /ontap_isolated/nicolashug/tinyimagenet /scratch/nicolashug/tinyimagenet -rv
cp /ontap_isolated/imagenet_full_size /scratch/nicolashug/imagenet_full_size -rv

python bench_transforms.py #--tiny
python bench_decoding.py #--tiny

for fs in "scratch" "fsx_isolated" # "ontap_isolated"
do
    for script in "bench_data_reading.py" "bench_data_reading_decoding.py" "bench_e2e.py"
    do
        for num_workers in 0 12
        do
            # echo $script $fs $num_workers
            python $script --fs $fs --num-workers $num_workers #--tiny
        done
    done
done