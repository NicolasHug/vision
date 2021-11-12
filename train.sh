#!/bin/bash
#SBATCH --partition=train
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --time=70:00:00
#SBATCH --nodes=1
#SBATCH --output=/data/home/nicolashug/cluster/experiments/slurm-%j.out
#SBATCH --error=/data/home/nicolashug/cluster/experiments/slurm-%j.err



n_gpus=2
n_nodes=1

output_dir=~/cluster/experiments/id_$SLURM_JOB_ID
mkdir -p $output_dir



this_script=./train.sh  # depends where you call it from
cp $this_script $output_dir

function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
master_port=$(unused_port)

# FlyingChairs
batch_size_chairs=5
lr_chairs=0.0004
num_steps_chairs=100000
name_chairs=raft_chairs
wdecay_chairs=0.0001

chairs_dir=$output_dir/chairs
mkdir -p $chairs_dir
torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/raft/train.py \
    --name $name_chairs \
    --train-dataset chairs \
    --batch-size $batch_size_chairs \
    --lr $lr_chairs \
    --weight-decay $wdecay_chairs \
    --num-steps $num_steps_chairs \
    --num-epochs 100000 \
    --output-dir $chairs_dir

# FlyingThings3D
batch_size_things=3
lr_things=0.000125
num_steps_things=100000
name_things=raft_things
wdecay_things=0.0001

things_dir=$output_dir/things
mkdir -p $things_dir
torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/raft/train.py \
    --name $name_things \
    --train-dataset things \
    --batch-size $batch_size_things \
    --lr $lr_things \
    --weight-decay $wdecay_things \
    --num-steps $num_steps_things \
    --num-epochs 100000 \
    --output-dir $things_dir\
    --resume $chairs_dir/$name_chairs.pth

# # Sintel S+K+H
# batch_size_sintel_skh=6
# lr_sintel_skh=0.000125
# num_steps_sintel_skh=25000
# name_sintel_skh=raft_sintel_skh
# wdecay_sintel_skh=0.00001
# gamma_sintel_skh=0.85

# sintel_skh_dir=$output_dir/sintel_skh
# mkdir -p $sintel_skh_dir
# torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/raft/train.py \
#     --name $name_sintel_skh \
#     --train-dataset sintel_skh \
#     --batch-size $batch_size_sintel_skh \
#     --lr $lr_sintel_skh \
#     --weight-decay $wdecay_sintel_skh \
#     --gamma $gamma_sintel_skh \
#     --num-steps $num_steps_sintel_skh \
#     --num-epochs 100000 \
#     --output-dir $sintel_skh_dir\
#     --resume $things_dir/$name_things.pth

# # # Sintel
# # batch_size_sintel=6
# # lr_sintel=0.000125
# # num_steps_sintel=25000
# # name_sintel=raft_sintel
# # wdecay_sintel=0.00001
# # gamma_sintel=0.85

# # sintel_dir=$output_dir/sintel
# # mkdir -p $sintel_dir
# # torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/raft/train.py \
# #     --name $name_sintel \
# #     --train-dataset sintel \
# #     --batch-size $batch_size_sintel \
# #     --lr $lr_sintel \
# #     --weight-decay $wdecay_sintel \
# #     --gamma $gamma_sintel \
# #     --num-steps $num_steps_sintel \
# #     --num-epochs 100000 \
# #     --output-dir $sintel_dir\
# #     --resume $things_dir/$name_things.pth

# # # Kitti
# # batch_size_kitti=6
# # lr_kitti=0.0001
# # num_steps_kitti=12500
# # name_kitti=raft_kitti
# # wdecay_kitti=0.00001
# # gamma_kitti=0.85

# # kitti_dir=$output_dir/kitti
# # mkdir -p $kitti_dir
# # torchrun --nproc_per_node $n_gpus --nnodes $n_nodes --master_port $master_port references/raft/train.py \
# #     --name $name_kitti \
# #     --train-dataset kitti \
# #     --batch-size $batch_size_kitti \
# #     --lr $lr_kitti \
# #     --weight-decay $wdecay_kitti \
# #     --gamma $gamma_kitti \
# #     --num-steps $num_steps_kitti \
# #     --num-epochs 100000 \
# #     --output-dir $kitti_dir \
# #     --resume $sintel_dir/$name_sintel.pth