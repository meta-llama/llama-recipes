#!/bin/bash
###
#SBATCH --job-name=llama_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out"
#SBATCH --exclusive

module load image-defaults

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export LD_LIBRARY_PATH=/home/andy_convergence_ai/20240808/lenv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# stops running after a fail (e stops the code) (x prints each command before it runs it)
# (o pipefail fails in series)
set -euxo pipefail

source ../../../lenv/bin/activate

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node IP: $head_node_ip"

srun torchrun --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node 8 --nnodes 1 finetuning.py --enable_fsdp --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned-batch-1
