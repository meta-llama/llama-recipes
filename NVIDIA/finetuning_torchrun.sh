#!/bin/bash

# Load environment modules and load a virtual environment that you created
module load cuda/11.8.0 
module load python/3.11.7-gcc11 
source /scratch/project/90000001/ta1/2B_RFP/venv_2b_rfp_bench_ai/bin/activate 

# Read script arguments
N_NODE=$1
N_PROC_PER_NODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
N_PROCESS=$((N_NODE*N_PROC_PER_NODE))
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

# Set Slingshot interfaces to be used by NCCL
NCCL_SOCKET_IFNAME=hsn0,hsn1

# Kick LLM fine-tuning
if [ $N_PROCESS -eq 1 ]; then
CUDA_VISIBLE_DEVICES=0
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
torchrun \
--standalone \
--nproc_per_node=$N_PROC_PER_NODE \
/scratch/project/90000001/ta1/2B_RFP/llama-recipes/recipes/finetuning/finetuning.py \
--model_name /scratch/project/90000001/ta1/2B_RFP/llama-recipes/TinyLlama_v1.1 \
--use_peft \
--peft_method lora \
--output_dir ./ \
--use_fast_kernels

else
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_PROC_PER_NODE-1)))
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
torchrun \
--nnodes=$N_NODE \
--nproc_per_node=$N_PROC_PER_NODE \
--node_rank=$NODE_RANK \
--rdzv_id=456 \
--rdzv_backend=c10d \
--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
/scratch/project/90000001/ta1/2B_RFP/llama-recipes/recipes/finetuning/finetuning.py \
--enable_fsdp \
--model_name /scratch/project/90000001/ta1/2B_RFP/llama-recipes/TinyLlama_v1.1 \
--use_peft \
--peft_method lora \
--output_dir ./ \
--use_fast_kernels

fi
