#!/bin/bash
###
#SBATCH --job-name=llama_convert
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

srun python -m llama_recipes.inference.checkpoint_converter_fsdp_hf --fsdp_checkpoint_path=model_checkpoints/fine-tuned-batch-1-meta-llama/Meta-Llama-3.1-8B-Instruct/ --consolidated_model_path=model_checkpoints/fine-tuned-batch-1-meta-llama/Meta-Llama-3.1-8B-Instruct-Converted-2/ --HF_model_path_or_name=meta-llama/Meta-Llama-3.1-8B-Instruct
