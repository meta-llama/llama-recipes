#!/bin/bash
###
#SBATCH --job-name=llama_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=20:00
#SBATCH --output="%x_%j.out"
#SBATCH --exclusive

module load image-defaults

# NCCL environment variables are documented at:
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ibp
export UCX_NET_DEVICES=ibp0:1,ibp1:1,ibp2:1,ibp3:1,ibp4:1,ibp5:1,ibp6:1,ibp7:1
export SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1
export NCCL_COLLNET_ENABLE=0
export LD_LIBRARY_PATH=/home/llama3_poc/llama-recipes/lenv/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# stops running after a fail (e stops the code) (x prints each command before it runs it)
# (o pipefail fails in series)
set -euxo pipefail

source ../../../../lenv/bin/activate

# Log the assigned nodes
echo "Using nodes: $SLURM_JOB_NODELIST"

### meta-llama/Meta-Llama-3.1-8B-Instruct
### "../../finetuning/model_checkpoints/fine-tuned-batch-1-meta-llama/Meta-Llama-3.1-8B-Instruct-Converted-2"
srun torchrun --nproc_per_node 1 chat_completion/chat_completion.py --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" --prompt_file chat_completion/inference_data.json
