# Procedure to run benchmark environment on ASPIRE 2A
- This document is to describe how to run LLM fine-tuning benchmark on ASPIRE 2A.
- This Llama recipes repo was forked from original Meta's repo (original repo's URL: https://github.com/meta-llama/llama-recipes). 

# Set up at a login node
## Load environment modules and create a virtual environment
```
module load cuda/11.8.0
module load python/3.11.7-gcc11
cd /scratch/project/90000001/ta1/
mkdir 2B_RFP; cd 2B_RFP/
python -m venv venv_2b_rfp_bench_ai
source /scratch/project/90000001/ta1/2B_RFP/venv_2b_rfp_bench_ai/bin/activate
```
Note: You may change directory path according to your need

## Install Flash Attention (optional)
Flash Attention is to improve Transformer based model performance.
https://huggingface.co/docs/transformers/perf_infer_gpu_one?install=NVIDIA#flashattention-2

At this point of time, CUDA version needs to be 11. See following discussions in Flash Attention repo.
- https://github.com/Dao-AILab/flash-attention/issues/620
- https://github.com/Dao-AILab/flash-attention/issues/853
```
pip install wheel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# Run following command beforehand in case of installation failure due to disk quota exceeded
# rm -rf ~/.cache/pip/
```
Note: You may use `pip install flash-attn --no-build-isolation` to install a Flash Attention module, but a specific module is specified in this example in order to avoid the issue mentioned above.

## Install modules to run code in Llama receipes repository
```
cd /scratch/project/90000001/ta1/2B_RFP
git clone https://github.com/nsccsg/llama-recipes.git
# You may also clone from the original repo from https://github.com/meta-llama/llama-recipes.git if you want
cd llama-recipes/
pip install -r requirements.txt
pip install -e .
```

## Download TinyLlama v1.1
Download TinyLlama_v1.1 model from Hugging Face website. You need to create Hugging Face access token beforehand.
```
cd /scratch/project/90000001/ta1/2B_RFP/llama-recipes
# Git LFS is required to download a model from Hugging Face
git lfs install
# Login Hugging Face in order to download a model
huggingface-cli login
# Download TinyLlama v1.1 model
git clone https://huggingface.co/TinyLlama/TinyLlama_v1.1
```

## Run TinyLlama fine-tuning benchmark with PyTorch FSDP
Create a PBS script files to run 

PBS script example for 2 nodes and 2 GPUs in each:
```
#!/bin/bash
#PBS -N taka_bench_2brfp
#PBS -l select=2:ngpus=2
#PBS -l walltime=00:30:00
#PBS -P 90000001
#PBS -q normal
#PBS -j oe
#PBS -o /home/users/astar/ares/ta1/taka_bench_2brfp_2node2GPU.log

# Set parallelism
N_NODE=2
N_PROC_PER_NODE=2
N_PROCESS=$((N_PROC_PER_NODE*$N_NODE))

# Directory that a user submits a job at
cd $PBS_O_WORKDIR

# Set master address and port number
MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
MASTER_PORT=29603

# Set log directory
LOG_DIR=${PBS_O_WORKDIR}/n_node${N_NODE}__n_proc_per_node${N_PROC_PER_NODE}/logs
echo $LOG_DIR
mkdir -p $LOG_DIR

# Set node index
NODE_RANK=0
for NODE in `cat $PBS_NODEFILE`;
do
echo $NODE
echo $NODE_RANK
ssh ta1@$NODE "bash /home/users/astar/ares/ta1/finetuning_torchrun.sh $N_NODE $N_PROC_PER_NODE $NODE_RANK $MASTER_ADDR $MASTER_PORT" > $LOG_DIR/log_$NODE_RANK.txt 2>&1 &

NODE_RANK=$((NODE_RANK+1))
done
wait
```

`finetuning_torchrun.sh`
```
#!/bin/bash

# Load environment modules
module load cuda/11.8.0 
module load python/3.11.7-gcc11 
source /scratch/project/90000001/ta1/2B_RFP/venv_2b_rfp_bench_ai/bin/activate 

N_NODE=$1
N_PROC_PER_NODE=$2
NODE_RANK=$3
MASTER_ADDR=$4
MASTER_PORT=$5
N_PROCESS=$((N_NODE*N_PROC_PER_NODE))
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
# Set Slingshot interfaces to be used by NCCL
NCCL_SOCKET_IFNAME=hsn0,hsn1

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
```

# Optional
## Enable Flash Attention
NSCC may give vendors the freedom to use Flash Attention if their accelerator supports and it boosts benchmark performance. As of today, NVIDIA GPU and AMD GPU support Flash Attention 2 according to Hugging Face website. Not sure about Intel Gaudi/Falcon Shores.
- NVIDIA: https://huggingface.co/docs/transformers/perf_infer_gpu_one?install=NVIDIA#flashattention-2
- AMD: https://huggingface.co/docs/transformers/perf_infer_gpu_one?install=AMD#flashattention-2

In order to use Flash Attention 2 for this benchmark, please update src/llama_recipes/finetuning.py as follows:
Original code using scaled dot product attention (SDPA):
```
    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )
```

Updated code using Flash Attention 2:
```
    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
```
