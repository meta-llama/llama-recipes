# Fine-tuning with Multi GPU

To run fine-tuning on multi-GPUs, we will  make use of two packages:

1. [PEFT](https://huggingface.co/blog/peft) methods and in particular using the Hugging Face [PEFT](https://github.com/huggingface/peft)library.

2. [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html) which helps us parallelize the training over multiple GPUs. [More details](LLM_finetuning.md/#2-full-partial-parameter-finetuning).

Given the combination of PEFT and FSDP, we would be able to fine tune a Llama 2 model on multiple GPUs in one node or multi-node.

## Requirements 
To run the examples, make sure to install the requirements using 

```bash

pip install -r requirements.txt

```

**Please note that the above requirements.txt will install PyTorch 2.0.1 version, in case you want to run FSDP + PEFT, please make sure to install PyTorch nightlies.**

## How to run it

Get access to a machine with multiple GPUs ( in this case we tested with 4 A100 and A10s).
This runs with the `samsum_dataset` for summarization application by default.

**Multiple GPUs one node**:

**NOTE** please make sure to use PyTorch Nightlies for using PEFT+FSDP. Also, note that int8 quantization from bit&bytes currently is not supported in FSDP.

```bash

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --use_peft --peft_method lora --output_dir Path/to/save/PEFT/model

```

The args used in the command above are:

* `--enable_fsdp` boolean flag to enable FSDP  in the script

* `--use_peft` boolean flag to enable PEFT methods in the script

* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`.

We use `torchrun` here to spawn multiple processes for FSDP.


### Fine-tuning using FSDP Only

If interested in running full parameter finetuning without making use of PEFT methods, please use the following command. Make sure to change the `nproc_per_node` to your available GPUs. This has been tested with `BF16` on 8xA100, 40GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8  llama_finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 

```

**Multi GPU multi node**:

Here we use a slurm script to schedule a job with slurm over multiple nodes.

```bash

sbatch multi_node.slurm
# Change the num nodes and GPU per nodes in the script before running.

```

## How to run with different datasets?

Currently 4 datasets are supported that can be found in [Datasets config file](../configs/datasets.py).

* `grammar_dataset` : use this [notebook](../ft_datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process theJfleg and C4 200M datasets for grammar checking.

* `alpaca_dataset` : to get this open source data please download the `aplaca.json` to `ft_dataset` folder.

```bash
wget -P ft_datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

* `samsum_dataset`

To run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset
torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp  --model_name /patht_of_model_folder/7B --use_peft --peft_method lora --dataset grammar_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned  --pure_bf16 --output_dir Path/to/save/PEFT/model

# alpaca_dataset

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp  --model_name /patht_of_model_folder/7B --use_peft --peft_method lora --dataset alpaca_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --output_dir Path/to/save/PEFT/model


# samsum_dataset

torchrun --nnodes 1 --nproc_per_node 4  ../llama_finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --use_peft --peft_method lora --dataset samsum_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --output_dir Path/to/save/PEFT/model

```

## Where to configure settings?

* [Training config file](../configs/training.py) is the main config file that helps to specify the settings for our run and can be found in [configs folder](../configs/)

It lets us specify the training settings for everything from `model_name` to `dataset_name`, `batch_size` and so on. Below is the list of supported settings:

```python

model_name: str="PATH/to/LLAMA 2/7B"
enable_fsdp: bool= False
run_validation: bool=True
batch_size_training: int=4
num_epochs: int=3
num_workers_dataloader: int=2
lr: float=2e-4
weight_decay: float=0.0
gamma: float= 0.85
use_fp16: bool=False
mixed_precision: bool=True
val_batch_size: int=4
dataset = "samsum_dataset" # alpaca_dataset, grammar_dataset
micro_batch_size: int=1
peft_method: str = "lora" # None , llama_adapter, prefix
use_peft: bool=False
output_dir: str = "./ft-output"
freeze_layers: bool = False
num_freeze_layers: int = 1
quantization: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False

```

* [Datasets config file](../configs/datasets.py) provides the available options for datasets.

* [peft config file](../configs/peft.py) provides the supported PEFT methods and respective settings that can be modified.

* [FSDP config file](../configs/fsdp.py) provides FSDP settings such as:

    * `mixed_precision` boolean flag to specify using mixed precision, defatults to true.

    * `use_fp16` boolean flag to specify using FP16 for mixed precision, defatults to False. We recommond not setting this flag, and only set `mixed_precision` that will use `BF16`, this will help with speed and memory savings while avoiding challenges of scaler accuracies with `FP16`.

    *  `sharding_strategy` this specifies the sharding strategy for FSDP, it can be:
        * `FULL_SHARD` that shards model parameters, gradients and optimizer states, results in the most memory savings.

        * `SHARD_GRAD_OP` that shards gradinets and optimizer states and keeps the parameters after the first `all_gather`. This reduces communication overhead specially if you are using slower networks more specifically beneficial on multi-node cases. This comes with the trade off of higher memory consumption.

        * `NO_SHARD` this is equivalent to DDP, does not shard model parameters, gradinets or optimizer states. It keeps the full parameter after the first `all_gather`.

        * `HYBRID_SHARD` available on PyTorch Nightlies. It does FSDP within a node and DDP between nodes. It's for multi-node cases and helpful for slower networks, given your model will fit into one node.

* `checkpoint_type` specifies the state dict checkpoint type for saving the model. `FULL_STATE_DICT` streams state_dict of each model shard from a rank to CPU and assembels the full state_dict on CPU. `SHARDED_STATE_DICT` saves one checkpoint per rank, and enables the re-loading the model in a different world size.

* `fsdp_activation_checkpointing` enables activation checkpoining for FSDP, this saves significant amount of memory with the trade off of recomputing itermediate activations during the backward pass. The saved memory can be re-invested in higher batch sizes to increase the throughput. We recommond you use this option.

* `pure_bf16` it moves the  model to `BFloat16` and if `optimizer` is set to `anyprecision` then optimizer states will be kept in `BFloat16` as well. You can use this option if necessary.
