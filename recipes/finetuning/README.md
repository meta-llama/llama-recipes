# Finetuning Llama

This folder contains instructions to fine-tune Llama 2 on a 
* [single-GPU setup](./singlegpu_finetuning.md)
* [multi-GPU setup](./multigpu_finetuning.md) 

using the canonical [finetuning script](../../src/llama_recipes/finetuning.py) in the llama-recipes package.

If you are new to fine-tuning techniques, check out an overview: [](./LLM_finetuning_overview.md)

> [!TIP]
> If you want to try finetuning Llama 2 with Huggingface's trainer, here is a Jupyter notebook with an [example](./huggingface_trainer/peft_finetuning.ipynb)


## How to configure finetuning settings?

> [!TIP]
> All the setting defined in [config files](../../src/llama_recipes/configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.


* [Training config file](../../src/llama_recipes/configs/training.py) is the main config file that helps to specify the settings for our run and can be found in [configs folder](../../src/llama_recipes/configs/)

It lets us specify the training settings for everything from `model_name` to `dataset_name`, `batch_size` and so on. Below is the list of supported settings:

```python

model_name: str="PATH/to/LLAMA 2/7B"
enable_fsdp: bool= False
run_validation: bool=True
batch_size_training: int=4
gradient_accumulation_steps: int=1
num_epochs: int=3
num_workers_dataloader: int=2
lr: float=2e-4
weight_decay: float=0.0
gamma: float= 0.85
use_fp16: bool=False
mixed_precision: bool=True
val_batch_size: int=4
dataset = "samsum_dataset" # alpaca_dataset, grammar_dataset
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

* [Datasets config file](../../src/llama_recipes/configs/datasets.py) provides the available options for datasets.

* [peft config file](../../src/llama_recipes/configs/peft.py) provides the supported PEFT methods and respective settings that can be modified.

* [FSDP config file](../../src/llama_recipes/configs/fsdp.py) provides FSDP settings such as:

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


## Weights & Biases Experiment Tracking

You can enable [W&B](https://wandb.ai/) experiment tracking by using `use_wandb` flag as below. You can change the project name, entity and other `wandb.init` arguments in `wandb_config`.

```bash
python -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model --use_wandb
```
You'll be able to access a dedicated project or run link on [wandb.ai](https://wandb.ai) and see your dashboard like the one below. 
<div style="display: flex;">
    <img src="../../docs/images/wandb_screenshot.png" alt="wandb screenshot" width="500" />
</div>
