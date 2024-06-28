# Fine-tuning with Multi GPU

To run fine-tuning on multi-GPUs, we will  make use of two packages:

1. [PEFT](https://huggingface.co/blog/peft) methods and in particular using the Hugging Face [PEFT](https://github.com/huggingface/peft)library.

2. [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html) which helps us parallelize the training over multiple GPUs. [More details](LLM_finetuning.md/#2-full-partial-parameter-finetuning).

Given the combination of PEFT and FSDP, we would be able to fine tune a Meta Llama 3 8B model on multiple GPUs in one node or multi-node.

## Requirements
To run the examples, make sure to install the llama-recipes package and clone the github repository in order to use the provided [`finetuning.py`](../recipes/quickstart/finetuning/finetuning.py) script with torchrun (See [README.md](../README.md) for details).

**Please note that the llama_recipes package will install PyTorch 2.0.1 version, in case you want to run FSDP + PEFT, please make sure to install PyTorch nightlies.**

## How to run it

Get access to a machine with multiple GPUs ( in this case we tested with 4 A100 and A10s).
This runs with the `samsum_dataset` for summarization application by default.

**Multiple GPUs one node**:

**NOTE** please make sure to use PyTorch Nightlies for using PEFT+FSDP. Also, note that int8 quantization from bit&bytes currently is not supported in FSDP.

```bash

torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --model_name /path_of_model_folder/8B --use_peft --peft_method lora --output_dir Path/to/save/PEFT/model

```

The args used in the command above are:

* `--enable_fsdp` boolean flag to enable FSDP  in the script

* `--use_peft` boolean flag to enable PEFT methods in the script

* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`.

We use `torchrun` here to spawn multiple processes for FSDP.

## Flash Attention and Xformer Memory Efficient Kernels

Setting `use_fast_kernels` will enable using of Flash Attention or Xformer memory-efficient kernels based on the hardware being used. This would speed up the fine-tuning job. This has been enabled in `optimum` library from HuggingFace as a one-liner API, please read more [here](https://pytorch.org/blog/out-of-the-box-acceleration/).

```bash
torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --model_name /path_of_model_folder/8B --use_peft --peft_method lora --output_dir Path/to/save/PEFT/model --use_fast_kernels
```

### Fine-tuning using FSDP Only

If interested in running full parameter finetuning without making use of PEFT methods, please use the following command. Make sure to change the `nproc_per_node` to your available GPUs. This has been tested with `BF16` on 8xA100, 40GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8  examples/finetuning.py --enable_fsdp --model_name /path_of_model_folder/8B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --use_fast_kernels

```

### Fine-tuning using FSDP on 70B Model

If you are interested in running full parameter fine-tuning on the 70B model, you can enable `low_cpu_fsdp` mode as the following command. This option will load model on rank0 only before moving model to devices to construct FSDP. This can dramatically save cpu memory when loading large models like 70B (on a 8-gpu node, this reduces cpu memory from 2+T to 280G for 70B model). This has been tested with `BF16` on 16xA100, 80GB GPUs.

```bash

torchrun --nnodes 1 --nproc_per_node 8 examples/finetuning.py --enable_fsdp --low_cpu_fsdp --pure_bf16 --model_name /path_of_model_folder/70B --batch_size_training 1 --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned

```

**Multi GPU multi node**:

Here we use a slurm script to schedule a job with slurm over multiple nodes.

```bash

sbatch examples/multi_node.slurm
# Change the num nodes and GPU per nodes in the script before running.

```

## How to run with different datasets?

Currently 4 datasets are supported that can be found in [Datasets config file](../src/llama_recipes/configs/datasets.py).

* `grammar_dataset` : use this [notebook](../src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process theJfleg and C4 200M datasets for grammar checking.

* `alpaca_dataset` : to get this open source data please download the `aplaca.json` to `dataset` folder.

```bash
wget -P src/llama_recipes/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

* `samsum_dataset`

To run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset
torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp  --model_name /path_of_model_folder/8B --use_peft --peft_method lora --dataset grammar_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned  --pure_bf16 --output_dir Path/to/save/PEFT/model

# alpaca_dataset

torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp  --model_name /path_of_model_folder/8B --use_peft --peft_method lora --dataset alpaca_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --output_dir Path/to/save/PEFT/model


# samsum_dataset

torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --model_name /path_of_model_folder/8B --use_peft --peft_method lora --dataset samsum_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --output_dir Path/to/save/PEFT/model

```

## Where to configure settings?

* [Training config file](../src/llama_recipes/configs/training.py) is the main config file that helps to specify the settings for our run and can be found in [configs folder](../src/llama_recipes/configs/)

It lets us specify the training settings for everything from `model_name` to `dataset_name`, `batch_size` and so on. Below is the list of supported settings:

```python
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=False
    from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
```

* [Datasets config file](../src/llama_recipes/configs/datasets.py) provides the available options for datasets.

* [peft config file](../src/llama_recipes/configs/peft.py) provides the supported PEFT methods and respective settings that can be modified.

* [FSDP config file](../src/llama_recipes/configs/fsdp.py) provides FSDP settings such as:

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

## FLOPS Counting and Pytorch Profiling

To help with benchmarking effort, we are adding the support for counting the FLOPS during the fine-tuning process. You can achieve this by setting `--flop_counter` when launching your single/multi GPU fine-tuning. Use `--flop_counter_start` to choose which step to count the FLOPS. It is recommended to allow a warm-up stage before using the FLOPS counter.

Similarly, you can set `--use_profiler` flag and pass a profiling output path using `--profiler_dir` to capture the profile traces of your model using [PyTorch profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html). To get accurate profiling result, the pytorch profiler requires a warm-up stage and the current config is wait=1, warmup=2, active=3, thus the profiler will start the profiling after step 3 and will record the next 3 steps. Therefore, in order to use pytorch profiler, the --max-train-step has been greater than 6.  The pytorch profiler would be helpful for debugging purposes. However, the `--flop_counter` and `--use_profiler` can not be used in the same time to ensure the measurement accuracy.
