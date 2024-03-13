# Fine-tuning with Multi GPU
This recipe steps you through how to finetune a Llama 2 model on the text summarization task using the [samsum](https://huggingface.co/datasets/samsum) dataset on multiple GPUs in a single or across multiple nodes.


## Requirements
Ensure that you have installed the llama-recipes package ([details](../../README.md#installing)).

We will also need 2 packages:
1. [PEFT](https://github.com/huggingface/peft) to use parameter-efficient finetuning.
2. [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html) which helps us parallelize the training over multiple GPUs. [More details](./LLM_finetuning_overview.md#2-full-partial-parameter-finetuning).

> [!NOTE]  
> The llama-recipes package will install PyTorch 2.0.1 version. In case you want to use FSDP with PEFT for multi GPU finetuning, please install the PyTorch nightlies ([details](../../README.md#pytorch-nightlies))
>
> INT8 quantization is not currently supported in FSDP


## How to run it
Get access to a machine with multiple GPUs (in this case we tested with 4 A100 and A10s).

### With FSDP + PEFT

<details open>
<summary>Single-node Multi-GPU</summary>

    torchrun --nnodes 1 --nproc_per_node 4  finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --use_peft --peft_method lora --output_dir Path/to/save/PEFT/model

</details>

<details>
<summary>Multi-node Multi-GPU</summary>
Here we use a slurm script to schedule a job with slurm over multiple nodes.
    
    # Change the num nodes and GPU per nodes in the script before running.
    sbatch ./multi_node.slurm

</details>


We use `torchrun` to spawn multiple processes for FSDP.

The args used in the command above are:
* `--enable_fsdp` boolean flag to enable FSDP  in the script
* `--use_peft` boolean flag to enable PEFT methods in the script
* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`.


### With only FSDP
If interested in running full parameter finetuning without making use of PEFT methods, please use the following command. Make sure to change the `nproc_per_node` to your available GPUs. This has been tested with `BF16` on 8xA100, 40GB GPUs.

```bash
torchrun --nnodes 1 --nproc_per_node 8  finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --use_fast_kernels
```

### Using less CPU memory (FSDP on 70B model)

If you are running full parameter fine-tuning on the 70B model, you can enable `low_cpu_fsdp` mode as the following command. This option will load model on rank0 only before moving model to devices to construct FSDP. This can dramatically save cpu memory when loading large models like 70B (on a 8-gpu node, this reduces cpu memory from 2+T to 280G for 70B model). This has been tested with `BF16` on 16xA100, 80GB GPUs.

```bash
torchrun --nnodes 1 --nproc_per_node 8 finetuning.py --enable_fsdp --low_cpu_fsdp --pure_bf16 --model_name /patht_of_model_folder/70B --batch_size_training 1 --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned
```



## Running with different datasets
Currently 3 open source datasets are supported that can be found in [Datasets config file](../../src/llama_recipes/configs/datasets.py). You can also use your custom dataset (more info [here](./datasets/README.md)).

* `grammar_dataset` : use this [notebook](../../src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process the Jfleg and C4 200M datasets for grammar checking.

* `alpaca_dataset` : to get this open source data please download the `aplaca.json` to `dataset` folder.

```bash
wget -P ../../src/llama_recipes/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

* `samsum_dataset`

To run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset
torchrun --nnodes 1 --nproc_per_node 4  finetuning.py --enable_fsdp  --model_name /patht_of_model_folder/7B --use_peft --peft_method lora --dataset grammar_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned  --pure_bf16 --output_dir Path/to/save/PEFT/model

# alpaca_dataset

torchrun --nnodes 1 --nproc_per_node 4  finetuning.py --enable_fsdp  --model_name /patht_of_model_folder/7B --use_peft --peft_method lora --dataset alpaca_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --output_dir Path/to/save/PEFT/model


# samsum_dataset

torchrun --nnodes 1 --nproc_per_node 4  finetuning.py --enable_fsdp --model_name /patht_of_model_folder/7B --use_peft --peft_method lora --dataset samsum_dataset --save_model --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --pure_bf16 --output_dir Path/to/save/PEFT/model

```



## [TIP] Slow interconnect between nodes?
In case you are dealing with slower interconnect network between nodes, to reduce the communication overhead you can make use of `--hsdp` flag. 

HSDP (Hybrid sharding Data Parallel) helps to define a hybrid sharding strategy where you can have FSDP within `sharding_group_size` which can be the minimum number of GPUs you can fit your model and DDP between the replicas of the model specified by `replica_group_size`.

This will require to set the Sharding strategy in [fsdp config](../../src/llama_recipes/configs/fsdp.py) to `ShardingStrategy.HYBRID_SHARD` and specify two additional settings, `sharding_group_size` and `replica_group_size` where former specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model and latter specifies the replica group size, which is world_size/sharding_group_size.

```bash

torchrun --nnodes 4 --nproc_per_node 8 ./finetuning.py --enable_fsdp --low_cpu_fsdp --fsdp_config.pure_bf16 --model_name /patht_of_model_folder/70B --batch_size_training 1 --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --hsdp --sharding_group_size n --replica_group_size world_size/n

```



