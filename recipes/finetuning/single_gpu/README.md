# Fine-tuning with Single GPU

This recipe steps you through how to finetune a Llama 2 model on the text summarization task using the [samsum](https://huggingface.co/datasets/samsum). The notebook uses parameter efficient finetuning (PEFT) and int8 quantization to finetune a 7B on a single GPU like an A10 with 24GB gpu memory.

There are two ways to do finetuning with a single GPU:
1. From the python notebook [](./peft_finetuning.ipynb)
2. From the python script [](../finetuning.py); the same script can be used for [multi GPU finetuning](../multigpu/) as well.
   
## Requirements

Ensure that you have installed the llama-recipes package ([details](../../../README.md#installing)).

> [!NOTE]  
> The llama-recipes package will install PyTorch 2.0.1 version. In case you want to use FSDP with PEFT for multi GPU finetuning, please install the PyTorch nightlies ([details](../../../README.md#pytorch-nightlies))

To run fine-tuning on a single GPU, we will make use of two packages:
1. [PEFT](https://github.com/huggingface/peft) to use parameter-efficient finetuning.
2. [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for int8 quantization.


## How to run it?

### Notebook
Execute each cell in a GPU-enabled environment.
### Script
```bash

python -m ../finetuning.py  --use_peft --peft_method lora --quantization --use_fp16 --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```
The args used in the command above are:

* `--use_peft` boolean flag to enable PEFT methods in the script

* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`.

* `--quantization` boolean flag to enable int8 quantization

> [!NOTE]  
> In case you are using a multi-GPU machine please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`.
>
> All the setting defined in [config files](src/llama_recipes/configs/) can be passed as args through CLI when running the script, there is no need to change from config files directly.

### How to run with different datasets?

Currently 4 datasets are supported that can be found in [Datasets config file](../../../src/llama_recipes/configs/datasets.py).

* `grammar_dataset` : use this [notebook](../src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process theJfleg and C4 200M datasets for grammar checking.

* `alpaca_dataset` : to get this open source data please download the `aplaca.json` to `ft_dataset` folder.

```bash
wget -P src/llama_recipes/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

* `samsum_dataset`

to run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset

python -m ../finetuning.py  --use_peft --peft_method lora --quantization  --dataset grammar_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

# alpaca_dataset

python -m ../finetuning.py  --use_peft --peft_method lora --quantization  --dataset alpaca_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model


# samsum_dataset

python -m ../finetuning.py  --use_peft --peft_method lora --quantization  --dataset samsum_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```

## Where to configure settings?

* [Training config file](../src/llama_recipes/configs/training.py) is the main config file that help to specify the settings for our run can be found in

It let us specify the training settings, everything from `model_name` to `dataset_name`, `batch_size` etc. can be set here. Below is the list of supported settings:

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
dataset = "samsum_dataset" # alpaca_dataset,grammar_dataset
peft_method: str = "lora" # None , llama_adapter, prefix
use_peft: bool=False
output_dir: str = "./ft-output"
freeze_layers: bool = False
num_freeze_layers: int = 1
quantization: bool = False
one_gpu: bool = False
save_model: bool = False
dist_checkpoint_root_folder: str="model_checkpoints"
dist_checkpoint_folder: str="fine-tuned"
save_optimizer: bool=False

```

* [Datasets config file](../src/llama_recipes/configs/datasets.py) provides the available options for datasets.

* [peft config file](../src/llama_recipes/configs/peft.py) provides the supported PEFT methods and respective settings that can be modified.

## Weights & Biases Experiment Tracking

You can enable [W&B](https://wandb.ai/) experiment tracking by using `use_wandb` flag as below. You can change the project name, entity and other `wandb.init` arguments in `wandb_config`.

```bash
python -m llama_recipes.finetuning  --use_peft --peft_method lora --quantization --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model --use_wandb
```
You'll be able to access a dedicated project or run link on [wandb.ai](https://wandb.ai) and see your dashboard like the one below. 
<div style="display: flex;">
    <img src="../../../docs/images/wandb_screenshot.png" alt="wandb screenshot" width="500" />
</div>