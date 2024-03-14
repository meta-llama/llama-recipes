# Fine-tuning with Single GPU
This recipe steps you through how to finetune a Llama 2 model on the text summarization task using the [samsum](https://huggingface.co/datasets/samsum) dataset on a single GPU.

These are the instructions for using the canonical [finetuning script](../../src/llama_recipes/finetuning.py) in the llama-recipes package.


## Requirements

Ensure that you have installed the llama-recipes package ([details](../../README.md#installing)).

To run fine-tuning on a single GPU, we will make use of two packages:
1. [PEFT](https://github.com/huggingface/peft) to use parameter-efficient finetuning.
2. [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for int8 quantization.


## How to run it?

```bash
python -m finetuning.py  --use_peft --peft_method lora --quantization --use_fp16 --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model
```
The args used in the command above are:

* `--use_peft` boolean flag to enable PEFT methods in the script
* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`.
* `--quantization` boolean flag to enable int8 quantization

> [!NOTE]  
> In case you are using a multi-GPU machine please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`.

 
### How to run with different datasets?

Currently 3 open source datasets are supported that can be found in [Datasets config file](../../src/llama_recipes/configs/datasets.py). You can also use your custom dataset (more info [here](./datasets/README.md)).

* `grammar_dataset` : use this [notebook](../../src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process the Jfleg and C4 200M datasets for grammar checking.

* `alpaca_dataset` : to get this open source data please download the `aplaca.json` to `dataset` folder.


```bash
wget -P ../../src/llama_recipes/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

* `samsum_dataset`

to run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammer_dataset

python -m finetuning.py  --use_peft --peft_method lora --quantization  --dataset grammar_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

# alpaca_dataset

python -m finetuning.py  --use_peft --peft_method lora --quantization  --dataset alpaca_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model


# samsum_dataset

python -m finetuning.py  --use_peft --peft_method lora --quantization  --dataset samsum_dataset --model_name /patht_of_model_folder/7B --output_dir Path/to/save/PEFT/model

```
