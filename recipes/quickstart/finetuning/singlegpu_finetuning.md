# Fine-tuning with Single GPU
This recipe steps you through how to finetune a Meta Llama 3 model on the text summarization task using the [samsum](https://huggingface.co/datasets/samsum) dataset on a single GPU.

These are the instructions for using the canonical [finetuning script](../../../src/llama_recipes/finetuning.py) in the llama-recipes package.


## Requirements

Ensure that you have installed the llama-recipes package ([details](../../../README.md#installing)).

To run fine-tuning on a single GPU, we will make use of two packages:
1. [PEFT](https://github.com/huggingface/peft) to use parameter-efficient finetuning.
2. [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for int8 quantization.


## How to run it?

**NOTE** To run the fine-tuning with `QLORA`, make sure to set `--peft_method lora` and `--quantization 4bit --quantization_config.quant_type nf4`.


```bash
FSDP_CPU_RAM_EFFICIENT_LOADING=1 python finetuning.py  --use_peft --peft_method lora --quantization 8bit --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model
```
The args used in the command above are:

* `--use_peft` boolean flag to enable PEFT methods in the script
* `--peft_method` to specify the PEFT method, here we use `lora` other options are `llama_adapter`, `prefix`.
* `--quantization` string flag to enable 8bit or 4bit quantization

> [!NOTE]
> In case you are using a multi-GPU machine please make sure to only make one of them visible using `export CUDA_VISIBLE_DEVICES=GPU:id`.


### How to run with different datasets?

Currently 3 open source datasets are supported that can be found in [Datasets config file](../../../src/llama_recipes/configs/datasets.py). You can also use your custom dataset (more info [here](./datasets/README.md)).

* `grammar_dataset` : use this [notebook](../../../src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb) to pull and process the Jfleg and C4 200M datasets for grammar checking.

* `alpaca_dataset` : to get this open source data please download the `alpaca.json` to `dataset` folder.


```bash
wget -P ../../../src/llama_recipes/datasets https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

* `samsum_dataset`

to run with each of the datasets set the `dataset` flag in the command as shown below:

```bash
# grammar_dataset

python -m finetuning.py  --use_peft --peft_method lora --quantization 8bit --dataset grammar_dataset --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model

# alpaca_dataset

python -m finetuning.py  --use_peft --peft_method lora --quantization 8bit  --dataset alpaca_dataset --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model


# samsum_dataset

python -m finetuning.py  --use_peft --peft_method lora --quantization 8bit  --dataset samsum_dataset --model_name /path_of_model_folder/8B --output_dir Path/to/save/PEFT/model

```

## FLOPS Counting and Pytorch Profiling

To help with benchmarking effort, we are adding the support for counting the FLOPS during the fine-tuning process. You can achieve this by setting `--flop_counter` when launching your single/multi GPU fine-tuning. Use `--flop_counter_start` to choose which step to count the FLOPS. It is recommended to allow a warm-up stage before using the FLOPS counter.

Similarly, you can set `--use_profiler` flag and pass a profiling output path using `--profiler_dir` to capture the profile traces of your model using [PyTorch profiler](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html). To get accurate profiling result, the pytorch profiler requires a warm-up stage and the current config is wait=1, warmup=2, active=3, thus the profiler will start the profiling after step 3 and will record the next 3 steps. Therefore, in order to use pytorch profiler, the --max-train-step has been greater than 6.  The pytorch profiler would be helpful for debugging purposes. However, the `--flop_counter` and `--use_profiler` can not be used in the same time to ensure the measurement accuracy.
