# torchtune recipe for Meta Llama 3 finetuning

torchtune is a PyTorch-native library for easily authoring, fine-tuning and experimenting with LLMs.

torchtune provides:

- Native-PyTorch implementations of popular LLMs using composable and modular building blocks
- Easy-to-use and hackable training recipes for popular fine-tuning techniques (LoRA, QLoRA) - no trainers, no frameworks, just PyTorch!
- YAML configs for easily configuring training, evaluation, quantization or inference recipes
- Built-in support for many popular dataset formats and prompt templates to help you quickly get started with training

Here we will show the how to use torchtune to easily finetune meta Llama 3. For more usage details, please check the [torchtune repo](https://github.com/pytorch/torchtune/tree/main).

## Installation

**Step 1:** [Install PyTorch](https://pytorch.org/get-started/locally/). torchtune is tested with the latest stable PyTorch release as well as the preview nightly version.

**Step 2:** The latest stable version of torchtune is hosted on PyPI and can be downloaded with the following command:

```bash
pip install torchtune
```

To confirm that the package is installed correctly, you can run the following command:

```bash
tune --help
```

And should see the following output:

```bash
usage: tune [-h] {ls,cp,download,run,validate} ...

Welcome to the torchtune CLI!

options:
  -h, --help            show this help message and exit

...
```

You can also install the latest and greatest torchtune has to offer by [installing a nightly build](https://pytorch.org/torchtune/main/install.html).

&nbsp;

---


&nbsp;

## Meta Llama3 fine-tuning steps

torchtune supports fine-tuning for the Llama3 8B and 70B size models. It currently support LoRA, QLoRA and full fine-tune on a single GPU as well as LoRA and full fine-tune on multiple devices for the 8B model, and LoRA on multiple devices for the 70B model. For all the details, take a look at the [tutorial](https://pytorch.org/torchtune/main/tutorials/llama3.html).


**Note**: The Llama3 LoRA and QLoRA configs default to the instruct fine-tuned models.
This is because not all special token embeddings are initialized in the base 8B and 70B models.

In the initial experiments for Llama3-8B, QLoRA has a peak allocated memory of ``~9GB`` while LoRA on a single GPU has a peak allocated memory of ``~19GB``. To get started, you can use the default configs to kick off training.


### Torchtune repo download
We need to clone the torchtune repo to get the configs.

```bash
git clone git@github.com:pytorch/torchtune.git
cd torchtune/recipes/configs/
```

### Model download
Select which model you want to download, in this example we want to use meta-llama/Meta-Llama-3-8B.
```bash
tune download meta-llama/Meta-Llama-3-8B \
--output-dir /tmp/Meta-Llama-3-8B \
--hf-token <HF_TOKEN> \
```


> Tip: Set your environment variable `HF_TOKEN` or pass in `--hf-token` to the command in order to validate your access.
You can find your token at https://huggingface.co/settings/tokens

### Single GPU finetune

LoRA 8B

```bash
tune run lora_finetune_single_device --config llama3/8B_lora_single_device
```

QLoRA 8B

```bash
tune run lora_finetune_single_device --config llama3/8B_qlora_single_device
```

Full 8B

```bash
tune run full_finetune_single_device --config llama3/8B_full_single_device
```

### Multi GPU finetune

Full 8B

```bash
tune run --nproc_per_node 4 full_finetune_distributed --config llama3/8B_full
```

LoRA 8B

```bash
tune run --nproc_per_node 2 lora_finetune_distributed --config llama3/8B_lora
```

LoRA 70B

Note that the download command for the Meta-Llama3 70B model slightly differs from download commands for the 8B models. This is because it use the HuggingFace [safetensor](https://huggingface.co/docs/safetensors/en/index) model format to load the model. To download the 70B model, run
```bash
tune download meta-llama/Meta-Llama-3-70b --hf-token <> --output-dir /tmp/Meta-Llama-3-70b --ignore-patterns "original/consolidated*"
```

Then, a finetune can be kicked off:

```bash
tune run --nproc_per_node 8 lora_finetune_distributed --config recipes/configs/llama3/70B_lora.yaml
```

You can find a full list of all the Llama3 configs [here](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama3).


&nbsp;
