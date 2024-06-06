# Llama Recipes: Examples to get started using the Llama models from Meta
<!-- markdown-link-check-disable -->
The 'llama-recipes' repository is a companion to the [Meta Llama 3](https://github.com/meta-llama/llama3) models. The goal of this repository is to provide a scalable library for fine-tuning Meta Llama models, along with some example scripts and notebooks to quickly get started with using the models in a variety of use-cases, including fine-tuning for domain adaptation and building LLM-based applications with Meta Llama and other tools in the LLM ecosystem. The examples here showcase how to run Meta Llama locally, in the cloud, and on-prem. [Meta Llama 2](https://github.com/meta-llama/llama) is also supported in this repository. We highly recommend everyone to utilize [Meta Llama 3](https://github.com/meta-llama/llama3) due to its enhanced capabilities.

<!-- markdown-link-check-enable -->
> [!IMPORTANT]
> Meta Llama 3 has a new prompt template and special tokens (based on the tiktoken tokenizer).
> | Token | Description |
> |---|---|
> `<\|begin_of_text\|>` | This is equivalent to the BOS token. |
> `<\|end_of_text\|>` | This is equivalent to the EOS token. For multiturn-conversations it's usually unused. Instead, every message is terminated with `<\|eot_id\|>` instead.|
> `<\|eot_id\|>` | This token signifies the end of the message in a turn i.e. the end of a single message by a system, user or assistant role as shown below.|
> `<\|start_header_id\|>{role}<\|end_header_id\|>` | These tokens enclose the role for a particular message. The possible roles can be: system, user, assistant. |
>
> A multiturn-conversation with Meta Llama 3 follows this prompt template:
> ```
> <|begin_of_text|><|start_header_id|>system<|end_header_id|>
>
> {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>
>
> {{ user_message_1 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
>
> {{ model_answer_1 }}<|eot_id|><|start_header_id|>user<|end_header_id|>
>
> {{ user_message_2 }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
> ```
> Each message gets trailed by an `<|eot_id|>` token before a new header is started, signaling a role change.
>
> More details on the new tokenizer and prompt template can be found [here](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3#special-tokens-used-with-meta-llama-3).
>
> [!NOTE]
> The llama-recipes repository was recently refactored to promote a better developer experience of using the examples. Some files have been moved to new locations. The `src/` folder has NOT been modified, so the functionality of this repo and package is not impacted.
>
> Make sure you update your local clone by running `git pull origin main`

## Table of Contents

- [Llama Recipes: Examples to get started using the Meta Llama models from Meta](#llama-recipes-examples-to-get-started-using-the-llama-models-from-meta)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
      - [PyTorch Nightlies](#pytorch-nightlies)
    - [Installing](#installing)
      - [Install with pip](#install-with-pip)
      - [Install with optional dependencies](#install-with-optional-dependencies)
      - [Install from source](#install-from-source)
    - [Getting the Llama models](#getting-the-llama-models)
      - [Model conversion to Hugging Face](#model-conversion-to-hugging-face)
  - [Repository Organization](#repository-organization)
    - [`recipes/`](#recipes)
    - [`src/`](#src)
  - [Contributing](#contributing)
  - [License](#license)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

#### PyTorch Nightlies
If you want to use PyTorch nightlies instead of the stable release, go to [this guide](https://pytorch.org/get-started/locally/) to retrieve the right `--extra-index-url URL` parameter for the `pip install` commands on your platform.

### Installing
Llama-recipes provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

> [!NOTE]
> Ensure you use the correct CUDA version (from `nvidia-smi`) when installing the PyTorch wheels. Here we are using 11.8 as `cu118`.
> H100 GPUs work better with CUDA >12.0

#### Install with pip
```
pip install llama-recipes
```

#### Install with optional dependencies
Llama-recipes offers the installation of optional packages. There are three optional dependency groups.
To run the unit tests we can install the required dependencies with:
```
pip install llama-recipes[tests]
```
For the vLLM example we need additional requirements that can be installed with:
```
pip install llama-recipes[vllm]
```
To use the sensitive topics safety checker install with:
```
pip install llama-recipes[auditnlg]
```
Optional dependencies can also be combines with [option1,option2].

#### Install from source
To install from source e.g. for development use these commands. We're using hatchling as our build backend which requires an up-to-date pip as well as setuptools package.
```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install -e .
```
For development and contributing to llama-recipes please install all optional dependencies:
```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install -e .[tests,auditnlg,vllm]
```


### Getting the Meta Llama models
You can find Meta Llama models on Hugging Face hub [here](https://huggingface.co/meta-llama), **where models with `hf` in the name are already converted to Hugging Face checkpoints so no further conversion is needed**. The conversion step below is only for original model weights from Meta that are hosted on Hugging Face model hub as well.

#### Model conversion to Hugging Face
The recipes and notebooks in this folder are using the Meta Llama model definition provided by Hugging Face's transformers library.

Given that the original checkpoint resides under models/7B you can install all requirements and convert the checkpoint with:

```bash
## Install Hugging Face Transformers from source
pip freeze | grep transformers ## verify it is version 4.31.0 or higher

git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```



## Repository Organization
Most of the code dealing with Llama usage is organized across 2 main folders: `recipes/` and `src/`.

### `recipes/`

Contains examples are organized in folders by topic:
| Subfolder | Description |
|---|---|
[quickstart](./recipes/quickstart) | The "Hello World" of using Llama, start here if you are new to using Llama.
[finetuning](./recipes/finetuning)|Scripts to finetune Llama on single-GPU and multi-GPU setups
[inference](./recipes/inference)|Scripts to deploy Llama for inference locally and using model servers
[use_cases](./recipes/use_cases)|Scripts showing common applications of Meta Llama3
[responsible_ai](./recipes/responsible_ai)|Scripts to use PurpleLlama for safeguarding model outputs
[llama_api_providers](./recipes/llama_api_providers)|Scripts to run inference on Llama via hosted endpoints
[benchmarks](./recipes/benchmarks)|Scripts to benchmark Llama models inference on various backends
[code_llama](./recipes/code_llama)|Scripts to run inference with the Code Llama models
[evaluation](./recipes/evaluation)|Scripts to evaluate fine-tuned Llama models using `lm-evaluation-harness` from `EleutherAI`

### `src/`

Contains modules which support the example recipes:
| Subfolder | Description |
|---|---|
| [configs](src/llama_recipes/configs/) | Contains the configuration files for PEFT methods, FSDP, Datasets, Weights & Biases experiment tracking. |
| [datasets](src/llama_recipes/datasets/) | Contains individual scripts for each dataset to download and process. Note |
| [inference](src/llama_recipes/inference/) | Includes modules for inference for the fine-tuned models. |
| [model_checkpointing](src/llama_recipes/model_checkpointing/) | Contains FSDP checkpoint handlers. |
| [policies](src/llama_recipes/policies/) | Contains FSDP scripts to provide different policies, such as mixed precision, transformer wrapping policy and activation checkpointing along with any precision optimizer (used for running FSDP with pure bf16 mode). |
| [utils](src/llama_recipes/utils/) | Utility files for:<br/> - `train_utils.py` provides training/eval loop and more train utils.<br/> - `dataset_utils.py` to get preprocessed datasets.<br/> - `config_utils.py` to override the configs received from CLI.<br/> - `fsdp_utils.py` provides FSDP  wrapping policy for PEFT methods.<br/> - `memory_utils.py` context manager to track different memory stats in train loop. |


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License
<!-- markdown-link-check-disable -->

See the License file for Meta Llama 3 [here](https://llama.meta.com/llama3/license/) and Acceptable Use Policy [here](https://llama.meta.com/llama3/use-policy/)

See the License file for Meta Llama 2 [here](https://llama.meta.com/llama2/license/) and Acceptable Use Policy [here](https://llama.meta.com/llama2/use-policy/)
<!-- markdown-link-check-enable -->
