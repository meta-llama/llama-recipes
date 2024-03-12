# Llama Recipes: Examples to get started using the Llama models from Meta

The 'llama-recipes' repository is a companion to the [Llama 2 model](https://github.com/facebookresearch/llama). The goal of this repository is to provide example scripts and notebooks to quickly get started with using the Llama models in a variety of use-cases, including fine-tuning for domain adaptation and building LLM-based applications with Llama 2 and other tools in the LLM ecosystem. The examples here showcase how to run Llama 2 locally, in the cloud, and on-prem.

## Table of Contents

- [Llama Recipes: Examples to get started using the Llama models from Meta](#llama-recipes-examples-to-get-started-using-the-llama-models-from-meta)
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
  - [News](#news)
  - [Contributing](#contributing)
  - [License](#license)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

#### PyTorch Nightlies
Some features (especially fine-tuning with FSDP + PEFT) currently require PyTorch nightlies to be installed. Please make sure to install the nightlies if you're using these features following [this guide](https://pytorch.org/get-started/locally/).

### Installing
Llama-recipes provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

#### Install with pip
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes
```

#### Install with optional dependencies
Llama-recipes offers the installation of optional packages. There are three optional dependency groups.
To run the unit tests we can install the required dependencies with:
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[tests]
```
For the vLLM example we need additional requirements that can be installed with:
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[vllm]
```
To use the sensitive topics safety checker install with:
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[auditnlg]
```
Optional dependencies can also be combines with [option1,option2].

#### Install from source
To install from source e.g. for development use these commands. We're using hatchling as our build backend which requires an up-to-date pip as well as setuptools package.
```
git clone git@github.com:facebookresearch/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .
```
For development and contributing to llama-recipes please install all optional dependencies:
```
git clone git@github.com:facebookresearch/llama-recipes.git
cd llama-recipes
pip install -U pip setuptools
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .[tests,auditnlg,vllm]
```


### Getting the Llama models
You can find Llama 2 models on Hugging Face hub [here](https://huggingface.co/meta-llama), where models with `hf` in the name are already converted to Hugging Face checkpoints so no further conversion is needed. The conversion step below is only for original model weights from Meta that are hosted on Hugging Face model hub as well.

#### Model conversion to Hugging Face
The recipes and notebooks in this folder are using the Llama 2 model definition provided by Hugging Face's transformers library.

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
[quickstart](./recipes/quickstart) | The "Hello World" of using Llama2, start here if you are new to using Llama2.
[benchmarks](./recipes/benchmarks)|Scripts to benchmark Llama 2 models inference on various backends
[code_llama](./recipes/code_llama)|Scripts to run inference with the Code Llama models
[evaluation](./recipes/evaluation)|Scripts to evaluate fine-tuned Llama2 models using `lm-evaluation-harness` from `EleutherAI`
[finetuning](./recipes/finetuning)|Scripts to finetune Llama2 on single-GPU and multi-GPU setups
[inference](./recipes/inference)|Scripts to deploy Llama2 for inference locally and using model servers
[llama_api_providers](./recipes/llama_api_providers)|Scripts to run inference on Llama via hosted endpoints
[responsible_ai](./recipes/responsible_ai)|Scripts to use PurpleLlama for safeguarding model outputs
[use_cases](./recipes/use_cases)|Scripts showing common applications of Llama2

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

## News
**[Update Feb. 26, 2024] We added examples to showcase OctoAI's cloud APIs for Llama2, CodeLlama, and LlamaGuard: including [PurpleLlama overview](./examples/Purple_Llama_OctoAI.ipynb), [hello Llama2 cloud](./demo_apps/OctoAI_API_examples/HelloLlamaCloud.ipynb), [getting to know Llama2](./demo_apps/OctoAI_API_examples/Getting_to_know_Llama.ipynb), [live search example](./demo_apps/OctoAI_API_examples/LiveData.ipynb), [Llama2 Gradio demo](./demo_apps/OctoAI_API_examples/Llama2_Gradio.ipynb), [Youtube video summarization](./demo_apps/OctoAI_API_examples/VideoSummary.ipynb), and [retrieval augmented generation overview](./demo_apps/OctoAI_API_examples/RAG_Chatbot_example/RAG_Chatbot_Example.ipynb)**.

**[Update Feb. 5, 2024] We added support for Code Llama 70B instruct in our example [inference script](./examples/code_llama/code_instruct_example.py). For details on formatting the prompt for Code Llama 70B instruct model please refer to [this document](./docs/inference.md)**.

**[Update Dec. 28, 2023] We added support for Llama Guard as a safety checker for our example inference script and also with standalone inference with an example script and prompt formatting. More details [here](./examples/llama_guard/README.md). For details on formatting data for fine tuning Llama Guard, we provide a script and sample usage [here](./src/llama_recipes/data/llama_guard/README.md).**

**[Update Dec 14, 2023] We recently released a series of Llama 2 demo apps [here](./demo_apps). These apps show how to run Llama (locally, in the cloud, or on-prem),  how to use Azure Llama 2 API (Model-as-a-Service), how to ask Llama questions in general or about custom data (PDF, DB, or live), how to integrate Llama with WhatsApp and Messenger, and how to implement an end-to-end chatbot with RAG (Retrieval Augmented Generation).**


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## License
See the License file [here](LICENSE) and Acceptable Use Policy [here](USE_POLICY.md)

