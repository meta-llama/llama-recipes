## Llama-cookbook Getting Started

If you are new to developing with Meta Llama models, this is where you should start. This folder contains introductory-level notebooks across different techniques relating to Meta Llama.

* The [Build_with_Llama 3.2](./build_with_Llama_3_2.ipynb) notebook showcases a comprehensive walkthrough of the new capabilities of Llama 3.2 models, including multimodal use cases, function/tool calling, Llama Stack, and Llama on edge.
* The [Prompt_Engineering_with_Llama](./Prompt_Engineering_with_Llama.ipynb) notebook showcases the various ways to elicit appropriate outputs from Llama. Take this notebook for a spin to get a feel for how Llama responds to different inputs and generation parameters.
* The [inference](./inference/) folder contains scripts to deploy Llama for inference on server and mobile. See also [3p_integrations/vllm](../3p-integrations/vllm/) and [3p_integrations/tgi](../3p-integrations/tgi/) for hosting Llama on open-source model servers.
* The [RAG](./RAG/) folder contains a simple Retrieval-Augmented Generation application using Llama.
* The [finetuning](./finetuning/) folder contains resources to help you finetune Llama on your custom datasets, for both single- and multi-GPU setups. The scripts use the native llama-cookbook finetuning code found in [finetuning.py](../src/llama_cookbook/finetuning.py) which supports these features:
