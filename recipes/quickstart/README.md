## Llama-Recipes Quickstart

If you are new to developing with Meta Llama models, this is where you should start. This folder contains introductory-level notebooks across different techniques relating to Meta Llama.

* The [](./Running_Llama3_Anywhere/) notebooks demonstrate how to run Llama inference across Linux, Mac and Windows platforms using the appropriate tooling.
* The [](./prompt_engineering/Prompt_Engineering_with_Llama_3.ipynb) notebook showcases the various ways to elicit appropriate outputs from Llama. Take this notebook for a spin to get a feel for how Llama responds to different inputs and generation parameters.
* The [](./inference/) folder contains scripts to deploy Llama for inference on server and mobile. See also [](../3p_integrations/vllm/) and [](../3p_integrations/tgi/) for hosting Llama on open-source model servers.
* The [](./RAG/) folder contains a simple Retrieval-Augmented Generation application using Llama 3.
* The [](./finetuning/) folder contains resources to help you finetune Llama 3 on your custom datasets, for both single- and multi-GPU setups. The scripts use the native llama-recipes finetuning code found in [](../../src/llama_recipes/finetuning.py) which supports these features:

| Feature                                        |   |
| ---------------------------------------------- | - |
| HF support for finetuning                      | ✅ |
| Deferred initialization ( meta init)           | ✅ |
| HF support for inference                       | ✅ |
| Low CPU mode for multi GPU                     | ✅ |
| Mixed precision                                | ✅ |
| Single node quantization                       | ✅ |
| Flash attention                                | ✅ |
| PEFT                                           | ✅ |
| Activation checkpointing FSDP                  | ✅ |
| Hybrid Sharded Data Parallel (HSDP)            | ✅ |
| Dataset packing & padding                      | ✅ |
| BF16 Optimizer ( Pure BF16)                    | ✅ |
| Gradient accumulation                          | ✅ |
| CPU offloading                                 | ✅ |
| FSDP checkpoint conversion to HF for inference | ✅ |
| W&B experiment tracker                         | ✅ |
