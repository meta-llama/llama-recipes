# Llama Cookbook: The Official Guide to building with Llama Models

> Note: We recently did a refactor of the repo, [archive-main](https://github.com/meta-llama/llama-recipes/tree/archive-main) is a snapshot branch from before the refactor

Welcome to the official repository for helping you get started with [inference](./getting-started/inference/), [fine-tuning](./getting-started/finetuning) and [end-to-end use-cases](./end-to-end-use-cases) of building with the Llama Model family.

The examples cover the most popular community approaches, popular use-cases and the latest Llama 3.2 Vision and Llama 3.2 Text, in this repository. 

> [!TIP]
> Popular getting started links:
> * [Build with Llama Notebook](./getting-started/build_with_Llama_3_2.ipynb)
> * [Multimodal Inference with Llama 3.2 Vision](./getting-started/inference/local_inference/README.md#multimodal-inference)
> * [Inference on Llama Guard 1B + Multimodal inference on Llama Guard 11B-Vision](./end-to-end-use-cases/responsible_ai/llama_guard/

> [!TIP]
> Popular end to end recipes:
> * [Finetune Llama 3.2 Vision](./getting-started/finetuning/finetune_vision_model.md)llama_guard_text_and_vision_inference.ipynb)

> [!NOTE]
> Llama 3.2 follows the same prompt template as Llama 3.1, with a new special token `<|image|>` representing the input image for the multimodal models.
> 
> More details on the prompt templates for image reasoning, tool-calling and code interpreter can be found [on the documentation website](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_2).


## Repository Structure:

- [3P Integrations](./3p-integrations): Getting Started Recipes and End to End Use-Cases from various Llama providers
- [End to End Use Cases](./end-to-end-use-cases): As the name suggests, spanning various domains and applications
- [Getting Started](./getting-started/): Reference for inferencing, fine-tuning and RAG examples

## FAQ: 

- Q: Some links are broken/folders are missing: 

A: We recently did a refactor of the repo, [archive-main](https://github.com/meta-llama/llama-recipes/tree/archive-main) is a snapshot branch from before the refactor

- Where can we find details about the latest models?

A: Official [Llama models website](https://www.llama.com)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License
<!-- markdown-link-check-disable -->

See the License file for Meta Llama 3.2 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md)

See the License file for Meta Llama 3.1 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md)

See the License file for Meta Llama 3 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3/USE_POLICY.md)

See the License file for Meta Llama 2 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama2/USE_POLICY.md)
<!-- markdown-link-check-enable -->
