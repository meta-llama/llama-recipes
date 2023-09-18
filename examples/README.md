# Examples

This folder contains finetuning and inference examples for Llama 2.
For the full documentation on these examples please refer to [docs/inference.md](../docs/inference.md)

## Finetuning

Please refer to the main [README.md](../README.md) for information on how to use the [finetuning.py](./finetuning.py) script.
After installing the llama-recipes package through [pip](../README.md#installation) you can also invoke the finetuning in two ways:
```
python -m llama_recipes.finetuning <parameters>

python examnples/finetuning.py <parameters>
```
Please see [README.md](../README.md) for details.

## Inference 
So far, we have provide the following inference examples:

1. [inference script](./inference.py) script provides support for Hugging Face accelerate, PEFT and FSDP fine tuned models. It also demonstrates safety features to protect the user from toxic or harmful content.

2. [vllm/inference.py](./vllm/inference.py) script takes advantage of vLLM's paged attention concept for low latency.

3. The [hf_text_generation_inference](./hf_text_generation_inference/README.md) folder contains information on Hugging Face Text Generation Inference (TGI).

4. A [chat completion](./chat_completion/chat_completion.py) example highlighting the handling of chat dialogs.

5. [Code Llama](./code_llama/) folder which provides examples for [code completion](./code_llama/code_completion_example.py) and [code infilling](./code_llama/code_infilling_example.py).

For more in depth information on inference including inference safety checks and examples, see the inference documentation [here](../docs/inference.md).

**Note** The [sensitive topics safety checker](../src/llama_recipes/inference/safety_utils.py) utilizes AuditNLG which is an optional dependency. Please refer to installation section of the main [README.md](../README.md#install-with-optional-dependencies) for details.

**Note** The **vLLM** example requires additional dependencies. Please refer to installation section of the main [README.md](../README.md#install-with-optional-dependencies) for details.

## Train on custom dataset
To show how to train a model on a custom dataset we provide an example to generate a custom dataset in [custom_dataset.py](./custom_dataset.py).
The usage of the custom dataset is further described in the datasets [README](../docs/Dataset.md#training-on-custom-data).
