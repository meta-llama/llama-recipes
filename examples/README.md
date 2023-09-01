# Examples

This folder contains finetuning and inference examples for Llama 2.

## Finetuning

Please refer to the main [README.md](../README.md) for information on how to use the [finetuning.py](./finetuning.py) script.
After installing the llama-recipes package through [pip](../README.md#installation) you can also invoke the finetuning in two ways:
```
python -m llama_recipes.finetuning <parameters>

python examnples/finetuning.py <parameters>
```
Please see [README.md](../README.md) for details.

## Inference 
So far, we have provided support for three methods of inference:

1. [inference script](./inference.py) script provides support for Hugging Face accelerate, PEFT and FSDP fine tuned models.

2. [vllm/inference.py](./vllm/inference.py) script takes advantage of vLLM's paged attention concept for low latency.

3. The [hf_text_generation_inference](./hf_text_generation_inference/README.md) folder contains information on Hugging Face Text Generation Inference (TGI).

For more in depth information on inference including inference safety checks and examples, see the inference documentation [here](../docs/inference.md).

**Note** The **vLLM** example requires additional dependencies. Please refer to installation section of the main [README.md](../README.md#install-with-optional-dependencies) for details