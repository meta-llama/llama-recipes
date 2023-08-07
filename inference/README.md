# Inference

This folder contains inference examples for Llama 2. So far, we have provided support for three methods of inference:

1. [inference script](inference.py) script provides support for Hugging Face accelerate, PEFT and FSDP fine tuned models.

2. [vLLM_inference.py](vLLM_inference.py) script takes advantage of vLLM's paged attention concept for low latency.

3. The [hf-text-generation-inference](hf-text-generation-inference/README.md) folder contains information on Hugging Face Text Generation Inference (TGI).

For more in depth information on inference including inference safety checks and examples, see the inference documentation [here](../docs/inference.md).

## System Prompt Update

### Observed Issue
We received feedback from the community on our prompt template and we are providing an update to reduce the false refusal rates seen. False refusals occur when the model incorrectly refuses to answer a question that it should, for example due to overly broad instructions to be cautious in how it provides responses.

### Updated approach
Based on evaluation and analysis, we recommend the removal of the system prompt as the default setting. [Pull request #104](https://github.com/facebookresearch/llama-recipes/pull/104)] removes the system prompt as the default option, but still provides an example to help enable experimentation for those using it.