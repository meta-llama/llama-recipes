# Inference

This folder contains inference examples for Llama 2. So far, we have provided support for three methods of inference:

1. [inference script](inference.py) script provides support for Hugging Face accelerate, PEFT and FSDP fine tuned models.

2. [vLLM_inference.py](vLLM_inference.py) script takes advantage of vLLM's paged attention concept for low latency.

3. The [hf-text-generation-inference](hf-text-generation-inference/README.md) folder contains information on Hugging Face Text Generation Inference (TGI).

For more in depth information on inference including inference safety checks and examples, see the inference documentation [here](../docs/inference.md).
