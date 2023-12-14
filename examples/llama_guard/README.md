# Llama Guard demo
<!-- markdown-link-check-disable -->
Llama Guard is a new experimental model that provides input and output guardrails for LLM deployments. For more details, please visit the main [repository](https://github.com/facebookresearch/PurpleLlama/tree/main/Llama-Guard).

This folder contains the files for the function used in the safety_checker when running in the inference script.

## Requirements
1. Llama guard model weights downloaded. To download, follow the steps shown [here](https://github.com/facebookresearch/PurpleLlama/tree/main/Llama-Guard#download)
2. Llama recipes dependencies installed 
3. A GPU with at least 21 GB of free RAM to load the 7B model. To run both Llama 2 7B and Llama Guard, multiple GPUS or a single one with additional memory is required.
<!-- markdown-link-check-enable -->
### Inference Safety Checker
When running the regular inference script with prompts, Llama Guard will be used as a safety checker on the user prompt and the model output. If both are safe, the result will be show, else a message with the error will be show, with the word unsafe and a comma separated list of categories infringed. As the model is not quantized, it requires more GPU than the direct examples, to load the desired Llama model for inference and the Llama Guard model for safety checks. Using Llama 2 7B quantized, this was able to be run in a machine with four A10G GPUs.
Use this command for testing with a quantized Llama model, modifying the values accordingly:

`RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 python examples/inference.py --model_name <path_to_regular_llama_model> --prompt_file <path_to_prompt_file> --quantization --enable_llamaguard_content_safety --llamaguard_model_name <path_to_mode>`



