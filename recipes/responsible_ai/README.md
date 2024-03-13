# Llama Guard

Llama Guard is a new experimental model that provides input and output guardrails for LLM deployments. For more details, please visit the main [repository](https://github.com/facebookresearch/PurpleLlama/tree/main/Llama-Guard).

**Note** Please find the right model on HF side [here](https://huggingface.co/meta-llama/LlamaGuard-7b). 

### Running locally
The [llama_guard](llama_guard) folder contains the inference script to run Llama Guard locally. Add test prompts directly to the [inference script](llama_guard/inference.py) before running it.

### Running on the cloud
The notebooks [Purple_Llama_Anyscale](Purple_Llama_Anyscale.ipynb) & [Purple_Llama_OctoAI](Purple_Llama_OctoAI.ipynb) contain examples for running Llama Guard on cloud hosted endpoints.