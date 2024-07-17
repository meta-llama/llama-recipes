# Meta Llama Guard

Meta Llama Guard models provide input and output guardrails for LLM inference. For more details, please visit the main [repository](https://github.com/facebookresearch/PurpleLlama/).

**Note** Please find the right model on HF side [here](https://huggingface.co/meta-llama/Llama-Guard-3-8B).

### Running locally
The [llama_guard](llama_guard) folder contains the inference script to run Meta Llama Guard locally. Add test prompts directly to the [inference script](llama_guard/inference.py) before running it.

### Running on the cloud
The notebooks [Purple_Llama_Anyscale](Purple_Llama_Anyscale.ipynb) & [Purple_Llama_OctoAI](Purple_Llama_OctoAI.ipynb) contain examples for running Meta Llama Guard on cloud hosted endpoints.
