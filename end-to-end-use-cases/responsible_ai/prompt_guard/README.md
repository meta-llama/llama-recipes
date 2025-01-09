# Prompt Guard demo
<!-- markdown-link-check-disable -->
Prompt Guard is a classifier model that provides input guardrails for LLM inference, particularly against *prompt attacks. For more details and model cards, please visit the main repository, [Meta Prompt Guard](https://github.com/meta-llama/PurpleLlama/tree/main/Prompt-Guard)

This folder contains an example file to run inference with a locally hosted model, either using the Hugging Face Hub or a local path. It also contains a comprehensive demo demonstrating the scenarios in which the model is effective and a script for fine-tuning the model.

This is a very small model and inference and fine-tuning are feasible on local CPUs.

## Requirements
1. Access to Prompt Guard model weights on Hugging Face. To get access, follow the steps described [here](https://github.com/facebookresearch/PurpleLlama/tree/main/Prompt-Guard#download)
2. Llama recipes package and it's dependencies [installed](https://github.com/meta-llama/llama-recipes?tab=readme-ov-file#installing)
