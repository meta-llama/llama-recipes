# Meta Llama Guard demo
<!-- markdown-link-check-disable -->
Meta Llama Guard is a language model that provides input and output guardrails for LLM inference. For more details and model cards, please visit the [PurpleLlama](https://github.com/meta-llama/PurpleLlama) repository.

This [notebook](llama_guard_text_and_vision_inference.ipynb) shows how to load the models with the transformers library and how to customize the categories.

## Requirements
1. Access to Llama guard model weights on Hugging Face. To get access, follow the steps described in the top of the model card in [Hugging Face](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
2. Llama recipes package and its dependencies [installed](https://github.com/meta-llama/llama-recipes?tab=readme-ov-file#installing)
3. Pillow package installed

## Inference Safety Checker
When running the regular inference script with prompts, Meta Llama Guard will be used as a safety checker on the user prompt and the model output. If both are safe, the result will be shown, else a message with the error will be shown, with the word unsafe and a comma separated list of categories infringed. Meta Llama Guard is always loaded quantized using Hugging Face Transformers library with bitsandbytes.

In this case, the default categories are applied by the tokenizer, using the `apply_chat_template` method.

Use this command for testing with a quantized Llama model, modifying the values accordingly:

`python inference.py --model_name <path_to_regular_llama_model> --prompt_file <path_to_prompt_file> --enable_llamaguard_content_safety`

## Llama Guard 3 Finetuning & Customization
The safety categories in Llama Guard 3 can be tuned for specific application needs. Existing categories can be removed and new categories can be added to the taxonomy. The [Llama Guard Customization](./llama_guard_customization_via_prompting_and_fine_tuning.ipynb) notebook walks through the process.