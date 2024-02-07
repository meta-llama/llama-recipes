# Llama Guard demo
<!-- markdown-link-check-disable -->
Llama Guard is a new experimental model that provides input and output guardrails for LLM deployments. For more details, please visit the main [repository](https://github.com/facebookresearch/PurpleLlama/tree/main/Llama-Guard).

This folder contains an example file to run Llama Guard inference directly. 

## Requirements
1. Access to Llama guard model weights on Hugging Face. To get access, follow the steps described [here](https://github.com/facebookresearch/PurpleLlama/tree/main/Llama-Guard#download)
2. Llama recipes package and it's dependencies [installed](https://github.com/albertodepaola/llama-recipes/blob/llama-guard-data-formatter-example/README.md#installation)
3. A GPU with at least 21 GB of free RAM to load both 7B models quantized.

## Llama Guard inference script
For testing, you can add User or User/Agent interactions into the prompts list and the run the script to verify the results. When the conversation has one or more Agent responses, it's considered of type agent. 


```
    prompts: List[Tuple[List[str], AgentType]] = [
        (["<Sample user prompt>"], AgentType.USER),

        (["<Sample user prompt>",
        "<Sample agent response>"], AgentType.AGENT),

        (["<Sample user prompt>",
        "<Sample agent response>",
        "<Sample user reply>",
        "<Sample agent response>",], AgentType.AGENT),

    ]
```
The complete prompt is built with the `build_prompt` function, defined in [prompt_format.py](../../src/llama_recipes/inference/prompt_format.py). The file contains the default Llama Guard  categories. These categories can adjusted and new ones can be added, as described in the [research paper](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/), on section 4.5 Studying the adaptability of the model.
<!-- markdown-link-check-enable -->

To run the samples, with all the dependencies installed, execute this command:

`python examples/llama_guard/inference.py`

This is the output:

```
['<Sample user prompt>']
> safe

==================================

['<Sample user prompt>', '<Sample agent response>']
> safe

==================================

['<Sample user prompt>', '<Sample agent response>', '<Sample user reply>', '<Sample agent response>']
> safe

==================================
```

## Inference Safety Checker
When running the regular inference script with prompts, Llama Guard will be used as a safety checker on the user prompt and the model output. If both are safe, the result will be shown, else a message with the error will be shown, with the word unsafe and a comma separated list of categories infringed. Llama Guard is always loaded quantized using Hugging Face Transformers library.

In this case, the default categories are applied by the tokenizer, using the `apply_chat_template` method.

Use this command for testing with a quantized Llama model, modifying the values accordingly:

`python examples/inference.py --model_name <path_to_regular_llama_model> --prompt_file <path_to_prompt_file> --quantization --enable_llamaguard_content_safety`



