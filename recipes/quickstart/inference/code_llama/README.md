# Code Llama

Code llama was recently released with three flavors, base-model that support multiple programming languages, Python fine-tuned model and an instruction fine-tuned and aligned variation of Code Llama, please read more [here](https://ai.meta.com/blog/code-llama-large-language-model-coding/). Also note that the Python fine-tuned model and 34B models are not trained on infilling objective, hence can not be used for infilling use-case.

Find the scripts to run Code Llama, where there are two examples of running code completion and infilling.

**Note** Please find the right model on HF [here](https://huggingface.co/models?search=meta-llama%20codellama). 

Make sure to install Transformers from source for now

```bash

pip install git+https://github.com/huggingface/transformers

```

To run the code completion example:

```bash

python code_completion_example.py --model_name MODEL_NAME  --prompt_file code_completion_prompt.txt --temperature 0.2 --top_p 0.9

```

To run the code infilling example:

```bash

python code_infilling_example.py --model_name MODEL_NAME --prompt_file code_infilling_prompt.txt --temperature 0.2 --top_p 0.9

```
To run the 70B Instruct model example run the following (you'll need to enter the system and user prompts to instruct the model):

```bash

python code_instruct_example.py --model_name codellama/CodeLlama-70b-Instruct-hf --temperature 0.2 --top_p 0.9

```
You can learn more about the chat prompt template [on HF](https://huggingface.co/meta-llama/CodeLlama-70b-Instruct-hf#chat-prompt) and [original Code Llama repository](https://github.com/meta-llama/codellama/blob/main/README.md#fine-tuned-instruction-models). HF tokenizer has already taken care of the chat template as shown in this example. 
