# Inference

For inference we have provided an [inference script](../inference/inference.py). Depending on the type of finetuning performed during training the [inference script](../inference/inference.py) takes different arguments.
To finetune all model parameters the output dir of the training has to be given as --model_name argument.
In the case of a parameter efficient method like lora the base model has to be given as --model_name and the output dir of the training has to be given as --peft_model argument.
Additionally, a prompt for the model in the form of a text file has to be provided. The prompt file can either be piped through standard input or given as --prompt_file parameter.

**Content Safety**
The inference script also supports safety checks for both user prompt and model outputs. In particular, we use two packages, [AuditNLG](https://github.com/salesforce/AuditNLG/tree/main) and [Azure content safety](https://pypi.org/project/azure-ai-contentsafety/1.0.0b1/).

**Note**
If using Azure content Safety, please make sure to get the endpoint and API key as described [here](https://pypi.org/project/azure-ai-contentsafety/1.0.0b1/) and add them as  the following environment variables,`CONTENT_SAFETY_ENDPOINT` and `CONTENT_SAFETY_KEY`.

Examples:

 ```bash
# Full finetuning of all parameters
cat <test_prompt_file> | python inference/inference.py --model_name <training_config.output_dir> --use_auditnlg
# PEFT method
cat <test_prompt_file> | python inference/inference.py --model_name <training_config.model_name> --peft_model <training_config.output_dir> --use_auditnlg
# prompt as parameter
python inference/inference.py --model_name <training_config.output_dir> --prompt_file <test_prompt_file> --use_auditnlg
 ```
The inference folder contains test prompts for summarization use-case:
```
inference/samsum_prompt.txt
...
```

**Chat completion**
The inference folder also includes a chat completion example, that adds built-in safety features in fine-tuned models to the prompt tokens. To run the example:

```bash
python inference/chat_completion.py --model_name "PATH/TO/MODEL/7B/" --prompt_file inference/chats.json  --quantization --use_auditnlg

```

## Other Inference Options

Alternate inference options include:

[**vLLM**](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html):
To use vLLM you will need to install it using the instructions [here](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#installation).
Once installed, you can use the vLLM_ineference.py script provided [here](../inference/vLLM_inference.py).

Below is an example of how to run the vLLM_inference.py script found within the inference folder.

``` bash
python vLLM_inference.py --model_name <PATH/TO/MODEL/7B>
```

[**TGI**](https://github.com/huggingface/text-generation-inference): Text Generation Inference (TGI) is another inference option available to you. For more information on how to set up and use TGI see [here](../inference/hf-text-generation-inference/README.md).
