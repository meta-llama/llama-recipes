# Inference

For inference we have provided an [inference script](inference.py). Depending on the type of finetuning performed during training the [inference script](inference.py) takes different arguments.
To finetune all model parameters the output dir of the training has to be given as --model_name argument.
In the case of a parameter efficient method like lora the base model has to be given as --model_name and the output dir of the training has to be given as --peft_model argument.
Additionally, a prompt for the model in the form of a text file has to be provided. The prompt file can either be piped through standard input or given as --prompt_file parameter.

For other inference options, you can use the [vLLM_inference.py](vLLM_inference.py) script for vLLM or review the [hf-text-generation-inference](hf-text-generation-inference/README.md) folder for TGI.

For more information including inference safety checks, examples and other inference options available to you, see the inference documentation [here](../docs/inference.md).
