# Serving a fine tuned Llama model with HuggingFace text-generation-inference server

This document shows how to serve a fine tuned Llama mode with HuggingFace's text-generation-inference server. This option is currently only available for models that were trained using the LoRA method or without using the `--use_peft` argument.

## Step 0: Merging the weights (Only required if LoRA method was used) 

In case the model was fine tuned with LoRA method we need to merge the weights of the base model with the adapter weight. For this we can use the script `merge_lora_weights.py` which is located in the same folder as this README file.

The script takes the base model, the peft weight folder as well as an output as arguments:

```
python -m llama_recipes.inference.hf_text_generation_inference.merge_lora_weights --base_model llama-7B --peft_model ft_output --output_dir data/merged_model_output
```

## Step 1: Serving the model
Subsequently, the model can be served using the docker container provided by [hf text-generation-inference](https://github.com/huggingface/text-generation-inference) started from the main directory of this repository:

```bash
model=/data/merged_model_output
num_shard=2
volume=$PWD/inference/hf-text-generation-inference/data
docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model --num-shard $num_shard
```

The num_shard argument determines the number of GPUs the model should be sharded on.

## Step 2: Running inference
After the loading of the model shards completed an inference can be executed by using one of the following commands:

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17}}' \
    -H 'Content-Type: application/json'
# OR for streaming inference
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17}}' \
    -H 'Content-Type: application/json'
```

When performing inference on the chat-based model, ensure that the input prompt adheres to a specific structure that includes tags such as `<s>..</s>`, `[INST] .. [/INST]`, and optional `<<SYS>>` for system prompts. The structure of the input prompt should be as follows:

```
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
```
For an in-depth explanation and further guidance, please refer to the blog [how-to-prompt-llama-2](https://huggingface.co/blog/llama2#how-to-prompt-llama-2).

Additionally, you may consult the sample script  [hf_tgi_inference.py](./hf_tgi_inference.py) for comprehensive instructions on executing inference with the chat-based model.
