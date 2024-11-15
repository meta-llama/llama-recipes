# Local Inference

## Hugging face setup
**Important Note**: Before running the inference, you'll need your Hugging Face access token, which you can get at your Settings page [here](https://huggingface.co/settings/tokens). Then run `huggingface-cli login` and copy and paste your Hugging Face access token to complete the login to make sure the scripts can download Hugging Face models if needed.

## Multimodal Inference
For Multi-Modal inference we have added [multi_modal_infer.py](multi_modal_infer.py) which uses the transformers library.

The way to run this would be:
```
python multi_modal_infer.py --image_path PATH_TO_IMAGE --prompt_text "Describe this image" --temperature 0.5 --top_p 0.8 --model_name "meta-llama/Llama-3.2-11B-Vision-Instruct"
```
---
## Multi-modal Inferencing Using gradio UI for inferencing
For multi-modal inferencing using gradio UI we have added [multi_modal_infer_gradio_UI.py](multi_modal_infer_gradio_UI.py) which used gradio and transformers library.

### Steps to Run

The way to run this would be:
- Ensure having proper access to llama 3.2 vision models, then run the command given below

```
python multi_modal_infer_gradio_UI.py --hf_token <your hf_token here>
```

## Text-only Inference
For local inference we have provided an [inference script](inference.py). Depending on the type of finetuning performed during training the [inference script](inference.py) takes different arguments.


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
cat <test_prompt_file> | python inference.py --model_name <training_config.output_dir> --use_auditnlg
# PEFT method
cat <test_prompt_file> | python inference.py --model_name <training_config.model_name> --peft_model <training_config.output_dir> --use_auditnlg
# prompt as parameter
python inference.py --model_name <training_config.output_dir> --prompt_file <test_prompt_file> --use_auditnlg
 ```
The  folder contains test prompts for summarization use-case:
```
samsum_prompt.txt
...
```

**Note on Llama version < 3.1**
The default padding token in [HuggingFace Tokenizer is `None`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L110). To use padding the padding token needs to be added as a special token to the tokenizer, which in this case requires to resize the token_embeddings as shown below:

```python
tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )
model.resize_token_embeddings(model.config.vocab_size + 1)
```
Padding would be required for batched inference. In this [example](inference.py), batch size = 1 so essentially padding is not required. However, we added the code pointer as an example in case of batch inference. For Llama version 3.1 use the special token `<|finetune_right_pad_id|> (128004)` for padding.

## Chat completion
The inference folder also includes a chat completion example, that adds built-in safety features in fine-tuned models to the prompt tokens. To run the example:

```bash
python chat_completion/chat_completion.py --model_name "PATH/TO/MODEL/7B/" --prompt_file chat_completion/chats.json  --quantization 8bit --use_auditnlg

```

## Flash Attention and Xformer Memory Efficient Kernels

Setting `use_fast_kernels` will enable using of Flash Attention or Xformer memory-efficient kernels based on the hardware being used. This would speed up inference when used for batched inputs. This has been enabled in `optimum` library from HuggingFace as a one-liner API, please read more [here](https://pytorch.org/blog/out-of-the-box-acceleration/).

```bash
python chat_completion/chat_completion.py --model_name "PATH/TO/MODEL/7B/" --prompt_file chat_completion/chats.json  --quantization 8bit --use_auditnlg --use_fast_kernels

python inference.py --model_name <training_config.output_dir> --peft_model <training_config.output_dir> --prompt_file <test_prompt_file> --use_auditnlg --use_fast_kernels

```

## Inference with FSDP checkpoints

In case you have fine-tuned your model with pure FSDP and saved the checkpoints with "SHARDED_STATE_DICT" as shown [here](../../../../src/llama_recipes/configs/fsdp.py), you can use this converter script to convert the FSDP Sharded checkpoints into HuggingFace checkpoints. This enables you to use the inference script normally as mentioned above.
**To convert the checkpoint use the following command**:

This is helpful if you have fine-tuned you model using FSDP only as follows:

```bash
torchrun --nnodes 1 --nproc_per_node 8  recipes/quickstart/finetuning/finetuning.py --enable_fsdp --model_name /path_of_model_folder/7B --dist_checkpoint_root_folder model_checkpoints --dist_checkpoint_folder fine-tuned --fsdp_config.pure_bf16
```
Then convert your FSDP checkpoint to HuggingFace checkpoints using:
```bash
 python -m llama_recipes.inference.checkpoint_converter_fsdp_hf --fsdp_checkpoint_path  PATH/to/FSDP/Checkpoints --consolidated_model_path PATH/to/save/checkpoints --HF_model_path_or_name PATH/or/HF/model_name

 # --HF_model_path_or_name specifies the HF Llama model name or path where it has config.json and tokenizer.json
 ```
By default, training parameter are saved in `train_params.yaml` in the path where FSDP checkpoints are saved, in the converter script we frist try to find the HugingFace model name used in the fine-tuning to load the model with configs from there, if not found user need to provide it.

Then run inference using:

```bash
python inference.py --model_name <training_config.output_dir> --prompt_file <test_prompt_file>

```

## Inference on large models like Meta Llama 405B
The FP8 quantized variants of Meta Llama (i.e. meta-llama/Meta-Llama-3.1-405B-FP8 and meta-llama/Meta-Llama-3.1-405B-Instruct-FP8) can be executed on a single node with 8x80GB H100 using the scripts located in this folder.
To run the unquantized Meta Llama 405B variants (i.e. meta-llama/Meta-Llama-3.1-405B and meta-llama/Meta-Llama-3.1-405B-Instruct) we need to use a multi-node setup for inference. The llama-recipes inference script currently does not allow multi-node inference. To run this model you can use vLLM with pipeline and tensor parallelism as showed in [this example](../../../3p_integrations/vllm/README.md).
