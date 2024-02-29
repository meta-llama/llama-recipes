# Convert Hugging Face llama weights to official llama consolidated format

This is the reverse conversion for `convert_llama_weights_to_hf.py` script from the transformer package.

## Step 0: Convert to consolidated format
- Create an output directory for the converted weights, such as `test70B`.
- Copy file params.json from the official llama download into that directory.
- Run the conversion script. `model-path` can be a Hugging Face hub model or a local hf model directory.
```
python -m llama_recipes.tools.convert_hf_weights_to_llama --model-path meta-llama/Llama-2-70b-chat-hf --output-dir test70B --model-size 70B
```

## Step 1: Run inference
Checkout the official llama inference [repo](https://github.com/facebookresearch/llama). Test using chat or text completion.
```
torchrun --nproc_per_node 8 example_chat_completion.py --ckpt_dir ./test70B --tokenizer_path ${llama_2_dir}/tokenizer.model
```

For validation, please compare the converted weights with official llama 2 weights
```
python compare_llama_weights.py test70B ${llama_2_70b_chat_dir}
```
