CUDA_VISIBLE_DEVICES=$1 python -u generate.py \
--input-path data/summarization/xsum.jsonl \
--output-path xsum_baseline.jsonl \
--model-name meta-llama/Llama-2-7b-hf 


# CUDA_VISIBLE_DEVICES=$1 python -u generate.py \
# --input-path data/summarization/xsum.jsonl \
# --output-path xsum_h2o.jsonl \
# --model-name meta-llama/Llama-2-7b-hf \
# --enable_h2o_generation \
# --num_heavy_hitter_tokens 256 \
# --num_local_windows 256
