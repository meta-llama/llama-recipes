CUDA_VISIBLE_DEVICES=2 python -u run_needle_haystack_test.py \
--input-path data/needle_test/Huggingface \
--output-path needle_test_results/huggingface/llama-3-8b-instruct-h2o-4096/ \
--model-name meta-llama/Meta-Llama-3-8B-Instruct \
--enable_h2o_generation \
--num_window_length 4096 \
--num_heavy_hitter_tokens 2048