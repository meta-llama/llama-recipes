method=$1
if [[ ${method} == 'h2o' ]]; then
    python -u run_streaming.py \
        --input-path data \
        --model-name lmsys/vicuna-13b-v1.5 \
        --enable_h2o_generation \
        --num_heavy_hitter_tokens 2048 \
        --num_window_length 4096 \
        --enable_position_rolling
elif [[ ${method} == 'full' ]]; then
    python -u run_streaming.py \
        --input-path data \
        --model-name lmsys/vicuna-13b-v1.5
else
    echo 'unknown argment for method'
fi







