TOTAL_SECONDS=120
QPS_RATES=("1" "3" "5" "7" "9")

for QPS in ${QPS_RATES[@]}; do
    NUM_PROMPTS=$((TOTAL_SECONDS * QPS))
    echo "===== RUNNING NUM_PROMPTS = $NUM_PROMPTS QPS = $QPS ====="

    uv run benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name sonnet --sonnet-input-len 550 --sonnet-output-len 150 --dataset-path benchmarks/sonnet.txt \
        --num-prompts $NUM_PROMPTS --request-rate $QPS --save-result
done