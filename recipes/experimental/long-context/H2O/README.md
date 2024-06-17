## Run Llama with H2O for long context inference

### Overview:

Heavy-Hitter Oracle (H2O) is an efficient inference framework of LLMs. During the generative inference of transfomers, the size of KV cache grows linearly with the sequence length (prompt length + generation length) during long context generation. And the size KV cache is usually significantly larger than the model parameters, contrains the inference throughput. H2O identifies the critical KV pairs and evicts other unnecessary ones, maintaining a small cache size thus improving the throughput.

Besides, LLMs usually have poor generation to long sequence during inference. H2O handles this issue by maintaining only heavy-hitter tokens and the most recent tokens. Incorporated with the positional rolling strategy (reassigning the position of each kv with the position in the kv cache instead of the original sequence), H2O can process sequence length much longer than the pretrained context window. Different from other approaches, like [Positional Interpolation](https://arxiv.org/abs/2306.15595), H2O is a KV cache policy and do not involve any training process for long context processing.

Current implementation supports llama-1/2/3, from 7B to 70B. Since H2O only maintains the most important KV pairs, it might missing some important information in the middle content for some knowlege-intensive tasks.

More details please refer to Paper: **https://arxiv.org/pdf/2306.14048**; Blog: **https://allenz.work/?p=11**.

**Note: this implementation is tested with transformers == 4.39.0**

### Evaluation on Summarization Tasks

The following example runs inference of Llama-2-7b and Meta-Llama-3-8B on XSUM summarization tasks. We're using `--enable_h2o_generation` to enable H2O algorithm that only keeps heavy-hitter and the local KV pairs. Use `--num_window_length `to decide the KV cache size. The number of local and heavy-hitter KV pairs equals to half of the --num_window_length (Option: the number of heavy-hitters can also be decided by `--num_heavy_hitter_tokens`) Also, use --enable_position_rolling to enable position rolling in the KV cache size that assign the positions in the KV cache instead of the ones in original sequences. Enabling positional rolling is important when sequence length exceeds the pretrained context windows, e.g., 8K in Llama-3.

```
python run_summarization.py \
--input-path data/summarization/xsum.jsonl \
--output-path summarization_output/xsum_h2o.jsonl \
--model-name meta-llama/Meta-Llama-3-8B \
--enable_h2o_generation 
```

##### **Results**

Expected results on XSUM (Rouge-2 score, the higher the better) from the above scripts on Llama-2/3 models. The sequence length of inputs are ~2k. Here we constrains the size of KV cache, allowing only n KVs to be write/read after the prefilling stage. n ranges from **64** to **full** where we maintain all the KV pairs. With 128 KVs, the performance can be matched as the full baseline (~2k KVs) while performance degradation is observed with 64 KVs. Also, maintaining a smaller KV cache reduces the I/O cost of KVs, thus we can achieve better throughput.

| KV Cache Size | 64     | 128    | 256    | 512    | 1024   | Full   |
| ------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Llama-2-7B    | 0.0439 | 0.1127 | 0.1148 | 0.1182 | 0.1170 | 0.1164 |
| Llama-2-13B   | 0.1180 | 0.1217 | 0.1243 | 0.1291 | 0.1302 | 0.1332 |
| Llama-3-8B    | 0.1107 | 0.1189 | 0.1200 | 0.1347 | 0.1290 | 0.1311 |

### One Demo on Streaming to "Infinite" Context Length

The following example demonstrates the generation process of "infinite" sequence length. We use MT-Bench data and generate the context sample-by-sample. The KV Cache will keep the KV pairs from the previous samples while maintain a fixed size. Results can be found on [Demo](https://allenz.work/?p=11) (Video 1).

```
# run with full cache
# expected results: 1) normal generation at the early stage; 2) performance collapse and generation slow down at the middle stage, because the sequence length exceeds the context window and the I/O cost of KV cache contrains the throughput; 3) OOM errors and stop.
bash src/streaming.sh full

# run with h2o
# expected results: normal generation at all stage.
# adjust the number of heavy-hitter tokens with --num_heavy_hitter_tokens and size of KV cache with --num_window_length in src/streaming.sh
bash src/streaming.sh h2o
```
