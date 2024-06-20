import torch
import argparse
import json
import os
import time
import re
import sys

from utils.streaming import load, download_url, load_jsonl, greedy_generate

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.llama import H2OLlamaForCausalLM
from utils.cache import Cache, HHCache, StaticCache


@torch.no_grad()
def streaming_inference_h2o(model, tokenizer, config, prompts, max_gen_len=1000, enable_h2o_generation=False):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]

        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )
        if enable_h2o_generation:
            space_needed = seq_len + max_gen_len
            past_key_values = HHCache.from_legacy_cache(config.num_window_length, config.num_heavy_hitter_tokens, past_key_values)
            past_key_values.evict_for_space(space_needed)
            past_key_values = past_key_values.to_legacy_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--model-name", type=str, default="lmsys/vicuna-13b-v1.5")

    parser.add_argument("--enable_h2o_generation", action='store_true')
    parser.add_argument("--num_heavy_hitter_tokens", type=int, default=128)
    parser.add_argument("--num_window_length", type=int, default=256)

    parser.add_argument("--enable_position_rolling", action='store_true')

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    model_name = args.model_name
    data_root = args.input_path

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if args.enable_h2o_generation:
        config.num_heavy_hitter_tokens = args.num_heavy_hitter_tokens
        config.num_window_length = args.num_window_length
        config.enable_position_rolling = args.enable_position_rolling
        model = H2OLlamaForCausalLM.from_pretrained(model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True,
            config=config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True,)

    test_filepath = os.path.join(data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")

    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
            data_root,
        )
        os.rename(os.path.join(data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    streaming_inference_h2o(model, tokenizer, config, prompts, enable_h2o_generation=args.enable_h2o_generation)

if __name__ == "__main__":
    main()