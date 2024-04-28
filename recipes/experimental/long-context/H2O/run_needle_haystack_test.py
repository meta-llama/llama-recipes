import os
import tqdm
import glob
import json
import copy
import math

import torch
import logging
import argparse

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.llama import H2OLlamaForCausalLM

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")

    parser.add_argument("--model-provider", type=str, default="Huggingface")
    parser.add_argument("--model-name", type=str, default="")

    parser.add_argument("--enable_h2o_generation", action='store_true')
    parser.add_argument("--num_heavy_hitter_tokens", type=int, default=128)
    parser.add_argument("--num_window_length", type=int, default=256)

    parser.add_argument("--enable_position_rolling", action='store_true')

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path
    model_provider = args.model_provider
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    # load the testing prompts
    for filename in tqdm.tqdm(glob.glob(f'{input_path}/{args.model_provider}_*_prompts.json')):
        with open(filename, 'r') as f:
            input_data = json.load(f)
            prompt = input_data[0]['content']+'\n'+input_data[1]['content']

            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
            context_length = input.input_ids.shape[-1]
            output = model.generate(
                **input,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )
            pred = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
            pred = pred.strip()

        basename = os.path.basename(filename)
        newname = basename.replace('.json', '.txt').replace('_prompts', '')
        with open(f'{output_path}/{newname}', 'w') as f:
            f.write(pred)
















