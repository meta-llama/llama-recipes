# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import os
import sys
from typing import List

from transformers import LlamaTokenizer
from safety_utils import get_safety_checker
from model_utils import load_model, load_peft_model, load_llama_from_config
from accelerate import init_empty_weights
# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)
from model_checkpointing import load_sharded_model_single_gpu

def main(
    model_name,
    save_dir="", # Path to save the HF converted model checkpoints
    model_path="" # Path/ name of the HF model that include config.json and tokenizer_config.json
    ):
    #load the HF model definition from config
    model_def = load_llama_from_config(model_path)
    print("model is loaded from config")
    #load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model_def, model_name)
    print("model is loaded from FSDP checkpoints")
    #loading the tokenizer form the  model_path
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_dir)
    #save the FSDP sharded checkpoints in HF format
    model.save_pretrained(save_dir)
    print(f"HuggingFace model checkpoints has been saved in {save_dir}")
if __name__ == "__main__":
    fire.Fire(main)
