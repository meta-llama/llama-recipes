# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import os
import sys
import yaml
import fire

from transformers import AutoTokenizer

from llama_recipes.inference.model_utils import load_llama_from_config

from model_checkpointing import load_sharded_model_single_gpu

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)

def get_model_name_from_yaml(fsdp_checkpoint_path):
    """
    Retrieve the model name from the train_params.yaml file.
    
    Args:
        fsdp_checkpoint_path (str): Path to the FSDP sharded model checkpoints.
    
    Returns:
        str: Model name from the YAML file.
    """
    file_name = 'train_params.yaml'
    train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
    
    try:
        with open(train_params_path, 'r') as file:
            data = yaml.safe_load(file)
            model_name = data.get('model_name')
            print(f"Model name: {model_name}")
            return model_name
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        model_name = input("Please enter the model name: ")
        print(f"Model name: {model_name}")
        return model_name
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_and_save_model(fsdp_checkpoint_path, consolidated_model_path, hf_model_path_or_name):
    """
    Load the model from FSDP sharded checkpoints and save it in HuggingFace format.
    
    Args:
        fsdp_checkpoint_path (str): Path to FSDP sharded model checkpoints.
        consolidated_model_path (str): Path to save the HF converted model checkpoints.
        hf_model_path_or_name (str): Path or name of the HF model that includes config.json and tokenizer_config.json.
    """
    model_def = load_llama_from_config(hf_model_path_or_name)
    print("Model loaded from config")
    
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("Model loaded from FSDP checkpoints")
    
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path_or_name)
    tokenizer.save_pretrained(consolidated_model_path)
    
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints have been saved in {consolidated_model_path}")

def main(fsdp_checkpoint_path="", consolidated_model_path="", hf_model_path_or_name=""):
    """
    Main function to load and save the model.
    
    Args:
        fsdp_checkpoint_path (str): Path to FSDP sharded model checkpoints.
        consolidated_model_path (str): Path to save the HF converted model checkpoints.
        hf_model_path_or_name (str): Path or name of the HF model that includes config.json and tokenizer_config.json.
    """
    if not hf_model_path_or_name:
        hf_model_path_or_name = get_model_name_from_yaml(fsdp_checkpoint_path)
    
    if hf_model_path_or_name:
        load_and_save_model(fsdp_checkpoint_path, consolidated_model_path, hf_model_path_or_name)
    else:
        print("Model name is required to proceed.")

if __name__ == "__main__":
    fire.Fire(main)
