# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(push_to_hub: bool = True):
    base_model = "codellama/CodeLlama-13b-Instruct-hf"
    peft_model = "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/codellama-8bit-json-mkt-research-24-03-07_epoch_8"
    tokenizer_path = "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/codellama-8bit-json-mkt-research-24-03-07_tokenizer"
    output_dir = "HelixAI/codellama-8bit-json-mkt-research-24-03-07"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp", 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path
    )
        
    model = PeftModel.from_pretrained(
        model, 
        peft_model, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    model = model.merge_and_unload()
    
    if push_to_hub:
        print(f"Saving to hub ...")
        model.push_to_hub(f"{output_dir}", use_temp_dir=True, create_pr=1)
        tokenizer.push_to_hub(f"{output_dir}", use_temp_dir=True, create_pr=1)
    else:
        model.save_pretrained(f"{output_dir}")
        tokenizer.save_pretrained(f"{output_dir}")
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)