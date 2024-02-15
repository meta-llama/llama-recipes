# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from transformers import AutoTokenizer, AutoModelForCausalLM


from llama_recipes.inference.prompt_format_utils import build_prompt, create_conversation, LLAMA_GUARD_CATEGORY
from typing import List, Tuple
from enum import Enum

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

def main():
    """
    Entry point of the program for generating text using a pretrained model.
    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """

    prompts: List[Tuple[List[str], AgentType]] = [
        (["<Sample user prompt>"], AgentType.USER),

        (["<Sample user prompt>",
        "<Sample agent response>"], AgentType.AGENT),
        
        (["<Sample user prompt>",
        "<Sample agent response>",
        "<Sample user reply>",
        "<Sample agent response>",], AgentType.AGENT),

    ]

    model_id = "meta-llama/LlamaGuard-7b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

    
    for prompt in prompts:
        formatted_prompt = build_prompt(
                prompt[1], 
                LLAMA_GUARD_CATEGORY, 
                create_conversation(prompt[0]))


        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0)
        results = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
       
        
        print(prompt[0])
        print(f"> {results}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)