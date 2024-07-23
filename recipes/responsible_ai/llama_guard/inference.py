# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion
from typing import List, Tuple
from enum import Enum

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

def main(
    model_id: str = "meta-llama/Llama-Guard-3-8B",
    llama_guard_version: str = "LLAMA_GUARD_3"
):
    """
    Entry point for Llama Guard inference sample script.

    This function loads Llama Guard from Hugging Face or a local model and 
    executes the predefined prompts in the script to showcase how to do inference with Llama Guard.

    Args:
        model_id (str): The ID of the pretrained model to use for generation. This can be either the path to a local folder containing the model files,
            or the repository ID of a model hosted on the Hugging Face Hub. Defaults to 'meta-llama/LlamaGuard-7b'.
        llama_guard_version (LlamaGuardVersion): The version of the Llama Guard model to use for formatting prompts. Defaults to LLAMA_GUARD_1.
    """
    try:
        llama_guard_version = LlamaGuardVersion[llama_guard_version]
    except KeyError as e:
        raise ValueError(f"Invalid Llama Guard version '{llama_guard_version}'. Valid values are: {', '.join([lgv.name for lgv in LlamaGuardVersion])}") from e

    prompts: List[Tuple[List[str], AgentType]] = [
        (["<Sample user prompt>"], AgentType.USER),

        (["<Sample user prompt>",
        "<Sample agent response>"], AgentType.AGENT),
        
        (["<Sample user prompt>",
        "<Sample agent response>",
        "<Sample user reply>",
        "<Sample agent response>",], AgentType.AGENT),

    ]

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    
    for prompt in prompts:
        formatted_prompt = build_default_prompt(
                prompt[1], 
                create_conversation(prompt[0]),
                llama_guard_version)


        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0)
        results = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
       
        
        print(prompt[0])
        print(f"> {results}")
        print("\n==================================\n")

if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        print(e)