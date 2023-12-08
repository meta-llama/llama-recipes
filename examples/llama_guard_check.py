# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama_guard.generation import Llama
from llama_guard.prompt_format import build_prompt, create_conversation, LLAMA_GUARD_CATEGORY
from typing import List, Tuple
from enum import Enum

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
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
        (["<Sample User prompt goes here>"], AgentType.USER),

        (["<Sample User prompt goes here>",
        "<Sample Agent prompt goes here>"], AgentType.AGENT),

    ]

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    for prompt in prompts:
        formatted_prompt = build_prompt(
                prompt[1], 
                LLAMA_GUARD_CATEGORY, 
                create_conversation(prompt[0]))
        

        results = generator.single_prompt_completion(
            formatted_prompt,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        print(prompt)
        print(f"> {results}")
        print("\n==================================\n")



if __name__ == "__main__":
    fire.Fire(main)