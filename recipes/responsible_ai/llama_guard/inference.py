# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_recipes.inference.prompt_format_utils import build_prompt, create_conversation, LLAMA_GUARD_CATEGORY
from examples.llama_guard.generation import Llama
from examples.llama_guard.perf_utils import time_decorator

from typing import List, Optional, Tuple, Dict
from enum import Enum
import json
import numpy as np

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

def main():
   
    prompts: List[Dict[List[str], AgentType]] = [
        {
            "prompt": ["<Sample user prompt>"],
            "agent_type": AgentType.USER
        },
        {
            "prompt": ["<Sample user prompt>", "<Sample agent response>"],
            "agent_type": AgentType.AGENT
        },
        {
            "prompt": ["<Sample user prompt>", 
                       "<Sample agent response>", 
                       "<Sample user reply>", 
                       "<Sample agent response>"],
            "agent_type": AgentType.AGENT
        }
    ]

    
    results = llm_eval(prompts)
    
    for i, prompt in enumerate(prompts):
        print(prompt[0])
        print(f"> {results[i]}")
        print("\n==================================\n")

@time_decorator
def llm_eval(prompts, load_in_8bit=True, logprobs = False) -> Tuple[List[str], Optional[List[List[Tuple[int, float]]]]]:

    model_id = "meta-llama/LlamaGuard-7b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=load_in_8bit, device_map="auto")

    results: List[str] = []
    if logprobs:
        result_logprobs: List[List[Tuple[int, float]]] = []

    for prompt in prompts:
        formatted_prompt = build_prompt(
                prompt["agent_type"], 
                LLAMA_GUARD_CATEGORY, 
                create_conversation(prompt["prompt"]))


        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=10, pad_token_id=0, return_dict_in_generate=True, output_scores=logprobs)
        
        if logprobs:
            transition_scores = model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True)

        generated_tokens = output.sequences[:, prompt_len:]
        
        if logprobs:
            temp_logprobs: List[Tuple[int, float]] = []
            for tok, score in zip(generated_tokens[0], transition_scores[0]):
                temp_logprobs.append((tok.cpu().numpy(), score.cpu().numpy()))
            
            result_logprobs.append(temp_logprobs)
            prompt["logprobs"] = temp_logprobs
        
        # result = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        result = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)    

        prompt["result"] = result
        results.append(result)

    return (results, result_logprobs if logprobs else None)  

@time_decorator
def standard_llm_eval(prompts: List[Tuple[List[str], AgentType, str, str, str]], ckpt_dir, logprobs: bool = False):
    # defaults
    temperature = 1
    top_p = 1
    max_seq_len = 4096
    max_gen_len = 32
    max_batch_size = 1

    generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=ckpt_dir + "/tokenizer.model",
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )


    results: List[str] = []
    for prompt in prompts:
        formatted_prompt = build_prompt(
                prompt["agent_type"], 
                LLAMA_GUARD_CATEGORY, 
                create_conversation(prompt["prompt"]))

        # result = generator.single_prompt_completion(
        #     formatted_prompt,
        #     max_gen_len=max_gen_len,
        #     temperature=temperature,
        #     top_p=top_p,
        # )
        result = generator.text_completion(
            [formatted_prompt],
            temperature,
            top_p,
            max_gen_len,
            logprobs
        )
        # getting the first value only, as only a single prompt was sent to the function
        generation_result = result[0]["generation"]
        prompt["result"] = generation_result
        if logprobs:
            prompt["logprobs"] = result[0]["logprobs"]

        results.append(generation_result)

    return results

if __name__ == "__main__":
    fire.Fire(main)