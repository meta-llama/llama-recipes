# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Running Llama Guard inference
#
# This notebook is intented to showcase how to run Llama Guard inference on a sample prompt for testing.

# +
# # !pip install --upgrade huggingface_hub

# +
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion
from llama_recipes.inference.llama.generation import Llama

from typing import List, Optional, Tuple, Dict
from enum import Enum

import torch
from tqdm import tqdm


# -

# # Defining the main functions
#
# Agent type enum to define what type of inference Llama Guard should be doing, either User or Agent.
#
# The llm_eval function loads the Llama Guard model from Hugging Face. Then iterates over the prompts and generates the results for each token.

# +
class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

def llm_eval(prompts: List[Tuple[List[str], AgentType]],
            model_id: str = "meta-llama/Meta-Llama-Guard-2-8B",
            llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2.name, 
            load_in_8bit: bool = True, 
            load_in_4bit: bool = False, 
            logprobs: bool = False) -> Tuple[List[str], Optional[List[List[Tuple[int, float]]]]]:
    """
    Runs Llama Guard inference with HF transformers. Works with Llama Guard 1 or 2

    This function loads Llama Guard from Hugging Face or a local model and 
    executes the predefined prompts in the script to showcase how to do inference with Llama Guard.

    Parameters
    ----------
        prompts : List[Tuple[List[str], AgentType]]
            List of Tuples containing all the conversations to evaluate. The tuple contains a list of messages that configure a conversation and a role.
        model_id : str 
            The ID of the pretrained model to use for generation. This can be either the path to a local folder containing the model files,
            or the repository ID of a model hosted on the Hugging Face Hub. Defaults to 'meta-llama/Meta-Llama-Guard-2-8B'.
        llama_guard_version : LlamaGuardVersion
            The version of the Llama Guard model to use for formatting prompts. Defaults to LLAMA_GUARD_2.
        load_in_8bit : bool
            defines if the model should be loaded in 8 bit. Uses BitsAndBytes. Default True 
        load_in_4bit : bool
            defines if the model should be loaded in 4 bit. Uses BitsAndBytes and nf4 method. Default False
        logprobs: bool
            defines if it should return logprobs for the output tokens as well. Default False

    """

    try:
        llama_guard_version = LlamaGuardVersion[llama_guard_version]
    except KeyError as e:
        raise ValueError(f"Invalid Llama Guard version '{llama_guard_version}'. Valid values are: {', '.join([lgv.name for lgv in LlamaGuardVersion])}") from e

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    torch_dtype = torch.bfloat16
    # if load_in_4bit:
    #     torch_dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype
    )

    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    results: List[str] = []
    if logprobs:
        result_logprobs: List[List[Tuple[int, float]]] = []

    total_length = len(prompts)
    progress_bar = tqdm(colour="blue", desc=f"Prompts", total=total_length, dynamic_ncols=True)
    for prompt in prompts:
        formatted_prompt = build_default_prompt(
                prompt["agent_type"], 
                create_conversation(prompt["prompt"]),
                llama_guard_version)


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
        
        result = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)    

        prompt["result"] = result
        results.append(result)
        progress_bar.update(1)

    progress_bar.close()
    return (results, result_logprobs if logprobs else None)  


# -

def pytorch_llm_eval(prompts: List[Tuple[List[str], AgentType, str, str, str]], 
                     ckpt_dir, 
                     logprobs: bool = False,
                    llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2, 
):
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
    total_length = len(prompts)
    progress_bar = tqdm(colour="blue", desc=f"Prompts", total=total_length, dynamic_ncols=True)
    for prompt in prompts:
        formatted_prompt = build_default_prompt(
                prompt["agent_type"], 
                create_conversation(prompt["prompt"]),
                llama_guard_version)

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
        progress_bar.update(1)

    progress_bar.close()
    return results

# Setting variables used by the Llama classes
import os
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"


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
    
    # results = llm_eval(prompts, load_in_8bit = False, load_in_4bit = True)
    results = pytorch_llm_eval(prompts, ckpt_dir="../../../../llama/models/guard-llama/", llama_guard_version=LlamaGuardVersion.LLAMA_GUARD_1.name)
    
    for i, prompt in enumerate(prompts):
        print(prompt['prompt'])
        print(f"> {results[0][i]}")
        print("\n==================================\n")


# used to be able to import this script in another notebook and not run the main function
if __name__ == '__main__' and '__file__' not in globals():
    # from huggingface_hub import login
    # login()
    main()

torch.cuda.empty_cache()


