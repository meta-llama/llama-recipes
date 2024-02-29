# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import os
import sys
import time

import torch
from transformers import AutoTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model


def handle_safety_check(are_safe_user_prompt, user_prompt, safety_results_user_prompt, are_safe_system_prompt, system_prompt, safety_results_system_prompt):
    """
    Handles the output based on the safety check of both user and system prompts.

    Parameters:
    - are_safe_user_prompt (bool): Indicates whether the user prompt is safe.
    - user_prompt (str): The user prompt that was checked for safety.
    - safety_results_user_prompt (list of tuples): A list of tuples for the user prompt containing the method, safety status, and safety report.
    - are_safe_system_prompt (bool): Indicates whether the system prompt is safe.
    - system_prompt (str): The system prompt that was checked for safety.
    - safety_results_system_prompt (list of tuples): A list of tuples for the system prompt containing the method, safety status, and safety report.
    """
    def print_safety_results(are_safe_prompt, prompt, safety_results, prompt_type="User"):
        """
        Prints the safety results for a prompt.

        Parameters:
        - are_safe_prompt (bool): Indicates whether the prompt is safe.
        - prompt (str): The prompt that was checked for safety.
        - safety_results (list of tuples): A list of tuples containing the method, safety status, and safety report.
        - prompt_type (str): The type of prompt (User/System).
        """
        if are_safe_prompt:
            print(f"{prompt_type} prompt deemed safe.")
            print(f"{prompt_type} prompt:\n{prompt}")
        else:
            print(f"{prompt_type} prompt deemed unsafe.")
            for method, is_safe, report in safety_results:
                if not is_safe:
                    print(method)
                    print(report)
            print(f"Skipping the inference as the {prompt_type.lower()} prompt is not safe.")
            sys.exit(1)

    # Check user prompt
    print_safety_results(are_safe_user_prompt, user_prompt, safety_results_user_prompt, "User")
    
    # Check system prompt
    print_safety_results(are_safe_system_prompt, system_prompt, safety_results_system_prompt, "System")

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=False,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.6, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    enable_llamaguard_content_safety: bool=False, # Enable safety check with Llama-Guard
    use_fast_kernels: bool = True, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    system_prompt = input("Please insert your system prompt: ")
    user_prompt = input("Please insert your prompt: ")
    chat = [
   {"role": "system", "content": system_prompt},
   {"role": "user", "content": user_prompt},
    ]       
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    safety_checker = get_safety_checker(enable_azure_content_safety,
                                        enable_sensitive_topics,
                                        enable_salesforce_content_safety,
                                        enable_llamaguard_content_safety,
                                        )

    # Safety check of the user prompt
    safety_results_user_prompt = [check(user_prompt) for check in safety_checker]
    safety_results_system_prompt = [check(system_prompt) for check in safety_checker]
    are_safe_user_prompt = all([r[1] for r in safety_results_user_prompt])
    are_safe_system_prompt = all([r[1] for r in safety_results_system_prompt])
    handle_safety_check(are_safe_user_prompt, user_prompt, safety_results_user_prompt, are_safe_system_prompt, system_prompt, safety_results_system_prompt)
        
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs 
        )
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Safety check of the model output
    safety_results = [check(output_text) for check in safety_checker]
    are_safe = all([r[1] for r in safety_results])
    if are_safe:
        print("User input and model output deemed safe.")
        print(f"Model output:\n{output_text}")
    else:
        print("Model output deemed unsafe.")
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
                

if __name__ == "__main__":
    fire.Fire(main)
