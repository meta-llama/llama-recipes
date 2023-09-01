# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import os
import sys
import time

from transformers import AutoTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model

def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.6, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    use_fast_kernels: bool = True, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"
        with open(prompt_file, "r") as f:
            user_prompt = f.read()
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    model.config.tp_size=1
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()
    
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    safety_checker = get_safety_checker(enable_azure_content_safety,
                                        enable_sensitive_topics,
                                        enable_salesforce_content_safety,
                                        )

    # Safety check of the user prompt
    safety_results = [check(user_prompt) for check in safety_checker]
    are_safe = all([r[1] for r in safety_results])
    if are_safe:
        print("User prompt deemed safe.")
        print(f"User prompt:\n{user_prompt}")
    else:
        print("User prompt deemed unsafe.")
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
        print("Skipping the inference as the prompt is not safe.")
        sys.exit(1)  # Exit the program with an error status
        
    batch = tokenizer(user_prompt, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
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
    filling = tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    # Safety check of the model output
    safety_results = [check(filling) for check in safety_checker]
    are_safe = all([r[1] for r in safety_results])
    if are_safe:
        print("User input and model output deemed safe.")
        print(user_prompt.replace("<FILL_ME>", filling))
    else:
        print("Model output deemed unsafe.")
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
                

if __name__ == "__main__":
    fire.Fire(main)
