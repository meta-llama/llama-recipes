# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# Running the script without any arguments "python modelUpgradeExample.py" performs inference with the Llama 3 8B Instruct model. 
# Passing  --model-id "meta-llama/Meta-Llama-3.1-8B-Instruct" to the script will switch it to using the Llama 3.1 version of the same model. 
# The script also shows the input tokens to confirm that the models are responding to the same input

import fire
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main(model_id = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a helpful chatbot"},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "content": "Because the light is scattered"},
        {"role": "user", "content": "Please tell me more about that"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    print("Input tokens:")
    print(input_ids)
    
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids,
        max_new_tokens=400,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        attention_mask=attention_mask,
    )
    response = outputs[0][input_ids.shape[-1]:]
    print("\nOutput:\n")
    print(tokenizer.decode(response, skip_special_tokens=True))

if __name__ == "__main__":
  fire.Fire(main)