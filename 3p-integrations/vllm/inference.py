# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import uuid
import asyncio
import fire

import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from accelerate.utils import is_xpu_available

if is_xpu_available():
    torch.xpu.manual_seed(42)
else:
    torch.cuda.manual_seed(42)

torch.manual_seed(42)

def load_model(model_name, peft_model=None, pp_size=1, tp_size=1):
    additional_configs = {}
    if peft_model:
        additional_configs["enable_lora"] = True
        
    engine_config = AsyncEngineArgs(
        model=model_name,
        pipeline_parallel_size=pp_size,
        tensor_parallel_size=tp_size,
        max_loras=1,
        **additional_configs)

    llm = AsyncLLMEngine.from_engine_args(engine_config)
    return llm

async def main(
    model,
    peft_model_name=None,
    max_new_tokens=100,
    user_prompt=None,
    top_p=0.9,
    temperature=0.8
):
    while True:
        if user_prompt is None:
            user_prompt = input("Enter your prompt: ")
            
        print(f"User prompt:\n{user_prompt}")

        print(f"sampling params: top_p {top_p} and temperature {temperature} for this inference request")
        sampling_param = SamplingParams(top_p=top_p, temperature=temperature, max_tokens=max_new_tokens)

        lora_request = None
        if peft_model_name:
            lora_request = LoRARequest("lora",0,peft_model_name)

        req_id = str(uuid.uuid4())

        generator = model.generate(user_prompt, sampling_param, req_id, lora_request=lora_request)
        output = None
        async for request_output in generator:
            output = request_output
   
        print(f"model output:\n {user_prompt} {output.outputs[0].text}")
        user_prompt = input("Enter next prompt (press Enter to exit): ")
        if not user_prompt:
            break

def run_script(
    model_name: str,
    peft_model_name=None,
    pp_size : int = 1,
    tp_size : int = 1,
    max_new_tokens=100,
    user_prompt=None,
    top_p=0.9,
    temperature=0.8
):
    model = load_model(model_name, peft_model_name, pp_size, tp_size)

    asyncio.get_event_loop().run_until_complete(main(model, peft_model_name, max_new_tokens, user_prompt, top_p, temperature))

if __name__ == "__main__":
    fire.Fire(run_script)
