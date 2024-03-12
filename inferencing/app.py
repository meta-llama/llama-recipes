
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
from transformers import AutoTokenizer
from urllib.parse import unquote
import yaml

CUDA_VISIBLE_DEVICES = [0]
MAX_BATCH_SIZE = 30
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in CUDA_VISIBLE_DEVICES])

app = FastAPI()

import time
import datasets
from datasets import load_dataset
import time
import ctranslate2
import sentencepiece as spm
import torch
import random

model_dir="models/merged-codellama-ct2"
generator = ctranslate2.Generator(model_dir, device="cuda")
sp = spm.SentencePieceProcessor(os.path.join(model_dir, "tokenizer.model"))

max_tokens=128
intent_names =[]
continuities=[]
direct_commands=[]
model_raw_outputs = []

@app.get("/healthcheck")
async def health_check():
    return {"message": "Ok", "status": "Green"}

@app.get("/invoke")
async def invoke(query: str):

    query = unquote(query)
    with open(os.path.dirname(__file__) + "/hl_mr_prompt.yaml", "r") as file:
            yaml_data = yaml.safe_load(file)
    prompt_template = yaml_data["prompt"].strip()
    # self.sys_prompt = yaml_data["prompt"].strip()
    B_INST, E_INST = "[INST]", "[/INST]"
    bos_token = "<s>"
    prompt = f"{bos_token}{B_INST} {prompt_template.format(user_text=query).strip()} \
            {E_INST}"

    prompts=[prompt]

    prompt_tokens = sp.encode(prompts, out_type=str)
    gen_results = generator.generate_batch(
        prompt_tokens,
        max_batch_size=MAX_BATCH_SIZE,
        sampling_temperature=0.01,
        sampling_topk=1,
        sampling_topp=0.5,
        max_length=max_tokens,
        include_prompt_in_result=False
    )
    for gen_result in gen_results:
        output_ids = []
        step_results = gen_result.sequences_ids[0]
        for step_result in step_results:
            output_ids.append(step_result)
        
        result = sp.decode(output_ids)
        model_raw_outputs.append(result)

    return result