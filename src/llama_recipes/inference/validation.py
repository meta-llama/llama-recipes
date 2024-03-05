from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import yaml

import pandas as pd
import yaml
import random
import time
import torch
from peft import PeftModel
from datasets import load_dataset
import re

device_arg = { 'device_map': 'auto' }
base_model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-13b-Instruct-hf",
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg
    )

model = PeftModel.from_pretrained(base_model, "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/codellama-8bit-json-mkt-research-24-03-04-merged_epoch_8", **device_arg)

tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/codellama-8bit-json-mkt-research-24-03-04-merged_tokenizer")

# tokenizer.pad_token = tokenizer.eos_token
def load_prompt_template(prompt_template_filename, user_text):
    with open(prompt_template_filename, 'r') as file:
        yaml_data = yaml.safe_load(file)
    prompt = yaml_data['prompt']
    return prompt.format(user_text=user_text)

dataset = load_dataset("HelixAI/hl-text-standard-json-single-turn-2024-03-04")
df_train = pd.DataFrame(dataset['train'])
df_sub = df_train.sample(n=20)

start = time.time()
prompt_template_filename = "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/datasets/training_prompt_templates/hl_mr_prompt.yaml"
count = 0
match_func = 0
match_param = 0
for index, row in df_sub.iterrows():
    count += 1
    query = row['query']
    ground_truth = row['json']
    base_prompt_template = load_prompt_template(prompt_template_filename, query)
    B_INST, E_INST = "[INST]", "[/INST]"
    base_prompt_template = f"{tokenizer.bos_token}{B_INST} {base_prompt_template.strip()} {E_INST}"

    start = time.time()
    model_input = tokenizer.batch_encode_plus([base_prompt_template], return_tensors="pt", add_special_tokens=False) # padding="max_length", max_length=1500, truncation=True,  


    # model_input
    # token_num = model_input["input_ids"].size(-1)
    model_input["input_ids"] = model_input["input_ids"].to(model.device)
    sequence = model.generate(**model_input, max_new_tokens=256) # temperature=0.01, 

    predict = map(lambda x: x, tokenizer.batch_decode(sequence[:], skip_special_tokens=True))
    # print("total time ---", start-time.time())
    predictions = list(predict)[0]
    g_func = re.findall(r'{"function": "(.*?)"',ground_truth)
    p_func = re.findall(r'{"function": "(.*?)"',predictions.split('/INST]')[1])
    g_param = re.findall(r'"params": {(.*?)}}',ground_truth)
    p_param = re.findall(r'"params": {(.*?)}}',predictions.split('/INST]')[1])
    if g_func == p_func:
        match_func +=1
    if g_param == p_param:
        match_param +=1
    # print("-"*100)
    # print("AI Response: ")
    # print(query,'@@@',predictions.split('/INST]')[1])
    # print("-"*100)
    # break
print("Function accuracy = ", match_func/count)
print("Param accuracy = ", match_param/count)
