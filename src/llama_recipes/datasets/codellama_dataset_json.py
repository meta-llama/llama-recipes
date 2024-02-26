import pandas as pd
import datasets
import yaml
import random

from llama_recipes.configs import train_config

def get_code_llama_dataset_json(dataset_config, tokenizer, split):
    # dataset_train = datasets.load_dataset("HelixAI/function_full_dataset_30_11", split=split)#, revision="bb1cbb8a9369449456203b15dac4d6fe61d42f5c")
    # dataset_test = datasets.load_dataset("HelixAI/function_full_dataset_30_11", split='test')
    # dataset = datasets.concatenate_datasets([dataset_train,dataset_test])
    # dataset = dataset.filter(lambda x: x['json'] != None)
    dataset = datasets.load_dataset(train_config.dataset_path, split="train")
    chat_history_list=[]
    for sample in dataset:
        chat_history_turn = "Human: "+ sample["query"] + "\nAI Response JSON: " + sample["json"]
        chat_history_list.append(chat_history_turn)
    def load_prompt_template(prompt_template_filename):
        with open(prompt_template_filename, 'r') as file:
            yaml_data = yaml.safe_load(file)
        prompt_template = yaml_data['prompt_template'].strip()
        prompt = yaml_data['prompt'].strip()
        return prompt_template, prompt
    def apply_prompt_template(sample):
        prompt_template, prompt = load_prompt_template('/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/datasets/training_prompt_templates/prompt_template_codellama_json.yaml')
        chat_history = random.choice(chat_history_list)
        num = random.choice([1,2,3])
        if num//3 == 1:
            chat_history = ''
        return {
            "prompt": prompt_template.format(
                prompt=prompt,
                # chat_history=chat_history,
                user_text=sample["query"],
                completion=sample["json"],
                eos_token=tokenizer.eos_token,
        ).lstrip(),
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    tokenizer.pad_token = tokenizer.eos_token
    dataset = dataset.map(
        lambda sample: tokenizer(sample["prompt"], padding="max_length", max_length=1500, truncation=True, return_tensors="pt", add_special_tokens=False),
        batched=True,
        remove_columns=list(dataset.features)
    )
    #print(f"before concatenation {len(dataset)}")
    #dataset = dataset.map(Concatenator(chunk_size=512), batched=True)
    dataset = dataset.add_column('labels', dataset['input_ids'].copy()) # I removed this line
    #print(f"after concatenation {len(dataset)}")
    return dataset

