import copy
import datasets
import itertools
import yaml

from llama_recipes.configs import train_config

B_INST, E_INST = "[INST]", "[/INST]"

prompt_template_filename = "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/datasets/training_prompt_templates/hl_mr_prompt.yaml"

def tokenize_json(json_data, tokenizer):
    # prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
    # answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
    # dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(json_data['prompt']).strip()} {E_INST}", add_special_tokens=False)]
    json_tokens = [tokenizer.encode(f"{json_data['json'].strip()} {tokenizer.eos_token}", add_special_tokens=False)]
    
    
    # dialog_tokens = list(itertools.chain(prompt_tokens, json_tokens))
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, json_tokens)))
    
    # labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]
    
    #Add labels, convert prompt token to -100 in order to ignore in loss function
    labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_hl_custom_dataset(dataset_config, tokenizer, split="train"):
    dataset = datasets.load_dataset(train_config.dataset_path, split=split)
    
    with open(prompt_template_filename, 'r') as file:
        yaml_data = yaml.safe_load(file)
    prompt = yaml_data['prompt']
    
    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(user_text=sample["query"]),
            "json": sample["json"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    
    dataset = dataset.map(lambda x: tokenize_json(x, tokenizer), remove_columns=list(dataset.features))

    return dataset