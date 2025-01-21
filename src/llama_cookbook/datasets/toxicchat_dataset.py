# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3.1 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/lmsys/toxic-chat

import copy
import datasets
import itertools
from llama_recipes.inference.prompt_format_utils import  LLAMA_GUARD_3_CATEGORY
import ast
import fire

def tokenize_prompt_and_labels(full_prompt, tokenizer):
        prompt_tokens = tokenizer.encode(full_prompt)
        combined_tokens = {
            "input_ids": list(prompt_tokens),
            "labels": list(prompt_tokens)
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
    

from llama_recipes.data.llama_guard.finetuning_data_formatter import TrainingExample, Guidelines, Category, LlamaGuardPromptConfigs, LlamaGuardGenerationConfigs, ExplanationPosition, AugmentationConfigs, FormatterConfigs, create_formatted_finetuning_examples
from datasets import Dataset, DatasetInfo

def mapTcCategoriesToLGCategories(TcCategoriesString):
    TcCategories = ast.literal_eval(TcCategoriesString)
    if(len(TcCategories)==0):
         return None
    ranked = sorted(TcCategories, key=lambda x: x[1], reverse=True)
    primary = ranked[0][0] if len(ranked) else None
    TcMapping = {
        "sexual":"012", 
        "violence":"01", 
        "sexual/minors":"04", 
        "self-harm/intent":"011", 
        "hate":"010", 
        "harassment":"010", 
        "self-harm":"011", 
        "self-harm/instructions":"011", 
        "violence/graphic":"01", 
        "harassment/threatening":"010", 
        "hate/threatening":"010"
    }
    return TcMapping[primary]

def get_llamaguard_toxicchat_dataset(dataset_config, tokenizer, split, return_jsonl=False):
    dataset = datasets.load_dataset("lmsys/toxic-chat", 'toxicchat0124', split=split)


    if return_jsonl:
        jsonl_data = [];
        for x in dataset.to_iterable_dataset():
            jsonl_data.append({
                    "prompt":x['user_input'],
                    "generation":x['model_output'],
                    "label":("good" if x['toxicity'] == 0 else "bad"),
                    "unsafe_content": [mapTcCategoriesToLGCategories(x['openai_moderation'])]
                    })
        return jsonl_data
    
    guidelines = Guidelines(
        categories=LLAMA_GUARD_3_CATEGORY,
        category_code_prefix="O")
    
    # Setting up configs
    llama_guard_prompt_configs = LlamaGuardPromptConfigs(
        instructions_format_string="""<|begin_of_text|><|start_header_id|>user<|end_header_id|> Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

    <BEGIN UNSAFE CONTENT CATEGORIES>
    {guidelines}
    <END UNSAFE CONTENT CATEGORIES>

    <BEGIN CONVERSATION>

    {conversation}

    <END CONVERSATION>

    Provide your safety assessment for $agent_type in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories. <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        should_include_category_descriptions=False,
        should_shuffle_category_codes=False
    )

    llama_guard_generation_configs = LlamaGuardGenerationConfigs(
        should_list_violated_codes=True,
        explanation_position=None
    )

    augmentation_configs = AugmentationConfigs(
        should_add_examples_with_dropped_nonviolated_prompt_categories=False,
        should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=False,
        explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect."
    )

    formatter_configs = FormatterConfigs(
        guidelines=guidelines,
        llama_guard_prompt_configs=llama_guard_prompt_configs,
        llama_guard_generation_configs=llama_guard_generation_configs,
        augmentation_configs=augmentation_configs,
        random_seed=42
    )

    dataset = dataset.map(lambda x: {"full_prompt": create_formatted_finetuning_examples(
        [TrainingExample(
            prompt=x["user_input"],
            response=None,
            violated_category_codes = [] if x["toxicity"]==0 else [mapTcCategoriesToLGCategories(x["openai_moderation"])],
            label="safe" if x["toxicity"]==0 else "unsafe",
            explanation="The response contains violating information."
        )],
        formatter_configs)[0]}, 
        remove_columns=list(dataset.features))

    dataset = dataset.map(lambda x: tokenize_prompt_and_labels(x["full_prompt"], tokenizer), remove_columns=list(dataset.features))
    return dataset

def main(return_jsonl = False):
    from transformers import AutoTokenizer
    model_id: str = "/home/ubuntu/LG3-interim-hf-weights"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if return_jsonl:
        dataset = get_llamaguard_toxicchat_dataset(None, tokenizer, "train", return_jsonl = True)
        print(dataset[0:50])
    else:
        dataset = get_llamaguard_toxicchat_dataset(None, tokenizer, "train")
        print(dataset[0])

if __name__ == '__main__':
    fire.Fire(main)
