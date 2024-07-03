# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import itertools
from llama_recipes.inference.prompt_format_utils import  LLAMA_GUARD_2_CATEGORY

B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_prompt_and_labels(full_prompt, tokenizer):
        prompt_tokens = tokenizer.encode(full_prompt)
        combined_tokens = {
            "input_ids": list(prompt_tokens),
            "labels": list(prompt_tokens)
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
    

from llama_recipes.data.llama_guard.finetuning_data_formatter import TrainingExample, Guidelines, Category, LlamaGuardPromptConfigs, LlamaGuardGenerationConfigs, ExplanationPosition, AugmentationConfigs, FormatterConfigs, create_formatted_finetuning_examples
from datasets import Dataset, DatasetInfo

#TODO Convert to training using toxic chat categories so that this is not necessary
def mapTcCategoriesToLGCategories(TcCategories):
      return ["01"]

def get_llamaguard_toxicchat_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("lmsys/toxic-chat", 'toxicchat0124', split=split)

    guidelines = Guidelines(
        categories=LLAMA_GUARD_2_CATEGORY,
        category_code_prefix="O")
    
    # Setting up configs
    llama_guard_prompt_configs = LlamaGuardPromptConfigs(
        instructions_format_string="""[INST] Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

    <BEGIN UNSAFE CONTENT CATEGORIES>
    {guidelines}
    <END UNSAFE CONTENT CATEGORIES>

    <BEGIN CONVERSATION>

    {conversation}

    <END CONVERSATION>

    Provide your safety assessment for $agent_type in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""",
        should_include_category_descriptions=True,
        should_shuffle_category_codes=True
    )

    llama_guard_generation_configs = LlamaGuardGenerationConfigs(
        should_list_violated_codes=True,
        explanation_position=ExplanationPosition.AFTER_DECISION
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
            violated_category_codes = mapTcCategoriesToLGCategories(x["openai_moderation"]),
            label="safe" if x["toxicity"]==0 else "unsafe",
            explanation="The response contains violating information."
        )],
        formatter_configs)[0]}, 
        remove_columns=list(dataset.features))
    
    print(dataset)
    print(dataset[0])

    dataset = dataset.map(lambda x: tokenize_prompt_and_labels(x["full_prompt"], tokenizer), remove_columns=list(dataset.features))

    print(dataset)
    print(dataset[0])

    return dataset
