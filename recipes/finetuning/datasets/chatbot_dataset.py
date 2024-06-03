# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
import datasets
from datasets import Dataset, load_dataset, DatasetDict
import itertools


B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_dialog(q_a_pair, tokenizer):
    question, answer = q_a_pair["Question"], q_a_pair["Answer"]
    prompt_tokens = tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(question).strip()} {E_INST}", add_special_tokens=False)
    answer_tokens = tokenizer.encode(f"{answer.strip()} {tokenizer.eos_token}", add_special_tokens=False)
    sample = {
            "input_ids": prompt_tokens + answer_tokens,
            "attention_mask" : [1] * (len(prompt_tokens) + len(answer_tokens)),
            "labels": [-100] * len(prompt_tokens) + answer_tokens,
            }

    return sample


def get_custom_dataset(dataset_config, tokenizer, split, split_ratio=0.8):
    dataset_dict = load_dataset('json', data_files=dataset_config.data_path)
    dataset = dataset_dict['train']
    dataset = dataset.train_test_split(test_size=1-split_ratio, shuffle=True, seed=42)

    dataset = dataset[split].map(lambda sample: {
        "Question": sample["Question"],
        "Answer": sample["Answer"],
        },
        batched=True,
    )
    dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer))
    return dataset
