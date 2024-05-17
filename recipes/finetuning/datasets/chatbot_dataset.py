# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.


import copy
import datasets
from datasets import Dataset, load_dataset, DatasetDict
import itertools


B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_dialog(q_a_pair, tokenizer):
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(question).strip()} {E_INST}", add_special_tokens=False) for question in q_a_pair["Question"]]
    answer_tokens = [tokenizer.encode(f"{answer.strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in q_a_pair["Answer"]]
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    #Add labels, convert prompt token to -100 in order to ignore in loss function
    labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split, split_ratio=0.8):
    dataset = load_dataset('json', data_files=dataset_config.data_path)
    dataset = dataset['train'].train_test_split(test_size=1-split_ratio, shuffle=True)

    dataset = dataset[split].map(lambda sample: {
        "Question": sample["Question"],
        "Answer": sample["Answer"],
        },
        batched=True,
    )
    dataset = dataset.map(lambda x: tokenize_dialog(x, tokenizer))
    return dataset
