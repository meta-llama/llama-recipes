# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import datasets
from transformers import DataCollatorWithPadding
from .utils import Concatenator

def get_preprocessed_completion(dataset_config, tokenizer, split):
    if split == 'val':
        return {}
    
    def prepare(sample):
        sample = tokenizer(sample, padding=('max_length' if dataset_config.disable_packing else False))
        source_ids = sample["input_ids"]
        src_mask = sample["attention_mask"]

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": source_ids.copy(),
        }
    
    dataset = datasets.load_dataset(dataset_config.data_path, split=split)
    dataset = dataset.filter(lambda sample: sample["text"].strip() != '')
    dataset = dataset.map(
        lambda sample: prepare(sample["text"]),
        batched=True
    )
    if not dataset_config.disable_packing:
        dataset = dataset.map(Concatenator(chunk_size=4096), batched=True)
    return dataset
