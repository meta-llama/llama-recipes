# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import torch
from torch.utils.data import Dataset

RESPONSE_PROMPT = "\n\n### Response:{output}"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )

        response = RESPONSE_PROMPT.format_map(ann)
        response = self.tokenizer.encode(response)
        response.append(self.tokenizer.eos_token_id)

        response = torch.tensor(
            response, dtype=torch.int64
        )

        padding = self.max_words - (prompt.shape[0] + response.shape[0])
        print (f'max words: {self.max_words} padding: {padding}' )
        if padding > 0:
            example = torch.cat((prompt, response))
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        # we truncate the prompt and always keep the response
        elif padding <= 0:
            print (f'Truncating: {self.max_words} padding: {padding}')
            prompt = prompt[: self.max_words - response.shape[0]]
            example = torch.cat((prompt, response))

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
