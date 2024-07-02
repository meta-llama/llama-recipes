# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch.utils.data import Dataset
import copy
import json
import tqdm
import torch
from torch.utils.data import Dataset
import deeplake

PROMPT = """Given the instruction containing context and the question, provide the logical reasoning that led you to the answer.
        Please use the format of: ##Reason: reason ##Answer: answer.
        ###Instruction: {instruction}\n\n### Response:"""


class InstructionRAFTDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.tokenizer = tokenizer

        val_quantity = dataset_config.val_quantity
        ds = deeplake.load(dataset_config.dataset_path)

        if partition == "train":
            ds = ds[val_quantity:]
            self.ann = ds
        else:
            ds = ds[:val_quantity]
            self.ann = ds

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        if index < len(self):

            IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

            column_map = self.ann.tensors.keys()
            ann = {}
            for el in column_map:  # {"column_name" : "value"}
                ann[el] = self.ann[el][index].text().astype(str)

            prompt = PROMPT.format_map(ann)

            example = prompt + ann["cot_answer"]
            prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
            example = self.tokenizer.encode(example)
            example.append(self.tokenizer.eos_token_id)
            example = torch.tensor(
                example, dtype=torch.int64
            )
            labels = copy.deepcopy(example)
            labels[: len(prompt)] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = IGNORE_INDEX

            return {
                "input_ids": example.tolist(),
                "labels": labels.tolist(),
                "attention_mask": example_mask.tolist(),
            }
        else:
            raise IndexError
