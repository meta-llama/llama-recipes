import copy
import json
import os
import torch

from torch.utils.data import Dataset
from typing import List


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=4000):
        self.data = []
        with open(dataset_config.data_path) as f:
            self.data = [json.loads(line) for line in f]
        if partition == "train":
            self.data = self.data[500:15000]
        else:
            self.data = self.data[:500]
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        prompt = data["prompt"]
        example = prompt + data["response"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = torch.tensor(
            self.tokenizer.encode(example), dtype=torch.int64
        )
        response = torch.tensor(
            self.tokenizer.encode(data["response"]), dtype=torch.int64
        )

        # Calculate padding.
        padding = self.max_words - example.shape[0]
        # Padding
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            labels = copy.deepcopy(example)
            clipping = max(0, len(prompt)-1)
            # Mask the input labels.
            labels[:clipping] = -1
        # Clipping
        elif padding < 0:
            response_len = response.shape[0]
            example = example[-self.max_words:]
            labels = copy.deepcopy(example)
            # Mask the input labels.
            if self.max_words-response_len > 0:
                labels[: self.max_words-response_len] = -1 
        
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = -100
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        
        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
