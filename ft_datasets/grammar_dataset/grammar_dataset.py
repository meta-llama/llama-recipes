# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/jfleg
# For download and preparation see: recipes/ft_datasets/grammar_dataset/grammar_dataset_process.ipynb

import argparse
import csv
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from pathlib import Path

from ft_datasets.utils import ConcatDataset



class grammar(Dataset):
    def __init__(
        self,
        tokenizer,
        csv_name=None,
    ):

        try:
            self.dataset = load_dataset(
                "csv",
                data_files={"train": [csv_name]},  # "eval": "grammar_validation.csv"},
                delimiter=",",
            )
        except Exception as e:
            print("Loading of grammar dataset failed! Please see recipes/ft_datasets/grammar_dataset/grammar_dataset_process.ipynb for details on how to download the dataset.")
            raise e

        # self.dataset = load_dataset("wikihow", "all", data_dir="data/", split=type_path)
        # if num_samples:
        #    self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.tokenizer = tokenizer
        self.print_text = False  # print_text

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):

        # Create prompt and tokenize contexts and questions

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch["text"]))

        input_ = example_batch["input"]
        target_ = example_batch["target"]
        
        prompt = f"Correct this to standard English: {input_}\n---\nCorrected: {target_}"
        sample = self.tokenizer(prompt)
        
        return sample

    def __getitem__(self, index):
        sample = self.convert_to_features(self.dataset["train"][index])
        source_ids = sample["input_ids"]

        src_mask = sample["attention_mask"]

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": source_ids.copy(),
        }


def get_dataset(
    dataset_config, tokenizer, csv_name=None
):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    if csv_name is None:
        currPath = Path.cwd() / "datasets_grammar" / "grammar_train.csv"
        print(f"Loading dataset {currPath}")
        csv_name = str(currPath)
    dataset = grammar(
        tokenizer=tokenizer,
        csv_name=csv_name,
    )
    
    return ConcatDataset(dataset, chunk_size=dataset_config.input_length)

