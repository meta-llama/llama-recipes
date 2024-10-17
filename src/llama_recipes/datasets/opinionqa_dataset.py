# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum
import os
import ast
import json
import datasets
import pandas as pd


def get_preprocessed_opinionqa(dataset_config, tokenizer, split, save = True, debug = False):

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(
            tokenizer.bos_token + sample["input_prompt"],
            add_special_tokens=False
        )
        answer = tokenizer.encode(
            sample["output_prompt"].strip() + tokenizer.eos_token,
            add_special_tokens=False
        ) # detail: adding strip(), because " A" is tokenized as ['<s>', '', ' A']
          # i.e., the whitespace is automatically included in the token list..
        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "labels": [-100] * len(prompt) + answer,
            }
        return sample
    
    preprocessed_file_dir = split.split(".csv")[0] + "_preprocessed.json"

    if os.path.exists(preprocessed_file_dir): # if preprocessed file exists
        print("preprocessed file exists.")
        with open(split.split(".csv")[0] + "_preprocessed.json", 'r') as f:
            dataset_dict = json.load(f)
            dataset = datasets.Dataset.from_dict(dataset_dict)
    else:
        dataset = datasets.load_dataset(
            'csv', 
            data_files = split
        )['train']  # detail: not sure why, 
                    # but getting DatasetDict with 'train' key every time
        if debug:
            dataset = datasets.Dataset.from_dict(dataset[0:100]) # debug purpose, take 100 rows
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features), num_proc=32)

        if save:
            # save dataset to json format
            dataset_dict = dataset.to_dict()
            with open(split.split(".csv")[0] + "_preprocessed.json", 'w') as f:
                json.dump(dataset_dict, f)

    return dataset



def get_preprocessed_opinionqa_ce_or_wd_loss(dataset_config, tokenizer, split, save = True):

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(
            tokenizer.bos_token + sample["input_prompt"],
            add_special_tokens=False
        )
        padding_value = 10
        resp_dist = ast.literal_eval(sample["output_dist"])
        resp_dist = resp_dist + [0] * (padding_value - len(resp_dist)) # making resp_dist of same length
        ordinal_info = sample.get("ordinal", None)
        if ordinal_info is not None:
            ordinal_info = ast.literal_eval(ordinal_info)
            ordinal_info = ordinal_info + [max(ordinal_info)] * (padding_value - len(ordinal_info))

        answer = tokenizer.encode(
            "A" + tokenizer.eos_token, # "A" is just a placeholder
            add_special_tokens=False
        ) # detail: adding strip(), because " A" is tokenized as ['<s>', ' ', ' A'] with Llama2 tokenizer
          # i.e., the whitespace is automatically included in the token list
          # this problem is tokenizer-specific. @Joshua is aware of this issue.
        sample = {
            "input_ids": prompt + answer,
            "attention_mask" : [1] * (len(prompt) + len(answer)),
            "target_token_position": len(prompt),
            "response_distribution": resp_dist
            }
        if ordinal_info is not None:
            sample["ordinal_info"] = ordinal_info
        return sample

    preprocessed_file_dir = split.split(".csv")[0] + "_preprocessed.json"

    if os.path.exists(preprocessed_file_dir): # if preprocessed file exists
        with open(split.split(".csv")[0] + "_preprocessed.json", 'r') as f:
            print("preprocessed file exists.")
            dataset_dict = json.load(f)
            dataset = datasets.Dataset.from_dict(dataset_dict)
    else: # if preprocessed file does not exist, preprocess the dataset
        dataset = datasets.load_dataset(
            'csv', 
            data_files = split
        )['train']  # detail: not sure why, 
                    # but getting DatasetDict with 'train' key every time
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features), num_proc=32)

        if save:
            # save dataset to json format
            dataset_dict = dataset.to_dict()
            with open(split.split(".csv")[0] + "_preprocessed.json", 'w') as f:
                json.dump(dataset_dict, f)

    return dataset