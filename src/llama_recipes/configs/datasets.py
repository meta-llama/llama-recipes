# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    trust_remote_code: bool = False


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"

@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "recipes/quickstart/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""
    
@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class custom_dataset_celoss_testing:
    dataset: str = "custom_dataset_celoss_testing"
    file: str = "src/llama_recipes/datasets/opinionqa_dataset.py:get_preprocessed_opinionqa_ce_or_wd_loss"
    train_split: str = "src/llama_recipes/datasets/cross_entropy_testing/opinionqa_train.csv"
    valid_split: str = "src/llama_recipes/datasets/cross_entropy_testing/opinionqa_validation.csv"
    test_split: str = "src/llama_recipes/datasets/cross_entropy_testing/opinionqa_test.csv"

@dataclass
class custom_dataset_wdloss_testing:
    dataset: str = "custom_dataset_wdloss_testing"
    file: str = "src/llama_recipes/datasets/opinionqa_dataset.py:get_preprocessed_opinionqa_ce_or_wd_loss"
    train_split: str = "src/llama_recipes/datasets/wd_loss_testing/opinionqa_train.csv"
    valid_split: str = "src/llama_recipes/datasets/wd_loss_testing/opinionqa_validation.csv"
    test_split: str = "src/llama_recipes/datasets/wd_loss_testing/opinionqa_test.csv"

@dataclass
class custom_dataset_leave_wave_34_testing:
    dataset: str = "custom_dataset_leave_wave_34_testing"
    file: str = "src/llama_recipes/datasets/opinionqa_dataset.py:get_preprocessed_opinionqa_ce_or_wd_loss"
    train_split: str = "src/llama_recipes/datasets/leave_wave_34_testing/opnqa_500_QA_train.csv"
    valid_split: str = "src/llama_recipes/datasets/leave_wave_34_testing/opnqa_500_QA_val.csv"
    test_split: str = "src/llama_recipes/datasets/leave_wave_34_testing/opnqa_500_QA_test.csv"

@dataclass
class custom_dataset_leave_wave_34_testing_all_steering_method:
    dataset: str = "custom_dataset_leave_wave_34_testing_all_steering_method"
    file: str = "src/llama_recipes/datasets/opinionqa_dataset.py:get_preprocessed_opinionqa_ce_or_wd_loss"
    train_split: str = "src/llama_recipes/datasets/leave_wave_34_testing/opnqa_500_ALL_train.csv"
    valid_split: str = "src/llama_recipes/datasets/leave_wave_34_testing/opnqa_500_ALL_val.csv"
    test_split: str = "src/llama_recipes/datasets/leave_wave_34_testing/opnqa_500_ALL_test.csv"