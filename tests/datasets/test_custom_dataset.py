# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from unittest.mock import patch

from transformers import LlamaTokenizer

def check_padded_entry(batch):
    seq_len = sum(batch["attention_mask"][0])
    assert seq_len < len(batch["attention_mask"][0])

    assert batch["labels"][0][0] == -100
    assert batch["labels"][0][seq_len-1] == 2
    assert batch["labels"][0][-1] == -100
    assert batch["input_ids"][0][0] == 1
    assert batch["input_ids"][0][-1] == 2


@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_custom_dataset(step_lr, optimizer, get_model, tokenizer, train, mocker, setup_tokenizer):
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)

    kwargs = {
        "dataset": "custom_dataset",
        "model_name": "meta-llama/Llama-2-7b-hf",
        "custom_dataset.file": "examples/custom_dataset.py",
        "custom_dataset.train_split": "validation",
        "batch_size_training": 2,
        "val_batch_size": 4,
        "use_peft": False,
        "batching_strategy": "padding"
        }

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    tokenizer = args[3]

    assert len(train_dataloader) == 1120
    assert len(eval_dataloader) == 1120 //2

    it = iter(eval_dataloader)
    batch = next(it)
    STRING = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
    EXPECTED_STRING = "[INST] Who made Berlin [/INST] dunno"
    assert STRING.startswith(EXPECTED_STRING)

    assert batch["input_ids"].size(0) == 4
    assert set(("labels", "input_ids", "attention_mask")) == set(batch.keys())

    check_padded_entry(batch)

    it = iter(train_dataloader)
    for _ in range(5):
        next(it)

    batch = next(it)
    STRING = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
    EXPECTED_STRING = "[INST] How do I initialize a Typescript project using npm and git? [/INST] # Initialize a new NPM project"
    assert STRING.startswith(EXPECTED_STRING)

    assert batch["input_ids"].size(0) == 2
    assert set(("labels", "input_ids", "attention_mask")) == set(batch.keys())

    check_padded_entry(batch)



@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.LlamaTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_unknown_dataset_error(step_lr, optimizer, tokenizer, get_model, train, mocker):
    from llama_recipes.finetuning import main

    tokenizer.return_value = mocker.MagicMock(side_effect=lambda x: {"input_ids":[len(x)*[0,]], "attention_mask": [len(x)*[0,]]})

    kwargs = {
        "dataset": "custom_dataset",
        "custom_dataset.file": "examples/custom_dataset.py:get_unknown_dataset",
        "batch_size_training": 1,
        "use_peft": False,
        }
    with pytest.raises(AttributeError):
        main(**kwargs)
