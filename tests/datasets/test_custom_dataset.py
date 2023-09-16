# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from unittest.mock import patch


@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_custom_dataset(step_lr, optimizer, get_model, train, mocker):
    from llama_recipes.finetuning import main

    kwargs = {
        "dataset": "custom_dataset",
        "model_name": "decapoda-research/llama-7b-hf", # We use the tokenizer as a surrogate for llama2 tokenizer here
        "custom_dataset.file": "examples/custom_dataset.py",
        "custom_dataset.train_split": "validation",
        "batch_size_training": 2,
        "val_batch_size": 4,
        "use_peft": False,
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
    STRING = tokenizer.decode(next(it)["input_ids"][0], skip_special_tokens=True)
    EXPECTED_STRING = "[INST] Who made Berlin [/INST] dunno"
    assert STRING.startswith(EXPECTED_STRING)
    
    assert next(it)["input_ids"].size(0) == 4

    next(it)
    next(it)
    STRING = tokenizer.decode(next(it)["input_ids"][0], skip_special_tokens=True)
    EXPECTED_STRING = "[INST] Implementa el algoritmo `bubble sort` en C. [/INST] xdxdxd"
    assert STRING.startswith(EXPECTED_STRING)
    
    assert "labels" in next(iter(train_dataloader)).keys()
    assert "input_ids" in next(iter(train_dataloader)).keys()
    assert "attention_mask" in next(iter(train_dataloader)).keys()


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
