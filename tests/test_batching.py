# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from unittest.mock import patch


@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_packing(step_lr, optimizer, get_model, tokenizer, train, mocker, setup_tokenizer):
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)

    kwargs = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "batch_size_training": 8,
        "val_batch_size": 1,
        "use_peft": False,
        "dataset": "samsum_dataset",
        "batching_strategy": "packing",
        }

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]

    assert len(train_dataloader) == 96
    assert len(eval_dataloader) == 42

    batch = next(iter(train_dataloader))

    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert batch["labels"][0].size(0) == 4096
    assert batch["input_ids"][0].size(0) == 4096
    assert batch["attention_mask"][0].size(0) == 4096


@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
@patch('llama_recipes.finetuning.setup')
@patch('llama_recipes.finetuning.FSDP')
@patch('llama_recipes.finetuning.torch.distributed.is_initialized')
@patch('llama_recipes.utils.config_utils.dist')
def test_distributed_packing(dist, is_initialized, fsdp, setup, step_lr, optimizer, get_model, tokenizer, train, setup_tokenizer):
    import os
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)

    rank = 0
    os.environ['LOCAL_RANK'] = f'{rank}'
    os.environ['RANK'] = f'{rank}'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    kwargs = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "batch_size_training": 8,
        "val_batch_size": 1,
        "use_peft": False,
        "dataset": "samsum_dataset",
        "batching_strategy": "packing",
        "enable_fsdp": True
        }

    is_initialized.return_value = True
    dist.get_rank.return_value = rank
    dist.get_world_size.return_value = 2

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]

    assert len(train_dataloader) == 96 //2
    assert len(eval_dataloader) == 42 //2
