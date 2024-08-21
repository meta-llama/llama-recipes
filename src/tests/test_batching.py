# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from unittest.mock import patch

EXPECTED_SAMPLE_NUMBER ={
    "meta-llama/Llama-2-7b-hf": {
        "train": 96,
        "eval": 42,
    },
    "meta-llama/Meta-Llama-3.1-8B": {
        "train": 79,
        "eval": 34,
    }
}

@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.AutoTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_packing(step_lr, optimizer, get_model, tokenizer, train, setup_tokenizer, llama_version):
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]
    
    kwargs = {
        "model_name": llama_version,
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

    assert len(train_dataloader) == EXPECTED_SAMPLE_NUMBER[llama_version]["train"]
    assert len(eval_dataloader) == EXPECTED_SAMPLE_NUMBER[llama_version]["eval"]

    batch = next(iter(train_dataloader))

    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert batch["labels"][0].size(0) == 4096
    assert batch["input_ids"][0].size(0) == 4096
    assert batch["attention_mask"][0].size(0) == 4096


@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.AutoTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
@patch('llama_recipes.finetuning.setup')
@patch('llama_recipes.finetuning.FSDP')
@patch('llama_recipes.finetuning.torch.distributed.is_initialized')
@patch('llama_recipes.utils.config_utils.dist')
def test_distributed_packing(dist, is_initialized, fsdp, setup, step_lr, optimizer, get_model, tokenizer, train, setup_tokenizer, llama_version):
    import os
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]

    rank = 1
    os.environ['LOCAL_RANK'] = f'{rank}'
    os.environ['RANK'] = f'{rank}'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    kwargs = {
        "model_name": llama_version,
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

    assert len(train_dataloader) == EXPECTED_SAMPLE_NUMBER[llama_version]["train"] //2
    assert len(eval_dataloader) == EXPECTED_SAMPLE_NUMBER[llama_version]["eval"] //2
