# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from unittest.mock import patch

from transformers import LlamaTokenizer


@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_samsum_dataset(step_lr, optimizer, get_model, tokenizer, train, mocker):
    from llama_recipes.finetuning import main

    #Align with Llama 2 tokenizer
    tokenizer.from_pretrained.return_value = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer.from_pretrained.return_value.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    tokenizer.from_pretrained.return_value.bos_token_id = 1
    tokenizer.from_pretrained.return_value.eos_token_id = 2

    BATCH_SIZE = 8
    kwargs = {
        "model_name": "decapoda-research/llama-7b-hf",
        "batch_size_training": 8,
        "val_batch_size": 1,
        "use_peft": False,
        "dataset": "samsum_dataset",
        "batching_strategy": "padding",
        }

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]

    VAL_SAMPLES = 818
    TRAIN_SAMPLES = 14732

    assert len(train_dataloader) == TRAIN_SAMPLES // BATCH_SIZE
    assert len(eval_dataloader) == VAL_SAMPLES

    batch = next(iter(train_dataloader))

    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert batch["labels"][0][268] == -100
    assert batch["labels"][0][269] == 22291

    assert batch["input_ids"][0][0] == 1
    assert batch["labels"][0][-1] == 2
    assert batch["input_ids"][0][-1] == 2


@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_samsum_dataset_packing(step_lr, optimizer, get_model, tokenizer, train, mocker):
    from llama_recipes.finetuning import main

    #Align with Llama 2 tokenizer
    tokenizer.from_pretrained.return_value = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer.from_pretrained.return_value.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    tokenizer.from_pretrained.return_value.bos_token_id = 1
    tokenizer.from_pretrained.return_value.eos_token_id = 2

    BATCH_SIZE = 8
    kwargs = {
        "model_name": "decapoda-research/llama-7b-hf",
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
