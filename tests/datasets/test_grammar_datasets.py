# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from unittest.mock import patch


@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_grammar_dataset(step_lr, optimizer, get_model, train, mocker):
# def test_samsum_dataset(step_lr, optimizer, tokenizer, get_model, train, mocker):
    from llama_recipes.finetuning import main
    
    BATCH_SIZE = 8
    kwargs = {
        "model_name": "decapoda-research/llama-7b-hf",
        "batch_size_training": 8,
        "val_batch_size": 1,
        "use_peft": False,
        "dataset": "grammar_dataset",
        }
    
    main(**kwargs)
    
    assert train.call_count == 1
    
    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    
    VAL_SAMPLES = 2988
    TRAIN_SAMPLES = 13016
    
    assert len(train_dataloader) == TRAIN_SAMPLES // BATCH_SIZE
    assert len(eval_dataloader) == VAL_SAMPLES
    
    assert "labels" in next(iter(train_dataloader)).keys()
    assert "input_ids" in next(iter(train_dataloader)).keys()
    assert "attention_mask" in next(iter(train_dataloader)).keys()