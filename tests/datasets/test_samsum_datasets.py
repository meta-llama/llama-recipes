# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from unittest.mock import patch


@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.LlamaTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_custom_dataset(step_lr, optimizer, tokenizer, get_model, train, mocker):
    from llama_recipes.finetuning import main
        
    tokenizer.return_value = mocker.MagicMock(side_effect=lambda x: {"input_ids":[len(x)*[0,]], "attention_mask": [len(x)*[0,]]})
    
    
    kwargs = {
        "batch_size_training": 1,
        "use_peft": False,
        "dataset": "samsum_dataset",
        }
    
    main(**kwargs)
    
    assert train.call_count == 1
    
    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    
    VAL_SAMPLES = 818
    TRAIN_SAMPLES = 14732
    CONCAT_SIZE = 2048
    assert len(train_dataloader) == TRAIN_SAMPLES // CONCAT_SIZE
    assert len(eval_dataloader) == VAL_SAMPLES
    