# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from unittest.mock import patch
import importlib

from torch.utils.data.dataloader import DataLoader

from llama_recipes.finetuning import main

@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.LlamaTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.get_preprocessed_dataset')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_finetuning_no_validation(step_lr, optimizer, get_dataset, tokenizer, get_model, train):
    kwargs = {"run_validation": False}
    
    get_dataset.return_value = [1]
    
    main(**kwargs)
    
    assert train.call_count == 1
    
    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    
    assert isinstance(train_dataloader, DataLoader)
    assert eval_dataloader is None
    
    assert get_model.return_value.to.call_args.args[0] == "cuda"
    
    
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.LlamaTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.get_preprocessed_dataset')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_finetuning_with_validation(step_lr, optimizer, get_dataset, tokenizer, get_model, train):
    kwargs = {"run_validation": True}
    get_dataset.return_value = [1]
    
    main(**kwargs)
    
    assert train.call_count == 1
    
    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)
    
    assert get_model.return_value.to.call_args.args[0] == "cuda"
    
    
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.LlamaTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.get_preprocessed_dataset')
@patch('llama_recipes.finetuning.generate_peft_config')
@patch('llama_recipes.finetuning.get_peft_model')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_finetuning_peft(step_lr, optimizer, get_peft_model, gen_peft_config, get_dataset, tokenizer, get_model, train):
    kwargs = {"use_peft": True}
    
    get_dataset.return_value = [1]
    
    main(**kwargs)
    
    assert get_peft_model.return_value.to.call_args.args[0] == "cuda"
    assert get_peft_model.return_value.print_trainable_parameters.call_count == 1