# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from unittest.mock import patch

import torch

from llama_recipes.utils.train_utils import train

@patch("llama_recipes.utils.train_utils.MemoryTrace")
@patch("llama_recipes.utils.train_utils.nullcontext")
@patch("llama_recipes.utils.train_utils.torch.cuda.amp.GradScaler")
@patch("llama_recipes.utils.train_utils.torch.cuda.amp.autocast")
def test_gradient_accumulation(autocast, scaler, nullcontext, mem_trace, mocker):
    
    model = mocker.MagicMock(name="model")
    model().loss.__truediv__().detach.return_value = torch.tensor(1)
    mock_tensor = mocker.MagicMock(name="tensor")
    batch = {"input": mock_tensor}
    train_dataloader = [batch, batch, batch, batch, batch]
    eval_dataloader = None
    tokenizer = mocker.MagicMock()
    optimizer = mocker.MagicMock()
    lr_scheduler = mocker.MagicMock()
    gradient_accumulation_steps = 1
    train_config = mocker.MagicMock()
    train_config.enable_fsdp = False
    train_config.use_fp16 = False
    train_config.run_validation = False
    
    train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config,
    )
    
    assert optimizer.zero_grad.call_count == 5
    optimizer.zero_grad.reset_mock()
    
    assert nullcontext.call_count == 5
    nullcontext.reset_mock()
    
    assert autocast.call_count == 0
    
    gradient_accumulation_steps = 2
    train_config.use_fp16 = True
    train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        lr_scheduler,
        gradient_accumulation_steps,
        train_config,
    )
    assert optimizer.zero_grad.call_count == 3
    assert nullcontext.call_count == 0
    assert autocast.call_count == 5