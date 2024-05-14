# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from unittest.mock import patch

import pytest

import torch
from llama_recipes.data.sampler import LengthBasedBatchSampler

from llama_recipes.finetuning import main
from pytest import approx
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler


def get_fake_dataset():
    return [
        {
            "input_ids": [1],
            "attention_mask": [1],
            "labels": [1],
        }
    ]


@patch("llama_recipes.finetuning.torch.cuda.is_available")
@patch("llama_recipes.finetuning.train")
@patch("llama_recipes.finetuning.LlamaForCausalLM.from_pretrained")
@patch("llama_recipes.finetuning.AutoTokenizer.from_pretrained")
@patch("llama_recipes.finetuning.get_preprocessed_dataset")
@patch("llama_recipes.finetuning.optim.AdamW")
@patch("llama_recipes.finetuning.StepLR")
@pytest.mark.parametrize("cuda_is_available", [True, False])
def test_finetuning_no_validation(
    step_lr,
    optimizer,
    get_dataset,
    tokenizer,
    get_model,
    train,
    cuda,
    cuda_is_available,
):
    kwargs = {"run_validation": False}

    get_dataset.return_value = get_fake_dataset()
    cuda.return_value = cuda_is_available

    get_model.return_value.get_input_embeddings.return_value.weight.shape = [0]

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]

    assert isinstance(train_dataloader, DataLoader)
    assert eval_dataloader is None

    if cuda_is_available:
        assert get_model.return_value.to.call_count == 1
        assert get_model.return_value.to.call_args.args[0] == "cuda"
    else:
        assert get_model.return_value.to.call_count == 0


@patch("llama_recipes.finetuning.torch.cuda.is_available")
@patch("llama_recipes.finetuning.train")
@patch("llama_recipes.finetuning.LlamaForCausalLM.from_pretrained")
@patch("llama_recipes.finetuning.AutoTokenizer.from_pretrained")
@patch("llama_recipes.finetuning.get_preprocessed_dataset")
@patch("llama_recipes.finetuning.optim.AdamW")
@patch("llama_recipes.finetuning.StepLR")
@pytest.mark.parametrize("cuda_is_available", [True, False])
def test_finetuning_with_validation(
    step_lr,
    optimizer,
    get_dataset,
    tokenizer,
    get_model,
    train,
    cuda,
    cuda_is_available,
):
    kwargs = {"run_validation": True}

    get_dataset.return_value = get_fake_dataset()
    cuda.return_value = cuda_is_available

    get_model.return_value.get_input_embeddings.return_value.weight.shape = [0]

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(eval_dataloader, DataLoader)

    if cuda_is_available:
        assert get_model.return_value.to.call_count == 1
        assert get_model.return_value.to.call_args.args[0] == "cuda"
    else:
        assert get_model.return_value.to.call_count == 0


@patch("llama_recipes.finetuning.torch.cuda.is_available")
@patch("llama_recipes.finetuning.train")
@patch("llama_recipes.finetuning.LlamaForCausalLM.from_pretrained")
@patch("llama_recipes.finetuning.AutoTokenizer.from_pretrained")
@patch("llama_recipes.finetuning.get_preprocessed_dataset")
@patch("llama_recipes.finetuning.generate_peft_config")
@patch("llama_recipes.finetuning.get_peft_model")
@patch("llama_recipes.finetuning.optim.AdamW")
@patch("llama_recipes.finetuning.StepLR")
@pytest.mark.parametrize("cuda_is_available", [True, False])
def test_finetuning_peft_lora(
    step_lr,
    optimizer,
    get_peft_model,
    gen_peft_config,
    get_dataset,
    tokenizer,
    get_model,
    train,
    cuda,
    cuda_is_available,
):
    kwargs = {"use_peft": True}

    get_dataset.return_value = get_fake_dataset()
    cuda.return_value = cuda_is_available

    get_model.return_value.get_input_embeddings.return_value.weight.shape = [0]

    main(**kwargs)

    if cuda_is_available:
        assert get_peft_model.return_value.to.call_count == 1
        assert get_peft_model.return_value.to.call_args.args[0] == "cuda"
    else:
        assert get_peft_model.return_value.to.call_count == 0

    assert get_peft_model.return_value.print_trainable_parameters.call_count == 1


@patch("llama_recipes.finetuning.get_peft_model")
@patch("llama_recipes.finetuning.setup")
@patch("llama_recipes.finetuning.train")
@patch("llama_recipes.finetuning.LlamaForCausalLM.from_pretrained")
@patch("llama_recipes.finetuning.AutoTokenizer.from_pretrained")
@patch("llama_recipes.finetuning.get_preprocessed_dataset")
def test_finetuning_peft_llama_adapter(
    get_dataset, tokenizer, get_model, train, setup, get_peft_model
):
    kwargs = {
        "use_peft": True,
        "peft_method": "llama_adapter",
        "enable_fsdp": True,
    }

    get_dataset.return_value = get_fake_dataset()

    get_model.return_value.get_input_embeddings.return_value.weight.shape = [0]

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    with pytest.raises(
        RuntimeError,
        match="Llama_adapter is currently not supported in combination with FSDP",
    ):
        main(**kwargs)

    GET_ME_OUT = "Get me out of here"
    get_peft_model.side_effect = RuntimeError(GET_ME_OUT)

    kwargs["enable_fsdp"] = False

    with pytest.raises(
        RuntimeError,
        match=GET_ME_OUT,
    ):
        main(**kwargs)


@patch("llama_recipes.finetuning.train")
@patch("llama_recipes.finetuning.LlamaForCausalLM.from_pretrained")
@patch("llama_recipes.finetuning.AutoTokenizer.from_pretrained")
@patch("llama_recipes.finetuning.get_preprocessed_dataset")
@patch("llama_recipes.finetuning.get_peft_model")
@patch("llama_recipes.finetuning.StepLR")
def test_finetuning_weight_decay(
    step_lr, get_peft_model, get_dataset, tokenizer, get_model, train
):
    kwargs = {"weight_decay": 0.01}

    get_dataset.return_value = get_fake_dataset()

    get_model.return_value.parameters.return_value = [torch.ones(1, 1)]
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [0]

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    optimizer = args[4]

    print(optimizer.state_dict())

    assert isinstance(optimizer, AdamW)
    assert optimizer.state_dict()["param_groups"][0]["weight_decay"] == approx(0.01)


@patch("llama_recipes.finetuning.train")
@patch("llama_recipes.finetuning.LlamaForCausalLM.from_pretrained")
@patch("llama_recipes.finetuning.AutoTokenizer.from_pretrained")
@patch("llama_recipes.finetuning.get_preprocessed_dataset")
@patch("llama_recipes.finetuning.optim.AdamW")
@patch("llama_recipes.finetuning.StepLR")
def test_batching_strategy(
    step_lr, optimizer, get_dataset, tokenizer, get_model, train
):
    kwargs = {"batching_strategy": "packing"}

    get_dataset.return_value = get_fake_dataset()

    get_model.return_value.get_input_embeddings.return_value.weight.shape = [0]

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader, eval_dataloader = args[1:3]
    assert isinstance(train_dataloader.batch_sampler, BatchSampler)
    assert isinstance(eval_dataloader.batch_sampler, BatchSampler)

    kwargs["batching_strategy"] = "padding"
    train.reset_mock()
    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader, eval_dataloader = args[1:3]
    assert isinstance(train_dataloader.batch_sampler, LengthBasedBatchSampler)
    assert isinstance(eval_dataloader.batch_sampler, LengthBasedBatchSampler)

    kwargs["batching_strategy"] = "none"

    with pytest.raises(ValueError):
        main(**kwargs)
