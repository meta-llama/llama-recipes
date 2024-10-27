# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from contextlib import nullcontext
from dataclasses import dataclass
from datasets import Dataset
from unittest.mock import patch

@dataclass
class Config:
    model_type: str = "llama"

EXPECTED_SAMPLE_NUMBER ={
    "meta-llama/Llama-2-7b-hf": {
        "train": 4,
        "eval": 37,
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "train": 3,
        "eval": 30,
    },
    "fake_llama": {
        "train": 2,
        "eval": 17,
    }
}

fake_samsum_dataset = 2048*[{'id': '420',
 'dialogue': "Mario: It's a me, Mario!\nLuigi: It's a me, your brother!\nMario: I'm going to save the princess.\nLuigi: I'm going to help Mario.",
 'summary': 'Mario and Luigi are going to save the princess.'}]

@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.AutoTokenizer')
@patch("llama_recipes.finetuning.AutoConfig.from_pretrained")
@patch("llama_recipes.finetuning.AutoProcessor")
@patch("llama_recipes.finetuning.MllamaForConditionalGeneration.from_pretrained")
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
@patch('llama_recipes.datasets.samsum_dataset.datasets')
def test_packing(
    datasets,
    step_lr,
    optimizer,
    get_model,
    get_mmodel,
    processor,
    get_config,
    tokenizer,
    train,
    setup_tokenizer,
    setup_processor,
    llama_version,
    model_type,
    ):
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)
    setup_processor(processor)
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]
    get_mmodel.return_value.get_input_embeddings.return_value.weight.shape = [0]
    get_config.return_value = Config(model_type=model_type)

    datasets.load_dataset.return_value = Dataset.from_list(fake_samsum_dataset)
    
    kwargs = {
        "model_name": llama_version,
        "batch_size_training": 8,
        "val_batch_size": 1,
        "use_peft": False,
        "dataset": "samsum_dataset",
        "batching_strategy": "packing",
        }

    c = nullcontext() if model_type == "llama" else  pytest.raises(ValueError)

    with c:
        main(**kwargs)
    
    if model_type == "llama":
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
@patch("llama_recipes.utils.train_utils.torch.cuda.is_bf16_supported")
@patch("llama_recipes.finetuning.torch.cuda.is_available")
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.AutoTokenizer')
@patch("llama_recipes.finetuning.AutoConfig.from_pretrained")
@patch("llama_recipes.finetuning.AutoProcessor")
@patch("llama_recipes.finetuning.MllamaForConditionalGeneration.from_pretrained")
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
@patch('llama_recipes.finetuning.setup')
@patch('llama_recipes.finetuning.FSDP')
@patch('llama_recipes.finetuning.torch.distributed.is_initialized')
@patch('llama_recipes.utils.config_utils.dist')
@patch('llama_recipes.datasets.samsum_dataset.datasets')
def test_distributed_packing(
    datasets,
    dist,
    is_initialized,
    fsdp,
    setup,
    step_lr,
    optimizer,
    get_model,
    get_mmodel,
    processor,
    get_config,
    tokenizer,
    train,
    cuda_is_available,
    cuda_is_bf16_supported,
    setup_tokenizer,
    setup_processor,
    llama_version,
    model_type,
    ):
    import os
    from llama_recipes.finetuning import main

    setup_tokenizer(tokenizer)
    setup_processor(processor)
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]
    get_mmodel.return_value.get_input_embeddings.return_value.weight.shape = [0]
    get_config.return_value = Config(model_type=model_type)
    cuda_is_available.return_value = False
    cuda_is_bf16_supported.return_value = False

    datasets.load_dataset.return_value = Dataset.from_list(fake_samsum_dataset)

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

    c = nullcontext() if model_type == "llama" else  pytest.raises(ValueError)

    with c:
        main(**kwargs)

    if model_type == "llama":
        assert train.call_count == 1

        args, kwargs = train.call_args
        train_dataloader = args[1]
        eval_dataloader = args[2]

        assert len(train_dataloader) == EXPECTED_SAMPLE_NUMBER[llama_version]["train"] //2
        assert len(eval_dataloader) == EXPECTED_SAMPLE_NUMBER[llama_version]["eval"] //2
