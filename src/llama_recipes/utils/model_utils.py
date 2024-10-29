# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
import torch.nn as nn
import torch.distributed as dist
from llama_recipes.configs import (
    quantization_config as QuantizationConfig,
    train_config as TrainConfig
)
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    LlamaForCausalLM,
    MllamaForConditionalGeneration,
)


def print_model_size(model: nn.Module, config: TrainConfig, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def get_model_and_data_processor(
    train_config: TrainConfig, quant_config: QuantizationConfig
):
    bnb_config = None
    if quant_config:
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    use_cache = False if train_config.enable_fsdp else None
    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "mllama":
        is_vision = True
        model = MllamaForConditionalGeneration.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(
            train_config.model_name
            if train_config.tokenizer_name is None
            else train_config.tokenizer_name
        )
        processor.tokenizer.padding_side = "right"
        model.supports_gradient_checkpointing = True
        model.language_model.supports_gradient_checkpointing = True
    elif config.model_type == "llama":
        is_vision = False
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )

        # Load the tokenizer and add special tokens
        processor = AutoTokenizer.from_pretrained(
            train_config.model_name
            if train_config.tokenizer_name is None
            else train_config.tokenizer_name
        )
        if not processor.pad_token_id:
            processor.pad_token_id = processor.eos_token_id

        # If there is a mismatch between tokenizer vocab size and embedding matrix,
        # throw a warning and then expand the embedding matrix
        if len(processor) > model.get_input_embeddings().weight.shape[0]:
            print(
                "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
            )
            model.resize_token_embeddings(len(processor))

    else:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use llama or mllama model."
        )

    print_model_size(
        model, train_config, dist.get_rank() if train_config.enable_fsdp else 0
    )

    return model, processor, is_vision
