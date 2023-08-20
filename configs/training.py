# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    model_name: str="meta-llama-2-7b"
    model_path: str="/opt/ml/input/data/model"
    tokenizer_name: str="/opt/ml/input/data/tokenizer"
    enable_fsdp: bool=True
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=32
    num_epochs: int=1
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=False
    val_batch_size: int=2
    dataset = "tokenized_dataset"
    micro_batch_size: int=4
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "/opt/ml/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="/opt/ml" # will be used if using FSDP
    dist_checkpoint_folder: str="checkpoints" # will be used if using FSDP
    save_optimizer: bool=True # will be used if using FSDP
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels

    
    
    