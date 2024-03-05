# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="codellama/CodeLlama-13b-Instruct-hf"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=False
    batch_size_training: int=1
    batching_strategy: str="padding" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=8
    num_workers_dataloader: int=0
    lr: float=5e-5
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "hl_mr_dataset"
    n_gpu = 4
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=True
    output_dir: str = "/home/ubuntu/llama-recipes-fork/llama-recipes/src/llama_recipes/models/testing"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = True
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="models/fsdp" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    dataset_path: str = "HelixAI/hl-text-standard-json-single-turn-2024-03-04"
