# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field

@dataclass
class wandb_config:
    enable_wandb: bool = False
    wandb_id: str = None
    wandb_project: str = None
    wandb_entity: str = None
    wandb_mode: str = None
    wandb_dir: str = None
    wandb_token: str = None
    
    
