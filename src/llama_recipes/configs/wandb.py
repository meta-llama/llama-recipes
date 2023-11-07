# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field

@dataclass
class wandb_config:
    wandb_project: str='llama_recipes' # wandb project name
    wandb_entity: str='none' # wandb entity name
    wandb_log_model: bool=False # whether or not to log model as artifact at the end of training
    wandb_watch: str='false' # can be set to 'gradients' or 'all' to log gradients and parameters
