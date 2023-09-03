import os
from configs import wandb_config

def wandb_watch(model, cfg: wandb_config):
    import wandb
    if wandb_config.wandb_token is not None:
        wandb.login(key=wandb_config.wandb_token)
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        id=cfg.wandb_id,
        mode=cfg.wandb_mode,
        dir=cfg.wandb_dir
    )
    wandb.watch(model)
    return run
