# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions
from torch.distributed.checkpoint.state_dict_saver import save
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)

from torch.distributed.fsdp import (
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    FullyShardedDataParallel as FSDP,
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    StateDictType,
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_fsdp_checkpoint_sharded(model, cfg, epoch=1, optimizer=None):
    rank = dist.get_rank()
    folder_name = "-".join((cfg.dist_checkpoint_folder, cfg.model_name, str(epoch)))

    load_dir = Path.cwd() / cfg.dist_checkpoint_root_folder / folder_name

    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory at {load_dir.as_posix()} found...skipping")
        return
    if rank == 0:
        print(f"loading model from model path: {load_dir.as_posix()} ")
    reader = FileSystemReader(load_dir)

    checkpoint = {"model": model}
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer
    if rank == 0:
        ck = checkpoint.keys()
        print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")

    load(
        state_dict=checkpoint,
        storage_reader=reader,
    )
    if rank == 0:
        print(f"checkpoint after load_state_dict()")
        ck = checkpoint.keys()
        print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")

    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_fsdp_checkpoint_sharded(model, optimizer, train_config, epoch=1):
    """save model and optimizer via sharded_state_dict to save_dir"""

    folder_name = "-".join((train_config.dist_checkpoint_folder, train_config.model_name, str(epoch)))

    save_dir = Path.cwd() / train_config.dist_checkpoint_root_folder / folder_name

    rank = dist.get_rank()

    if rank == 0:
        print(f"Saving model to {save_dir.as_posix()}")

    distributed_writer = FileSystemWriter(
        save_dir,
        overwrite=True,
    )
    t0 = time.perf_counter()

    options = StateDictOptions(
        full_state_dict=False,
    )

    optim = optimizer if train_config.save_optimizer else []

    state_dict = {"model": model}
    if train_config.save_optimizer:
        state_dict["optimizer"] = optimizer

    save(
        state_dict=state_dict,
        storage_writer=distributed_writer,
        planner=DefaultSavePlanner(),
    )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir.as_posix()}")
        print(f"Checkpoint Time = {t1-t0:.4f}\n")


def save_fsdp_checkpoint_full(
    model,
    optimizer,
    train_config,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""

    options = StateDictOptions(
        full_state_dict=True,
    )

    optim = optimizer if train_config.save_optimizer else []

    model_state, optim_state = get_state_dict(model, optim, options=options)

    rank = dist.get_rank()

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        folder_name = "-".join((train_config.dist_checkpoint_folder, train_config.model_name))
        save_dir = Path.cwd() / train_config.dist_checkpoint_root_folder / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        save_name = train_config.model_name.replace("/", "--") + "-" + str(epoch) + ".pt"
        save_full_path = save_dir / save_name

        # save model
        torch.save(model_state, save_full_path)

        print(f"model checkpoint saved for epoch {epoch} at {save_full_path.as_posix()}\n")

        if not train_config.save_optimizer:
            return

        opt_save_name = "optimizer" + "-" + train_config.model_name.replace("/", "--") + "-" + str(epoch) + ".pt"
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path.as_posix()} to disk")


def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return

    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    print(f"model checkpoint loaded to rank0 cpu")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """

    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    print(f"optimizer shard loaded on rank {rank}")


def load_sharded_model_single_gpu(model, model_path):

    reader = FileSystemReader(model_path)

    state_dict = {"model": model.state_dict()}

    load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(model_path),
        no_dist=True,
    )

    model.load_state_dict(state_dict["model"])

    print(f"Sharded state checkpoint loaded from {model_path}")
    return model


def save_peft_checkpoint(model, train_config):
    """save_pretrained peft model"""
    if train_config.enable_fsdp:
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )

        model_state, _ = get_state_dict(model, [], options=options)

        rank = dist.get_rank()
        if rank == 0:
            model_path = train_config.output_dir
            model.save_pretrained(model_path, state_dict=model_state)
    else:
        model.save_pretrained(model_path)


def save_model_checkpoint(model, output_dir):
    """save model when not peft and on single device"""

    output_file = Path(output_dir) / "model.pt"

    state_dict = model.state_dict()

    torch.save(state_dict, output_file)


def save_checkpoint(model, optimizer, train_config, fsdp_config, epoch):
    """save model and optimizer"""
    rank = dist.get_rank() if train_config.enable_fsdp else 0

    if train_config.enable_fsdp:
        dist.barrier()
    if train_config.use_peft:
        if rank == 0:
            print(f"we are about to save the PEFT modules")
        save_peft_checkpoint(model, train_config)
        
        if rank == 0:
            print(f"PEFT modules are saved in {train_config.output_dir} directory")

    else:
        if not train_config.enable_fsdp:
            save_model_checkpoint(model, train_config.output_dir)

        elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
            if rank == 0:
                print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
                print("=====================================================")
            save_fsdp_checkpoint_full(
                model, optimizer, train_config, epoch=epoch
            )

        elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            if rank == 0:
                print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                print("=====================================================")
            save_fsdp_checkpoint_sharded(
                model, optimizer, train_config, epoch=epoch
            )

    if train_config.enable_fsdp:
        dist.barrier()
