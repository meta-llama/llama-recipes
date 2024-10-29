# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os

import torch
import torch.nn as nn
from llama_recipes.configs.fsdp import fsdp_config as FSDP_CONFIG
from llama_recipes.policies import get_mixed_precision_policies
from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh
from typing import List, Callable


def hsdp_device_mesh(replica_group_size, sharding_group_size, device=None):
    """
     Initializes a device mesh for use with Hybrid Sharding strategy in FSDP (HSDP) training.

    This function requires explicit sizes for replica and sharding groups to accommodate models
    whose GPU fit is unknown, providing flexibility in distributed training setups.

    Args:
        replica_group_size (int): The size of each replica group. Must be provided to ensure
            the model fits within the available resources.
        sharding_group_size (int): The size of each sharding group that the model can fit. Must be provided to
            ensure the correct distribution of model parameters.
        device (str, optional): The device to use (e.g., "cuda:0"). If None, defaults to "cuda"
            with the local rank as the device index.

    Returns:
        A device mesh object compatible with FSDP.

    Raises:
        ValueError: If replica_group_size or sharding_group_size are not provided, or if the
            world size is not evenly divisible by the sharding group size.
        RuntimeError: If a valid device mesh cannot be created.

    Usage:
        If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups
        >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
        >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    """

    if replica_group_size is None or sharding_group_size is None:
        raise ValueError(
            "Both replica_group_size and sharding_group_size must be provided."
        )

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    device = device or f"cuda"

    if world_size % sharding_group_size != 0:
        raise ValueError(
            f"World size {world_size} is not evenly divisible by "
            f"sharding group size {sharding_group_size}."
        )

    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(
            f"The calculated number of replica groups is not evenly divisible by "
            f"replica_group_size {replica_group_size}."
        )

    device_mesh = init_device_mesh(device, (replica_group_size, sharding_group_size))
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")

    return device_mesh


def parallelize_model(
    model: nn.Module,
    fsdp_config: FSDP_CONFIG,
    device_mesh: DeviceMesh = None,
    sharding_conditions: List[Callable] = None,
) -> nn.Module:
    """
    Parallelizes a Llama model using FSDP.

    Args:
        model (nn.Module): The Llama model to parallelize.
        fsdp_config (FSDP_CONFIG): The FSDP configuration.
        device_mesh (torch.device_mesh): The device mesh to use for parallelization.

    Returns:
        None
    """

    mp_policy = get_mixed_precision_policies(fsdp_config)
    fsdp_config = {
        "mesh": device_mesh,
        "mp_policy": None if fsdp_config.pure_bf16 else mp_policy,
        "offload_policy": CPUOffloadPolicy() if fsdp_config.fsdp_cpu_offload else None
        }

    # Following torchtune's approach to wrap Lora first as dtype is different from base
    for m in reversed(list(model.modules())):
        if any(c(m) for c in sharding_conditions):
            fully_shard(m, reshard_after_forward=True)

    # 
    # if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
    #     for n, m in reversed(list(model.named_modules())):
    #         if any(c(m) for c in sharding_conditions):
    #         # if (
    #         #     len(list(m.named_children())) == 0
    #         #     and getattr(m, "weight", None) is not None
    #         #     and m.weight.requires_grad
    #         # ):
    #             fully_shard(m, reshard_after_forward=True)
    #     layers = model.base_model.model.model.layers
    # else:
    #     layers = model.model.layers

    # for idx, layer in enumerate(layers):
    #     # Following torch titan we will not reshard the last layer
    #     # https://github.com/pytorch/torchtitan/blob/7310abea8782bbe459b662bc6d8411fe8d55f62c/torchtitan/parallelisms/parallelize_llama.py#L347
    #     reshard_after_forward = idx < len(layers) - 1
    #     fully_shard(
    #         layer,
    #         reshard_after_forward=reshard_after_forward,
    #     )

    # Shard remaining modules like embeddings
    fully_shard(model, **fsdp_config)
