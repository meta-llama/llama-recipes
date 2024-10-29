# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
 

# requires grad scaler in main loop
fpSixteen = MixedPrecisionPolicy(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
)

bfSixteen = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)

bfSixteen_mixed = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
)


def get_mixed_precision_policies(cfg):
    """Get the policies for mixed precision and fsdp wrapping"""

    rank = dist.get_rank()

    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and torch.version.cuda >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    ) or (is_xpu_available())

    mixed_precision_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            if rank == 0:
                print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    return mixed_precision_policy
