# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# AnyPrecisionAdamW: a flexible precision AdamW optimizer
# with optional Kahan summation for high precision weight updates.
# Allows direct control over momentum, variance and auxiliary compensation
# buffer dtypes.
# Optional Kahan summation is used to offset precision reduction for
# the weight updates. This allows full training in BFloat16 (equal or
# better than FP32 results in many cases) due to high precision weight upates.

import torch
from torch.optim.optimizer import Optimizer


class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=False,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
        """
        Args:
                params (iterable): iterable of parameters to optimize or dicts defining
                    parameter groups
                lr (float, optional): learning rate (default: 1e-3)
                betas (Tuple[float, float], optional): coefficients used for computing
                    running averages of gradient and its square (default: (0.9, 0.999))
                eps (float, optional): term added to the denominator to improve
                    numerical stability (default: 1e-8)
                weight_decay (float, optional): weight decay coefficient (default: 1e-2)

                # Any Precision specific
                use_kahan_summation = creates auxiliary buffer to ensure high precision
                model param updates (default: False)
                momentum_dtype = dtype for momentum  (default: BFloat32)
                variance_dtype = dtype for uncentered variance (default: BFloat16)
                compensation_buffer_dtype  = dtype for Kahan summation
                                             buffer (default: BFloat16)

                # Usage
                This optimizer implements optimizer states, and Kahan summation
                for high precision updates, all in user controlled dtypes.
                Defaults are variance in BF16, Momentum in FP32.
                This can be run in FSDP mixed precision, amp, or full precision,
                depending on what training pipeline you wish to work with.

                Setting to use_kahan_summation = False, and changing momentum and
                variance dtypes to FP32, reverts this to a standard AdamW optimizer.

        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_kahan_summation=use_kahan_summation,
            momentum_dtype=momentum_dtype,
            variance_dtype=variance_dtype,
            compensation_buffer_dtype=compensation_buffer_dtype,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                # to fix linter, we do not keep the returned loss for use atm.
                closure()

        for group in self.param_groups:

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AnyPrecisionAdamW does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:

                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        dtype=momentum_dtype,
                    )

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        dtype=variance_dtype,
                    )

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p,
                            dtype=compensation_buffer_dtype,
                        )

                # main processing -------------------------

                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad = p.grad

                # weight decay, AdamW style
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                # update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # update uncentered variance
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # adjust using bias1
                bias_correction1 = 1 - beta1**step

                step_size = lr / bias_correction1

                # adjust using bias2
                denom_correction = (1 - beta2**step) ** 0.5  # avoids math import

                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(
                    eps, alpha=1
                )

                # lr update to compensation
                if use_kahan_summation:
                    compensation = state["compensation"]

                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))

                else:
                    # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)