# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
import os
import collections
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from functools import partial

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)


def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


class FreeEventQueue:
    """https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_limiter_utils.py"""
    def __init__(self):
        self._queue = collections.deque()
        self._max_num_inflight_copy = 2

    def enqueue(self, free_event):
        self._queue.append(free_event)

    def dequeue_if_needed(self):
        if len(self._queue) >= self._max_num_inflight_copy:
            return self._dequeue()
        return None

    def _dequeue(self):
        if self._queue:
            event = self._queue.popleft()
            return event
        return None


class save_on_cpu_overlap(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self):
        copy_stream = torch.cuda.Stream()
        current_stream = torch.cuda.current_stream()
        pack_event_queue = FreeEventQueue()
        unpack_event_queue = FreeEventQueue()

        def _deque_event_and_synchronize(queue):
            event = queue.dequeue_if_needed()
            if event:
                event.synchronize()

        def _enque_event(queue):
            free_event = torch.cuda.Event()
            free_event.record()
            queue.enqueue(free_event)

        def pack_to_cpu(tensor):
            _deque_event_and_synchronize(pack_event_queue)
            copy_stream.wait_stream(current_stream)
            with torch.cuda.stream(copy_stream):
                packed = tensor.to("cpu", non_blocking=True)
            tensor.record_stream(copy_stream)
            #print(tensor.shape)
            _enque_event(pack_event_queue)
            return (tensor.device, packed)

        def unpack_from_cpu(packed):
            _deque_event_and_synchronize(unpack_event_queue)
            device, tensor = packed
            with torch.cuda.stream(copy_stream):
                unpacked = tensor.to(device, non_blocking=True)
            current_stream.wait_stream(copy_stream)
            unpacked.record_stream(current_stream)
            _enque_event(unpack_event_queue)
            return unpacked

        super().__init__(pack_to_cpu, unpack_from_cpu)

