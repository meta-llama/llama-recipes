from typing import Any, Dict, List, Optional, Union
import time
import torch
from torch.utils.flop_counter import FlopCounterMode


class FlopMeasure(FlopCounterMode):
    """
    ``FlopMeasure`` is a customized context manager that counts the number of
    flops within its context. It is based on ``FlopCounterMode`` with additional start_counting() and stop_counting() function so that the flop counting
    will only start after the warmup stage.
    It also supports hierarchical output by passing a module (or list of modules) to FlopCounterMode on construction.

    Example usage

    .. code-block:: python

        model = ...
        flop_counter = FlopMeasure(model,local_rank=0,warmup_step=3)
        for batch in enumerate(dataloader):
            with flop_counter:
                model(batch)
                flop_counter.step()
    """

    def __init__(
        self,
        mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
        depth: int = 2,
        display: bool = True,
        custom_mapping: Dict[Any, Any] = None,
        rank=None,
        warmup_step: int = 3,
    ):
        super().__init__(mods, depth, display, custom_mapping)
        self.rank = rank
        self.warmup_step = warmup_step
        self.start_time = 0
        self.end_time = 0

    def step(self):
        # decrease the warmup step by 1 for every step, so that the flop counting will start when warmup_step =0. Stop decreasing when warm_up reaches -1.
        if self.warmup_step >= 0:
            self.warmup_step -= 1
        if self.warmup_step == 0 and self.start_time == 0:
            self.start_time = time.time()
        elif self.warmup_step == -1 and self.start_time != 0 and self.end_time == 0:
            self.end_time = time.time()
    def __enter__(self):
        if self.warmup_step == 0:
            self.start_time = time.time()
        super().__enter__()
        return self
    def is_done(self):
        return self.warmup_step == -1
    def get_total_flops(self):
        return super().get_total_flops()
    def get_flops_per_sec(self):
        if self.start_time == 0 or self.end_time == 0:
            print("Warning: flop count did not finish correctly")
            return 0
        return super().get_total_flops()/ (self.end_time - self.start_time)
    def get_table(self, depth=2):
        return super().get_table(depth)

    def __exit__(self, *args):
        if self.get_total_flops() == 0:
            print(
                "Warning: did not record any flops this time. Skipping the flop report"
            )
        else:
            if self.display:
                if self.rank is None or self.rank == 0:
                    print("Total time used in this flop counting step is: {}".format(self.end_time - self.start_time))
                    print("The total TFlop per second is: {}".format(self.get_flops_per_sec() / 1e12))
                    print("The tflop_count table is below:")
                    print(self.get_table(self.depth))
            # Disable the display feature so that we don't print the table again
            self.display = False
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # when warmup_step is 0, count the flops and return the original output
        if self.warmup_step == 0:
            return super().__torch_dispatch__(func, types, args, kwargs)
        # otherwise, just return the original output
        return func(*args, **kwargs)
