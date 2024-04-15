from typing import Any, Dict, List, Optional, Union

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

        mod = ...
        flop_counter = FlopMeasure(mod)
        for batch in enumerate(dataloader):
            with flop_counter:
                if step == 3:
                    flop_counter.start_counting()
                mod(batch)
                flop_counter.stop_counting()
    """

    def __init__(
        self,
        mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
        depth: int = 2,
        display: bool = True,
        custom_mapping: Dict[Any, Any] = None,
        rank=None,
    ):
        super().__init__(mods, depth, display, custom_mapping)
        self.ready = False
        self.rank = rank

    def __enter__(self):
        self.ready = False
        super().__enter__()
        return self

    def get_total_flops(self):
        return super().get_total_flops()

    def get_table(self, depth=2):
        return super().get_table(depth)

    def __exit__(self, *args):
        self.ready = False
        if self.get_total_flops() == 0:
            print(
                "Warning: did not record any flops this time. Skipping the flop report"
            )
        else:
            self.stop_counting()
            if self.display:
                if self.rank is None or self.rank == 0:
                    print("self.flop_counts", self.get_total_flops())
                    print(self.get_table(self.depth))
            # Disable the display feature so that we don't print the table again
            self.display = False
        super().__exit__(*args)

    def start_counting(self):
        self.ready = True

    def is_ready(self):
        return self.ready

    def stop_counting(self):
        self.ready = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # return the original output if not ready
        if not self.ready:
            return func(*args, **kwargs)
        # otherwise, count the flops and return the original output
        return super().__torch_dispatch__(func, types, args, kwargs)
