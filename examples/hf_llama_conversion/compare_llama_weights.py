# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import gc
import glob
import os
import sys

import torch
import tqdm


def main() -> None:
    """Compare two llama checkpoint directories"""

    one_files = sorted(glob.glob(os.path.join(sys.argv[1], "consolidated.*.pth")))
    two_files = sorted(glob.glob(os.path.join(sys.argv[2], "consolidated.*.pth")))
    assert len(one_files) == len(
        two_files
    ), "One directory has {} files while another has {} files.".format(
        len(one_files), len(two_files)
    )

    deltas = []
    for i in tqdm.trange(len(one_files), desc="Comparing shards"):
        one = torch.load(one_files[i])
        two = torch.load(two_files[i])
        assert len(one) == len(
            two
        ), "shard should have the same length: {} != {}".format(len(one), len(two))

        for _, (v, w) in enumerate(zip(one.items(), two.items())):
            assert v[0] == w[0], "{} != {}".format(v[0], w[0])
            assert v[1].shape == w[1].shape, "tensor {} shape {} != {}".format(
                v[0], v[1].shape, w[1].shape
            )

            delta = (v[1] - w[1]).abs().max().item()
            deltas.append((i, v[0], delta))
        del one
        del two
        gc.collect()

    deltas = sorted(deltas, key=lambda x: x[-1], reverse=True)
    print("Top 10 largest deltas:")
    for i, k, v in deltas[:10]:
        print(f"  shard {i} {k}: {v}")


if __name__ == "__main__":
    main()
