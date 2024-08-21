# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import random
import pytest

import torch

from llama_recipes.data.sampler import LengthBasedBatchSampler
from llama_recipes.data.sampler import DistributedLengthBasedBatchSampler

SAMPLES = 33

@pytest.fixture
def dataset():
    random.seed(42)
    dataset = []
    def add_samples(ds, n, a, b):
        for _ in range(n):
            ds.append(random.randint(a,b) * [1,])
    add_samples(dataset, SAMPLES // 2, 1,9)
    add_samples(dataset, (SAMPLES // 2) + (SAMPLES % 2), 10,20)
    
    return random.sample(dataset, len(dataset))
    
    
@pytest.mark.parametrize("batch_size, drop_last", [(2, False), (8, False), (2, True), (8, True)])
def test_batch_sampler_array(dataset, batch_size, drop_last):
    
    sampler = LengthBasedBatchSampler(dataset, batch_size, drop_last)
    
    EXPECTED_LENGTH = SAMPLES // batch_size if drop_last else (SAMPLES // batch_size) + (SAMPLES % batch_size)
    
    all_ids = [i for b in sampler for i in b]
    assert len(set(all_ids)) == EXPECTED_LENGTH * batch_size if drop_last else len(dataset)
    
    assert len(sampler) == EXPECTED_LENGTH
    is_long = [len(d)>=10 for d in dataset]
    
    def check_batch(batch):
        return all(batch) or not any(batch)
    
    assert all(check_batch(is_long[i] for i in b) for b in sampler)
    
    
@pytest.mark.parametrize("batch_size, drop_last", [(2, False), (8, False), (2, True), (8, True)])
def test_batch_sampler_dict(dataset, batch_size, drop_last):
    
    dist_dataset = [{"input_ids": d, "attention_mask": d} for d in dataset]
    
    sampler = LengthBasedBatchSampler(dist_dataset, batch_size, drop_last)
    
    EXPECTED_LENGTH = SAMPLES // batch_size if drop_last else (SAMPLES // batch_size) + (SAMPLES % batch_size)
    
    assert len(sampler) == EXPECTED_LENGTH
    is_long = [len(d)>=10 for d in dataset]
    
    def check_batch(batch):
        return all(batch) or not any(batch)
    
    assert all(check_batch(is_long[i] for i in b) for b in sampler)
    
    
@pytest.mark.parametrize("batch_size", [2, 8])
def test_dist_batch_sampling(dataset, batch_size):
    sampler_1 = DistributedLengthBasedBatchSampler(
        dataset,
        batch_size=batch_size,
        rank=0,
        num_replicas=2,
        shuffle=False,
    )
    sampler_2 = DistributedLengthBasedBatchSampler(
        dataset,
        batch_size=batch_size,
        rank=1,
        num_replicas=2,
        shuffle=False,
    )
    
    ids_1 = set(i for b in sampler_1 for i in b)
    ids_2 = set(i for b in sampler_2 for i in b)
    
    assert ids_1.isdisjoint(ids_2)
    assert len(ids_1)+len(ids_2) > 0
    assert len(ids_1)+len(ids_2) == len(dataset) // batch_size  *  batch_size 