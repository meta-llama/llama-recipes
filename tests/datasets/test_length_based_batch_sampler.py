import random
import pytest

import torch

from llama_recipes.datasets.utils import LengthBasedBatchSampler

SAMPLES = 33

@pytest.fixture
def dataset():
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