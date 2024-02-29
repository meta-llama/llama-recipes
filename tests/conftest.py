# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest

from transformers import LlamaTokenizer

ACCESS_ERROR_MSG = "Could not access tokenizer at 'meta-llama/Llama-2-7b-hf'. Did you log into huggingface hub and provided the correct token?"


@pytest.fixture(scope="module")
def llama_tokenizer():
    return LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


@pytest.fixture
def setup_tokenizer(llama_tokenizer):
    def _helper(tokenizer_mock):
        #Align with Llama 2 tokenizer
        tokenizer_mock.from_pretrained.return_value = llama_tokenizer

    return _helper


def pytest_addoption(parser):
    parser.addoption(
        "--unskip-missing-tokenizer",
        action="store_true",
        default=False, help="disable skip missing tokenizer")
    
def pytest_configure(config):
    config.addinivalue_line("markers", "skip_missing_tokenizer: skip if tokenizer is unavailable")

    
def pytest_collection_modifyitems(config, items):
    if config.getoption("--unskip-missing-tokenizer"):
        return
    
    try:
        LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer_available = True
    except OSError:
        tokenizer_available = False
    
    skip_missing_tokenizer = pytest.mark.skip(reason=ACCESS_ERROR_MSG)
    for item in items:
        if "skip_missing_tokenizer" in item.keywords and not tokenizer_available:
            item.add_marker(skip_missing_tokenizer)
