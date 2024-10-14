# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest

from transformers import AutoTokenizer

ACCESS_ERROR_MSG = "Could not access tokenizer at 'meta-llama/Llama-2-7b-hf'. Did you log into huggingface hub and provided the correct token?"

try:
    AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    LLAMA_VERSIONS = ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
except OSError:
    LLAMA_VERSIONS = ["fake_llama"]

@pytest.fixture(params=LLAMA_VERSIONS)
def llama_version(request):
    return request.param


@pytest.fixture(params=["mllama", "llama"])
def model_type(request):
    return request.param

class FakeTokenier(object):
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id = 3

        self.pad_token = "<|pad_id|>"
        self.bos_token = "<|bos_id|>"
        self.eos_token = "<|eos_id|>"
        self.sep_token = "<|sep_id|>"

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode(self, text, *args, **kwargs):
        breakpoint()
        return [len(c) for c in text.split(" ")]
    
    def __len__(self):
        return 128256


@pytest.fixture(scope="module")
def llama_tokenizer(request):
    if LLAMA_VERSIONS == ["fake_llama"]:
        return {"fake_llama": FakeTokenier()}
    else:
        return {k: AutoTokenizer.from_pretrained(k) for k in LLAMA_VERSIONS}


@pytest.fixture
def setup_tokenizer(llama_tokenizer, llama_version):
    def _helper(tokenizer_mock):
        #Align with Llama 2 tokenizer
        tokenizer_mock.from_pretrained.return_value = llama_tokenizer[llama_version]

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
        AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer_available = True
    except OSError:
        tokenizer_available = False

    skip_missing_tokenizer = pytest.mark.skip(reason=ACCESS_ERROR_MSG)
    for item in items:
        if "skip_missing_tokenizer" in item.keywords and not tokenizer_available:
            item.add_marker(skip_missing_tokenizer)
