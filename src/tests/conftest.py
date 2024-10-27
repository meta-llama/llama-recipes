# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest

from utils import maybe_tokenizer

ACCESS_ERROR_MSG = "Could not access tokenizer. Did you log into huggingface hub and provided the correct token?"

LLAMA_VERSIONS = ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3.1-8B-Instruct", "fake_llama"]

LLAMA_TOKENIZERS = {k: maybe_tokenizer(k) for k in LLAMA_VERSIONS}

@pytest.fixture(params=LLAMA_VERSIONS)
def llama_version(request):
    return request.param


@pytest.fixture(params=["mllama", "llama"])
def model_type(request):
    return request.param


@pytest.fixture(scope="module")
def llama_tokenizer(request):
    return LLAMA_TOKENIZERS


@pytest.fixture
def setup_tokenizer(llama_tokenizer, llama_version):
    def _helper(tokenizer_mock):
        #Align with Llama 2 tokenizer
        tokenizer_mock.from_pretrained.return_value = llama_tokenizer[llama_version]

    return _helper

@pytest.fixture
def setup_processor(llama_tokenizer, llama_version):
    def _helper(processor_mock):
        processor_mock.from_pretrained.return_value.tokenizer = llama_tokenizer[llama_version]

    return _helper


def pytest_addoption(parser):
    parser.addoption(
        "--unskip-missing-tokenizer",
        action="store_true",
        default=False, help="disable skip missing tokenizer")

def pytest_configure(config):
    config.addinivalue_line("markers", "skip_missing_tokenizer: skip if tokenizer is unavailable")


def pytest_collection_modifyitems(config, items):
    #skip tests marked with skip_missing_tokenizer if tokenizer is unavailable unless --unskip-missing-tokenizer is passed
    if config.getoption("--unskip-missing-tokenizer"):
        return

    skip_missing_tokenizer = pytest.mark.skip(reason=ACCESS_ERROR_MSG)
    for item in items:
        # get the tokenizer for the test
        version = [v for v in LLAMA_VERSIONS for i in item.keywords if v in i]
        if len(version) == 0:
            # no tokenizer used in this test
            continue
        version = version.pop()
        assert version in LLAMA_TOKENIZERS
        if "skip_missing_tokenizer" in item.keywords and LLAMA_TOKENIZERS[version] is None:
            item.add_marker(skip_missing_tokenizer)
