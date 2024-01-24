# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest

from transformers import LlamaTokenizer

ACCESS_ERROR_MSG = "Could not access tokenizer at 'meta-llama/Llama-2-7b-hf'. Did you log into huggingface hub and provided the correct token?"

unskip_missing_tokenizer = False

@pytest.fixture(scope="module")
def llama_tokenizer():
    try:
        return LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    except OSError as e:
        if unskip_missing_tokenizer:
            raise e
    return None


@pytest.fixture
def setup_tokenizer(llama_tokenizer):
    def _helper(tokenizer_mock):
        #Align with Llama 2 tokenizer
        tokenizer_mock.from_pretrained.return_value = llama_tokenizer

    return _helper


@pytest.fixture(autouse=True)
def skip_if_tokenizer_is_missing(request, llama_tokenizer):
    if request.node.get_closest_marker("skip_missing_tokenizer") and not unskip_missing_tokenizer:
        if llama_tokenizer is None:
            pytest.skip(ACCESS_ERROR_MSG)


def pytest_addoption(parser):
    parser.addoption(
        "--unskip-missing-tokenizer",
        action="store_true",
        default=False, help="disable skip missing tokenizer")


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_preparse(config, args):
    if "--unskip-missing-tokenizer" not in args:
        return
    global unskip_missing_tokenizer
    unskip_missing_tokenizer = True
