# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest

from transformers import LlamaTokenizer

@pytest.fixture(scope="module")
def llama_tokenizer():
    try:
        return LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    except OSError:
        return None


@pytest.fixture
def setup_tokenizer(llama_tokenizer):
    def _helper(tokenizer_mock):
        #Align with Llama 2 tokenizer
        tokenizer_mock.from_pretrained.return_value = llama_tokenizer

    return _helper

@pytest.fixture(autouse=True)
def skip_if_tokenizer_is_missing(request, llama_tokenizer):
    if request.node.get_closest_marker("skip_missing_tokenizer"):
        if llama_tokenizer is None:
            pytest.skip("Llama tokenizer could not be accessed. Did you log into huggingface hub and provided the correct token?")
