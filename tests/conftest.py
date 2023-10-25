# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest

from transformers import LlamaTokenizer


@pytest.fixture
def setup_tokenizer():
    def _helper(tokenizer):
        #Align with Llama 2 tokenizer
        tokenizer.from_pretrained.return_value = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        tokenizer.from_pretrained.return_value.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
        tokenizer.from_pretrained.return_value.bos_token_id = 1
        tokenizer.from_pretrained.return_value.eos_token_id = 2

    return _helper
