# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from transformers import AutoTokenizer


class FakeTokenizer(object):
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 42
        self.eos_token_id = 43
        self.sep_token_id = 3
        self.vocab_size = 128256

        self.pad_token = "<|pad_id|>"
        self.bos_token = "<|bos_id|>"
        self.eos_token = "<|eos_id|>"
        self.sep_token = "<|sep_id|>"
        self.tokenizer = self
        self.padding_side = "left"

    def __call__(self, *args, **kwargs):
        ids = self.encode(*args, **kwargs)
        return {"input_ids": ids}

    def encode(self, text, *args, **kwargs):
        return [self.bos_token_id] + [len(c) for c in text.split(" ")] + [self.eos_token_id]
    
    def __len__(self):
        return 128256
    
    def pad(self, *args, **kwargs):
        args = args[0]
        max_len = max([len(a["input_ids"]) for a in args])
        for a in args:
            for k in a.keys():
                a[k] = a[k] + ([self.pad_token_id if k == "input_ids" else 0] * (max_len - len(a)))
        out = {}
        for k in args[0].keys():
            out[k] = [a[k] for a in args]
        return out


def maybe_tokenizer(name):
    if name == "fake_llama":
        return FakeTokenizer()
    try:
        return AutoTokenizer.from_pretrained(name)
    except OSError:
        return None
