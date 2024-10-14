import sys
from pathlib import Path
from typing import List, TypedDict
from unittest.mock import patch

import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file

ROOT_DIR = Path(__file__).parents[2]
CHAT_COMPLETION_DIR = ROOT_DIR / "recipes/quickstart/inference/local_inference/chat_completion/"

sys.path = [CHAT_COMPLETION_DIR.as_posix()] + sys.path

default_system_prompt = [{"role": "system", "content": "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"}]

def _encode_header(message, tokenizer):
    tokens = []
    tokens.extend(tokenizer.encode("<|start_header_id|>", add_special_tokens=False))
    tokens.extend(tokenizer.encode(message["role"], add_special_tokens=False))
    tokens.extend(tokenizer.encode("<|end_header_id|>", add_special_tokens=False))
    tokens.extend(tokenizer.encode("\n\n", add_special_tokens=False))
    return tokens


def _encode_message(message, tokenizer):
    tokens = _encode_header(message, tokenizer)
    tokens.extend(tokenizer.encode(message["content"], add_special_tokens=False))
    tokens.extend(tokenizer.encode("<|eot_id|>", add_special_tokens=False))
    return tokens


def _format_dialog(dialog, tokenizer):
    tokens = []
    tokens.extend(tokenizer.encode("<|begin_of_text|>", add_special_tokens=False))
    if dialog[0]["role"] == "system":
        dialog[0]["content"] = default_system_prompt[0]["content"] + dialog[0]["content"]
    else:
        dialog = default_system_prompt + dialog
    for msg in dialog:
        tokens.extend(_encode_message(msg, tokenizer))
    return tokens


def _format_tokens_llama3(dialogs, tokenizer):
    return [_format_dialog(dialog, tokenizer) for dialog in dialogs]


@pytest.mark.skip_missing_tokenizer
@patch("chat_completion.AutoTokenizer")
@patch("chat_completion.load_model")
def test_chat_completion(
    load_model, tokenizer, setup_tokenizer, llama_tokenizer, llama_version
):
    if "Llama-2" in llama_version or llama_version == "fake_llama":
        pytest.skip(f"skipping test for {llama_version}")

    from chat_completion import main

    setup_tokenizer(tokenizer)
    load_model.return_value.get_input_embeddings.return_value.weight.shape = [128256]

    kwargs = {
        "prompt_file": (CHAT_COMPLETION_DIR / "chats.json").as_posix(),
    }

    main(llama_version, **kwargs)

    dialogs = read_dialogs_from_file(kwargs["prompt_file"])

    REF_RESULT = _format_tokens_llama3(dialogs, llama_tokenizer[llama_version])

    assert all(
        (
            load_model.return_value.generate.mock_calls[0 * 4][2]["input_ids"].cpu()
            == torch.tensor(REF_RESULT[0]).long()
        ).tolist()
    )
    assert all(
        (
            load_model.return_value.generate.mock_calls[1 * 4][2]["input_ids"].cpu()
            == torch.tensor(REF_RESULT[1]).long()
        ).tolist()
    )
    assert all(
        (
            load_model.return_value.generate.mock_calls[2 * 4][2]["input_ids"].cpu()
            == torch.tensor(REF_RESULT[2]).long()
        ).tolist()
    )
    assert all(
        (
            load_model.return_value.generate.mock_calls[3 * 4][2]["input_ids"].cpu()
            == torch.tensor(REF_RESULT[3]).long()
        ).tolist()
    )
    assert all(
        (
            load_model.return_value.generate.mock_calls[4 * 4][2]["input_ids"].cpu()
            == torch.tensor(REF_RESULT[4]).long()
        ).tolist()
    )
