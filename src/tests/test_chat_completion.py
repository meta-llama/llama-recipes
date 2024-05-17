import sys
from pathlib import Path
from typing import List, Literal, TypedDict
from unittest.mock import patch

import pytest
import torch
from llama_recipes.inference.chat_utils import read_dialogs_from_file

ROOT_DIR = Path(__file__).parents[2]
CHAT_COMPLETION_DIR = ROOT_DIR / "recipes/inference/local_inference/chat_completion/"

sys.path = [CHAT_COMPLETION_DIR.as_posix()] + sys.path

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def _encode_header(message, tokenizer):
    tokens = []
    tokens.extend(tokenizer.encode("<|start_header_id|>"))
    tokens.extend(tokenizer.encode(message["role"]))
    tokens.extend(tokenizer.encode("<|end_header_id|>"))
    tokens.extend(tokenizer.encode("\n\n"))
    return tokens


def _encode_message(message, tokenizer):
    tokens = _encode_header(message, tokenizer)
    tokens.extend(tokenizer.encode(message["content"].strip()))
    tokens.extend(tokenizer.encode("<|eot_id|>"))
    return tokens


def _format_dialog(dialog, tokenizer):
    tokens = []
    tokens.extend(tokenizer.encode("<|begin_of_text|>"))
    for msg in dialog:
        tokens.extend(_encode_message(msg, tokenizer))
    tokens.extend(_encode_header({"role": "assistant", "content": ""}, tokenizer))
    return tokens


def _format_tokens_llama3(dialogs, tokenizer):
    return [_format_dialog(dialog, tokenizer) for dialog in dialogs]


def _format_tokens_llama2(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                + [tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens


@pytest.mark.skip_missing_tokenizer
@patch("chat_completion.AutoTokenizer")
@patch("chat_completion.load_model")
def test_chat_completion(
    load_model, tokenizer, setup_tokenizer, llama_tokenizer, llama_version
):
    from chat_completion import main

    setup_tokenizer(tokenizer)
    load_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]

    kwargs = {
        "prompt_file": (CHAT_COMPLETION_DIR / "chats.json").as_posix(),
    }

    main(llama_version, **kwargs)

    dialogs = read_dialogs_from_file(kwargs["prompt_file"])
    format_tokens = (
        _format_tokens_llama2
        if llama_version == "meta-llama/Llama-2-7b-hf"
        else _format_tokens_llama3
    )

    REF_RESULT = format_tokens(dialogs, llama_tokenizer[llama_version])

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
