# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
from typing import List, Literal, TypedDict


Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def format_tokens(dialogs, tokenizer):
    if tokenizer.vocab_size >= 128000:
        return _format_tokens_llama3(dialogs, tokenizer)
    else:
        return _format_tokens_llama2(dialogs, tokenizer)


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
    tokens.extend(
        _encode_header({"role": "assistant", "content": ""}, tokenizer)
        )
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
                ) + [tokenizer.eos_token_id]
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


def read_dialogs_from_file(file_path):
    with open(file_path, 'r') as file:
        dialogs = json.load(file)
    return dialogs
