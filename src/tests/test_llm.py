# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

from llama_cookbook.inference.llm import MAX_TOKENS, MINIMAX


def test_minimax_image_input() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="A test image."))]
    )

    with patch("llama_cookbook.inference.llm.openai.OpenAI") as openai_client:
        openai_client.return_value.chat.completions.create.return_value = response

        provider = MINIMAX("MiniMax-M3", "test-key")

        assert (
            provider.query("Describe this image.", "https://example.com/image.png")
            == "A test image."
        )
        openai_client.assert_called_once_with(
            base_url="https://api.minimax.io/v1", api_key="test-key"
        )
        openai_client.return_value.chat.completions.create.assert_called_once_with(
            model="MiniMax-M3",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.png",
                                "detail": "default",
                            },
                        },
                    ],
                }
            ],
            max_tokens=MAX_TOKENS,
        )


def test_minimax_china_endpoint() -> None:
    with patch("llama_cookbook.inference.llm.openai.OpenAI") as openai_client:
        MINIMAX("MiniMax-M3", "test-key", region="china")

        openai_client.assert_called_once_with(
            base_url="https://api.minimaxi.com/v1", api_key="test-key"
        )
