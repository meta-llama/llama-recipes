# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from itertools import product
from unittest.mock import patch

import anthropic
import httpx
import openai

from llama_cookbook.inference.llm import MAX_TOKENS, MINIMAX

OPENAI_CLIENT = openai.OpenAI
ANTHROPIC_CLIENT = anthropic.Anthropic


class MiniMaxTest(unittest.TestCase):
    def test_supported_models_and_endpoint_paths(self) -> None:
        endpoints = [
            ("openai", None, "https://api.minimax.io/v1/chat/completions"),
            (
                "openai",
                "https://api.minimaxi.com/v1",
                "https://api.minimaxi.com/v1/chat/completions",
            ),
            (
                "anthropic",
                None,
                "https://api.minimax.io/anthropic/v1/messages",
            ),
            (
                "anthropic",
                "https://api.minimaxi.com/anthropic",
                "https://api.minimaxi.com/anthropic/v1/messages",
            ),
            (
                "openai",
                "https://example.test/custom/v1",
                "https://example.test/custom/v1/chat/completions",
            ),
            (
                "anthropic",
                "https://example.test/custom/anthropic",
                "https://example.test/custom/anthropic/v1/messages",
            ),
        ]
        models = [
            "MiniMax-M3.1-Flash",
            "MiniMax-M3",
            "MiniMax-M2.7",
            "saved-custom-model",
        ]

        for (api_format, base_url, expected_url), model in product(endpoints, models):
            with self.subTest(api_format=api_format, base_url=base_url, model=model):
                requests = []
                transport_httpx = httpx
                if api_format == "anthropic" and not issubclass(
                    anthropic.DefaultHttpxClient, httpx.Client
                ):
                    import httpx2 as transport_httpx

                def handler(request):
                    requests.append(request)
                    if api_format == "openai":
                        body = {
                            "id": "chatcmpl-test",
                            "object": "chat.completion",
                            "created": 0,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Test response",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    else:
                        body = {
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Test response"}],
                            "model": model,
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    return transport_httpx.Response(200, json=body, request=request)

                with transport_httpx.Client(
                    transport=transport_httpx.MockTransport(handler)
                ) as http_client:
                    # Keep the real SDK serialization and the adapter's endpoint
                    # arguments; substitute only the offline HTTP transport.
                    client_class = (
                        OPENAI_CLIENT if api_format == "openai" else ANTHROPIC_CLIENT
                    )

                    def make_client(**kwargs):
                        return client_class(http_client=http_client, **kwargs)

                    kwargs = {}
                    if api_format != "openai":
                        kwargs["api_format"] = api_format
                    if base_url is not None:
                        kwargs["base_url"] = base_url
                    patch_target = (
                        "llama_cookbook.inference.llm.openai.OpenAI"
                        if api_format == "openai"
                        else "llama_cookbook.inference.llm.anthropic.Anthropic"
                    )
                    with patch(patch_target, side_effect=make_client):
                        provider = MINIMAX(model, "test-key", **kwargs)
                        self.assertEqual(provider.query("Test prompt"), "Test response")
                        self.assertEqual(
                            provider.valid_models(),
                            ["MiniMax-M3.1-Flash", "MiniMax-M3"],
                        )

                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0].method, "POST")
                self.assertEqual(str(requests[0].url), expected_url)
                # Exact payload equality also guards against adding unsupported
                # thinking, reasoning effort or service tier defaults.
                self.assertEqual(
                    json.loads(requests[0].content),
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": "Test prompt"}],
                        "max_tokens": MAX_TOKENS,
                    },
                )


if __name__ == "__main__":
    unittest.main()
