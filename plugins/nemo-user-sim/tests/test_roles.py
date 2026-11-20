# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import httpx
import openai
from nemo_user_sim_plugin.engine.roles import RoleAnswerer, RoleModels


async def test_judge_activation_uses_judge_model_and_maps_usage():
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "j",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "<rating>success</rating>"},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    client = openai.AsyncOpenAI(
        api_key="k", base_url="http://gw/v1", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    event = SimpleNamespace(
        activation_id="a1",
        role="judge",
        messages=[{"role": "user", "content": "rate"}],
        tools=None,
        tools_enabled=False,
        parameters={"max_tokens": 64, "stop": ["x"]},
    )

    payload = await RoleAnswerer(RoleModels(user="u", judge="j", summary="s"), client).answer(event)

    assert sent["model"] == "j" and sent["max_tokens"] == 64 and "stop" not in sent and "tools" not in sent
    assert payload == {
        "activation_id": "a1",
        "response": {"role": "assistant", "content": "<rating>success</rating>", "tool_calls": None},
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
