# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The closed-book middleware loaded by the Inference Gateway from its entry point, on a closed-book VirtualModel."""

import pytest
from nemo_evals.gateway.middleware.closed_book import CONFIG_TYPE, MIDDLEWARE_NAME
from nhx.core.inference_gateway.testing.fixtures import _igw_app_context, _igw_extra_services, igw_plugin_harness
from nhx.core.inference_gateway.testing.harness import IGWPluginHarness
from nhx.testing.mock_chat_completions import ChatCompletion, chat_completion

__all__ = ["_igw_app_context", "_igw_extra_services", "igw_plugin_harness"]

pytestmark = [pytest.mark.integration]

_CLOSED_BOOK = [{"name": MIDDLEWARE_NAME, "config_type": CONFIG_TYPE, "config": {}}]


def test_hosted_tools_never_reach_the_provider(igw_plugin_harness: IGWPluginHarness) -> None:
    h = igw_plugin_harness
    h.mock_chat_completions("luna", responses=[ChatCompletion(body=chat_completion(content="Paris"))])
    h.add_provider(workspace="default", served_models={"luna": "luna"})
    function_tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}

    with h.load_plugin(MIDDLEWARE_NAME):
        h.add_virtual_model(
            workspace="default", name="cb-luna", default_model_entity="default/luna", request_middleware=_CLOSED_BOOK
        )
        response = h.chat_completions(
            workspace="default",
            body={
                "model": "cb-luna",
                "messages": [{"role": "user", "content": "Capital of France?"}],
                "tools": [function_tool, {"type": "web_search"}],
                "web_search_options": {},
            },
        )

    assert response["choices"][0]["message"]["content"] == "Paris"
    h.assert_request_body_for(
        "luna", lambda body: body["tools"] == [function_tool] and "web_search_options" not in body
    )


def test_self_searching_model_is_refused_with_403(igw_plugin_harness: IGWPluginHarness) -> None:
    h = igw_plugin_harness
    h.add_provider(workspace="default", served_models={"sonar-pro": "sonar-pro"})

    with h.load_plugin(MIDDLEWARE_NAME):
        h.add_virtual_model(
            workspace="default",
            name="cb-sonar",
            default_model_entity="default/sonar-pro",
            request_middleware=_CLOSED_BOOK,
        )
        response = h.test_client.post(
            "/apis/inference-gateway/v2/workspaces/default/openai/-/v1/chat/completions",
            json={"model": "cb-sonar", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 403
    h.assert_no_calls_to("sonar-pro")
