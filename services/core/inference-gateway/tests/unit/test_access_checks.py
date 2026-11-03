# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every installed plugin's ``check_access`` runs on every proxied call, whichever VirtualModels list the plugin."""

import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.inference_middleware import (
    InferenceAccessRequest,
    InferenceMiddlewareError,
    NemoInferenceMiddleware,
)
from nhx.core.inference_gateway.api.middleware_registry import MiddlewareRegistry

CHAT = {"model": "e2e-test/meta_llama-3.2-1b-instruct", "messages": [{"role": "user", "content": "hi"}]}


class _Gatekeeper(NemoInferenceMiddleware):
    def __init__(self, refuse: bool = False, crash: bool = False) -> None:
        super().__init__()
        self.refuse = refuse
        self.crash = crash
        self.seen: list[InferenceAccessRequest] = []

    async def check_access(self, request: InferenceAccessRequest) -> None:
        self.seen.append(request)
        if self.crash:
            raise RuntimeError("boom")
        if self.refuse:
            raise InferenceMiddlewareError("not for you", status_code=403)


@pytest.fixture
def gatekeeper() -> _Gatekeeper:
    return _Gatekeeper()


class _Bystander(NemoInferenceMiddleware):
    """A plugin that leaves ``check_access`` alone."""


@pytest.fixture
def middleware_registry(gatekeeper: _Gatekeeper) -> MiddlewareRegistry:
    return MiddlewareRegistry(plugins={"bystander": _Bystander(), "gatekeeper": gatekeeper})


@pytest.mark.parametrize(
    ("method", "url", "virtual_model_id", "provider_name", "path"),
    [
        pytest.param(
            "post",
            "/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/chat/completions",
            "e2e-test/meta_llama-3.2-1b-instruct",
            None,
            "v1/chat/completions",
            id="model route",
        ),
        pytest.param(
            "post",
            "/v2/workspaces/e2e-test/openai/-/v1/chat/completions",
            "e2e-test/meta_llama-3.2-1b-instruct",
            None,
            "v1/chat/completions",
            id="openai route",
        ),
        pytest.param(
            "get",
            "/v2/workspaces/default/provider/ollama/-/v1/models",
            None,
            "ollama",
            "v1/models",
            id="provider route",
        ),
    ],
)
def test_each_proxy_route_asks_installed_plugins_before_serving_a_call(
    client: TestClient,
    gatekeeper: _Gatekeeper,
    mock_proxy_client,
    method: str,
    url: str,
    virtual_model_id: str | None,
    provider_name: str | None,
    path: str,
) -> None:
    headers = {"authorization": "Bearer caller-token", "x-request-id": "req-1"}
    response = client.request(method, url, headers=headers, json=CHAT if method == "post" else None)

    assert response.status_code == 200
    [seen] = gatekeeper.seen
    assert (seen.request_id, seen.virtual_model_id, seen.provider_name, seen.path) == (
        "req-1",
        virtual_model_id,
        provider_name,
        path,
    )
    assert seen.headers["authorization"] == "Bearer caller-token"
    assert seen.requested_model == (CHAT["model"] if virtual_model_id is not None else None)
    mock_proxy_client.request.assert_called_once()


@pytest.mark.parametrize(
    "url",
    [
        "/v2/workspaces/e2e-test/model/meta_llama-3.2-1b-instruct/-/v1/chat/completions",
        "/v2/workspaces/default/provider/ollama/-/v1/chat/completions",
    ],
)
def test_a_refusal_stops_the_call_before_it_reaches_the_provider(
    client: TestClient, gatekeeper: _Gatekeeper, mock_proxy_client, url: str
) -> None:
    gatekeeper.refuse = True

    response = client.post(url, json=CHAT)

    assert (response.status_code, response.json()["detail"]) == (403, "not for you")
    mock_proxy_client.request.assert_not_called()


def test_a_check_that_crashes_refuses_the_call(client: TestClient, gatekeeper: _Gatekeeper, mock_proxy_client) -> None:
    """A broken check must never let a call through unchecked."""
    gatekeeper.crash = True

    response = client.post("/v2/workspaces/e2e-test/openai/-/v1/chat/completions", json=CHAT)

    assert response.status_code == 500
    mock_proxy_client.request.assert_not_called()


def test_a_call_naming_a_lora_adapter_shows_the_adapter_to_the_check(
    client: TestClient, gatekeeper: _Gatekeeper
) -> None:
    """IGW serves the adapter through its base model's VirtualModel, so the check must see which model was asked for."""
    adapter = "meta_llama-3.2-1b-instruct&adapters/tuning/my-lora"

    client.post(f"/v2/workspaces/e2e-test/model/{adapter}/-/v1/chat/completions", json=CHAT)

    [seen] = gatekeeper.seen
    assert (seen.virtual_model_id, seen.requested_model) == (
        "e2e-test/meta_llama-3.2-1b-instruct",
        adapter,
    )
