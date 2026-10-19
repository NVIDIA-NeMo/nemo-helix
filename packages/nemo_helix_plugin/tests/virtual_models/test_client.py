# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VirtualModelsClient transport tests via mocked httpx transport.

Scope is deliberately narrow. Generic client machinery (pagination, error
mapping, 204 handling, query params) is covered in ``tests/client``, and the
client-to-router wire contract is covered end to end against the real IGW app
in ``services/core/inference-gateway/tests/unit/test_virtual_models_router.py``.
What remains here is the sync client wiring, which no other test exercises.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
from nemo_helix_plugin.inference_middleware_models import VirtualModel
from nemo_helix_plugin.virtual_models.client import VirtualModelsClient
from nemo_helix_plugin.virtual_models.types import CreateVirtualModelRequest

BASE = "http://test:8000"
PATH = "/apis/inference-gateway/v2/workspaces/default/virtual-models"


def test_sync_create_returns_typed_virtual_model_and_sends_exact_request() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(
        201,
        request=httpx.Request("POST", BASE + PATH),
        json={
            "id": "vm-router",
            "name": "router",
            "workspace": "default",
            "default_model_entity": "default/llama",
            "created_at": "2026-01-01T00:00:00Z",
        },
    )
    client = VirtualModelsClient(base_url=BASE, workspace="default", http_client=http)

    result = client.create_virtual_model(
        body=CreateVirtualModelRequest(name="router", default_model_entity="default/llama")
    ).data()

    assert isinstance(result, VirtualModel)
    assert result.id == "vm-router"
    assert result.name == "router"
    args, kwargs = http.request.call_args
    assert args == ("POST", BASE + PATH)
    assert kwargs["content"] == b'{"default_model_entity":"default/llama","name":"router"}'
    assert kwargs["params"] is None
