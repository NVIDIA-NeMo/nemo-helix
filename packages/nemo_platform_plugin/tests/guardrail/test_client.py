# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GuardrailClient via mocked httpx transport."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.guardrail.client import AsyncGuardrailClient, GuardrailClient
from nemo_platform_plugin.guardrail.types import (
    CreateGuardrailConfigRequest,
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    GuardrailConfig,
    RailsConfig,
)

BASE = "http://test:8000"


def _config_json(name: str = "safe", **extra: object) -> dict:
    base = {
        "id": f"guardrail-{name}",
        "name": name,
        "workspace": "default",
        "description": "Content safety",
        "data": {
            "models": [{"type": "content_safety", "engine": "nim", "model": "default/safety"}],
            "rails": {"output": {"flows": ["self check output"], "streaming": {"enabled": False}}},
        },
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }
    base.update(extra)
    return base


def test_create_guardrail_config_round_trip() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(201, request=httpx.Request("POST", BASE), json=_config_json())
    client = GuardrailClient(base_url=BASE, workspace="default", http_client=http)

    out = client.create_guardrail_config(
        body=CreateGuardrailConfigRequest(
            name="safe",
            data=RailsConfig.model_validate({"rails": {"input": {"flows": ["flow"]}}}),
        )
    ).data()

    assert isinstance(out, GuardrailConfig)
    assert out.name == "safe"
    assert out.data is not None
    assert out.data.rails is not None
    assert out.data.rails.output is not None
    assert out.data.rails.output.streaming is not None
    assert out.data.rails.output.streaming.enabled is False
    args, kwargs = http.request.call_args
    assert args == ("POST", f"{BASE}/apis/guardrails/v2/workspaces/default/configs")
    assert kwargs["content"] == b'{"name":"safe","data":{"rails":{"input":{"flows":["flow"]}}}}'


def test_list_guardrail_configs_paginates() -> None:
    http = MagicMock(spec=httpx.Client)
    page1 = {
        "data": [_config_json("a")],
        "pagination": {
            "page": 1,
            "page_size": 1,
            "current_page_size": 1,
            "total_pages": 2,
            "total_results": 2,
        },
    }
    page2 = {
        "data": [_config_json("b")],
        "pagination": {
            "page": 2,
            "page_size": 1,
            "current_page_size": 1,
            "total_pages": 2,
            "total_results": 2,
        },
    }
    http.request.side_effect = [
        httpx.Response(200, request=httpx.Request("GET", BASE), json=page1),
        httpx.Response(200, request=httpx.Request("GET", BASE), json=page2),
    ]
    client = GuardrailClient(base_url=BASE, workspace="default", http_client=http)

    assert [config.name for config in client.list_guardrail_configs().items()] == ["a", "b"]


def test_get_guardrail_config_not_found_raises() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(404, request=httpx.Request("GET", BASE), json={"detail": "not found"})
    client = GuardrailClient(base_url=BASE, workspace="default", http_client=http)

    with pytest.raises(NotFoundError) as exc:
        client.get_guardrail_config(name="missing")
    assert exc.value.status_code == 404


def test_check_guardrail_round_trip() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(
        200,
        request=httpx.Request("POST", BASE),
        json={
            "status": "blocked",
            "rails_status": {"self check input": {"status": "blocked"}},
            "guardrails_data": {
                "config_ids": ["default/safe"],
                "log": {"activated_rails": [{"name": "self check input", "type": "input", "stop": True}]},
            },
        },
    )
    client = GuardrailClient(base_url=BASE, workspace="default", http_client=http)

    out = client.check_guardrail(
        body=GuardrailCheckRequest(
            model="default/app",
            messages=[{"role": "user", "content": "hello"}],
            guardrails={"config_id": "default/safe"},
        )
    ).data()

    assert isinstance(out, GuardrailCheckResponse)
    assert out.status == "blocked"
    assert out.guardrails_data is not None
    assert out.guardrails_data.config_ids == ["default/safe"]
    assert out.guardrails_data.log is not None
    assert out.guardrails_data.log.activated_rails is not None
    assert out.guardrails_data.log.activated_rails[0].stop is True


@pytest.mark.asyncio
async def test_async_get_guardrail_config_round_trip() -> None:
    http = AsyncMock(spec=httpx.AsyncClient)
    http.request.return_value = httpx.Response(200, request=httpx.Request("GET", BASE), json=_config_json())
    client = AsyncGuardrailClient(base_url=BASE, workspace="default", http_client=http)

    out = (await client.get_guardrail_config(name="safe")).data()

    assert isinstance(out, GuardrailConfig)
    assert out.data is not None
    assert out.data.rails is not None
    http.request.assert_awaited_once()
