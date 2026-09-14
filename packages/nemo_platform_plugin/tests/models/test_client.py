# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelsClient / AsyncModelsClient via mocked httpx transport."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_platform_plugin.client.errors import ConflictError, NotFoundError
from nemo_platform_plugin.models.client import AsyncModelsClient, ModelsClient
from nemo_platform_plugin.models.types import (
    CreateModelDeploymentRequest,
    CreateModelEntityRequest,
    CreateModelProviderRequest,
    ModelDeployment,
    ModelEntity,
    ModelProvider,
)

BASE = "http://test:8000"


def _model_json(name: str = "llama", workspace: str = "default", **extra: object) -> dict:
    base = {
        "id": f"model-{name}",
        "name": name,
        "workspace": workspace,
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }
    base.update(extra)
    return base


def _provider_json(status: str = "PENDING", name: str = "my-provider", **extra: object) -> dict:
    base = {
        "id": f"provider-{name}",
        "name": name,
        "workspace": "default",
        "host_url": "https://api.example.com",
        "status": status,
        "status_message": "",
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }
    base.update(extra)
    return base


def _deployment_json(status: str = "PENDING", history: list | None = None, **extra: object) -> dict:
    base = {
        "id": "dep-1",
        "name": "my-deploy",
        "workspace": "default",
        "entity_version": 1,
        "config": "cfg",
        "config_version": 1,
        "status": status,
        "status_message": "",
        "status_history": history if history is not None else [],
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------


def test_create_model_round_trip() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(201, request=httpx.Request("POST", BASE), json=_model_json())
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    out = client.create_model(body=CreateModelEntityRequest(name="llama")).data()

    assert isinstance(out, ModelEntity)
    assert out.name == "llama"
    args, kwargs = http.request.call_args
    assert args == ("POST", f"{BASE}/apis/models/v2/workspaces/default/models")
    assert kwargs["content"] == b'{"name":"llama"}'


def test_list_models_paginates_across_pages() -> None:
    http = MagicMock(spec=httpx.Client)
    page1 = {
        "data": [_model_json("a"), _model_json("b")],
        "pagination": {
            "page": 1,
            "page_size": 2,
            "current_page_size": 2,
            "total_pages": 2,
            "total_results": 3,
        },
    }
    page2 = {
        "data": [_model_json("c")],
        "pagination": {
            "page": 2,
            "page_size": 2,
            "current_page_size": 1,
            "total_pages": 2,
            "total_results": 3,
        },
    }
    http.request.side_effect = [
        httpx.Response(200, request=httpx.Request("GET", BASE), json=page1),
        httpx.Response(200, request=httpx.Request("GET", BASE), json=page2),
    ]
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    names = [m.name for m in client.list_models().items()]
    assert names == ["a", "b", "c"]


def test_get_model_not_found_raises() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(404, request=httpx.Request("GET", BASE), json={"detail": "not found"})
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    with pytest.raises(NotFoundError) as exc:
        client.get_model(name="missing")
    assert exc.value.status_code == 404


def test_delete_deployment_returns_none_on_202() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(202, request=httpx.Request("DELETE", BASE))
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    resp = client.delete_deployment(name="d")
    assert resp.data() is None
    assert resp.http_response.status_code == 202


def test_delete_deployment_returns_none_on_204() -> None:
    """Synchronous hard-delete: 204 No Content is success with a ``None`` body."""
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(204, request=httpx.Request("DELETE", BASE))
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    resp = client.delete_deployment(name="d")
    assert resp.data() is None
    # The 202/204 distinction is only observable via the raw status code.
    assert resp.http_response.status_code == 204


def test_delete_deployment_version_accepts_202_and_204() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.side_effect = [
        httpx.Response(202, request=httpx.Request("DELETE", BASE)),
        httpx.Response(204, request=httpx.Request("DELETE", BASE)),
    ]
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.delete_deployment_version(deployment="d", name="1").data() is None
    assert client.delete_deployment_version(deployment="d", name="2").data() is None


def test_create_deployment_conflict_without_exist_ok_raises() -> None:
    """Default exist_ok=False: a 409 surfaces as ConflictError (no GET replay)."""
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(409, request=httpx.Request("POST", BASE), json={"detail": "exists"})
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    with pytest.raises(ConflictError) as exc:
        client.create_deployment(body=CreateModelDeploymentRequest(name="d", config="cfg"))
    assert exc.value.status_code == 409
    # A single POST was made; no conflict-resolving GET replay happened.
    assert http.request.call_count == 1


def test_create_provider_exist_ok_resolves_conflict() -> None:
    """exist_ok=True: a 409 replays the linked GET and returns the existing entity."""
    http = MagicMock(spec=httpx.Client)
    conflict = httpx.Response(409, request=httpx.Request("POST", BASE), json={"detail": "exists"})
    existing = httpx.Response(
        200,
        request=httpx.Request("GET", BASE),
        json=_model_json("p", host_url="http://x") | {"host_url": "http://x"},
    )
    http.request.side_effect = [conflict, existing]
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    out = client.create_provider(body=CreateModelProviderRequest(name="p", host_url="http://x"), exist_ok=True).data()

    assert isinstance(out, ModelProvider)
    assert out.name == "p"
    # Second call is the GET replay for the existing provider.
    assert http.request.call_args_list[1].args[0] == "GET"
    assert http.request.call_args_list[1].args[1].endswith("/providers/p")


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def test_openai_route_base_url() -> None:
    client = ModelsClient(base_url=BASE + "/", workspace="default")
    assert client.get_openai_route_base_url() == f"{BASE}/apis/inference-gateway/v2/workspaces/default/openai/-/v1"
    assert (
        client.get_openai_route_base_url(workspace="other")
        == f"{BASE}/apis/inference-gateway/v2/workspaces/other/openai/-/v1"
    )


def test_openai_route_base_url_missing_workspace_raises() -> None:
    client = ModelsClient(base_url=BASE)
    with pytest.raises(ValueError, match="Missing workspace"):
        client.get_openai_route_base_url()


def test_provider_route_appends_v1_conditionally() -> None:
    client = ModelsClient(base_url=BASE, workspace="default")
    p_openai = ModelProvider.model_validate(_model_json("p", host_url="https://api.openai.com"))
    p_nim = ModelProvider.model_validate(_model_json("p", host_url="https://nim.example.com/v1"))

    assert client.get_provider_route_openai_url(p_openai).endswith("/provider/p/-/v1")
    assert client.get_provider_route_openai_url(p_nim).endswith("/provider/p/-")


def test_model_entity_route_always_v1() -> None:
    client = ModelsClient(base_url=BASE, workspace="default")
    me = ModelEntity.model_validate(_model_json("m"))
    assert client.get_model_entity_route_openai_url(me).endswith("/model/m/-/v1")


def test_resolve_model_reference_fetches_model_and_provider() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.side_effect = [
        httpx.Response(
            200,
            request=httpx.Request("GET", BASE),
            json=_model_json("judge", model_providers=["default/provider"]),
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", BASE),
            json=_provider_json(
                "READY",
                name="provider",
                host_url="http://nim.example.test:8000",
                served_models=[
                    {
                        "model_entity_id": "default/judge",
                        "served_model_name": "publisher/upstream-judge",
                    }
                ],
            ),
        ),
    ]
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    result = client.resolve_model_reference("default/judge")

    assert [call.args[1] for call in http.request.call_args_list] == [
        f"{BASE}/apis/models/v2/workspaces/default/models/judge",
        f"{BASE}/apis/models/v2/workspaces/default/providers/provider",
    ]
    assert result.name == "judge"
    assert result.url == f"{BASE}/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1"
    assert result.host_url == "http://nim.example.test:8000"
    assert result.served_model_name == "publisher/upstream-judge"


def test_resolve_model_reference_ignores_missing_provider() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.side_effect = [
        httpx.Response(
            200,
            request=httpx.Request("GET", BASE),
            json=_model_json("judge", model_providers=["default/missing-provider"]),
        ),
        httpx.Response(404, request=httpx.Request("GET", BASE), json={"detail": "missing"}),
    ]
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    result = client.resolve_model_reference("default/judge")

    assert result.name == "judge"
    assert result.host_url is None


def test_provider_route_for_deployment_fetches_provider() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", BASE),
        json=_model_json("my-provider", host_url="https://api.example.com"),
    )
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)
    deployment = ModelDeployment.model_validate(_deployment_json(model_provider_id="default/my-provider"))

    url = client.get_provider_route_openai_url_for_deployment(deployment)
    assert url.endswith("/workspaces/default/provider/my-provider/-/v1")
    assert http.request.call_args.args[1].endswith("/providers/my-provider")


def test_provider_route_for_deployment_without_provider_id_raises() -> None:
    client = ModelsClient(base_url=BASE, workspace="default")
    deployment = ModelDeployment.model_validate(_deployment_json(model_provider_id=None))
    with pytest.raises(ValueError, match="no associated model_provider_id"):
        client.get_provider_route_openai_url_for_deployment(deployment)


def test_provider_route_for_deployment_malformed_id_raises() -> None:
    """A provider id without a ``workspace/`` prefix raises a clear error, not an unpack crash."""
    client = ModelsClient(base_url=BASE, workspace="default")
    deployment = ModelDeployment.model_validate(_deployment_json(model_provider_id="no-slash-here"))
    with pytest.raises(ValueError, match="malformed model_provider_id"):
        client.get_provider_route_openai_url_for_deployment(deployment)


# ---------------------------------------------------------------------------
# Deployment polling
# ---------------------------------------------------------------------------


def test_wait_for_deployment_status_reaches_ready() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.side_effect = [
        httpx.Response(200, request=httpx.Request("GET", BASE), json=_deployment_json("PENDING")),
        httpx.Response(200, request=httpx.Request("GET", BASE), json=_deployment_json("READY")),
    ]
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.wait_for_deployment_status("my-deploy", "READY", poll_interval=0.0) is True


def test_wait_for_deployment_status_deleted_on_404() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(404, request=httpx.Request("GET", BASE), json={"detail": "x"})
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.wait_for_deployment_status("my-deploy", "DELETED", poll_interval=0.0) is True


def test_wait_for_deployment_status_error_returns_false() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(
        200, request=httpx.Request("GET", BASE), json=_deployment_json("ERROR", status_message="boom")
    )
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.wait_for_deployment_status("my-deploy", "READY", poll_interval=0.0) is False


def test_wait_for_deployment_status_uses_history_tail() -> None:
    history = [
        {"timestamp": "2020-01-01T00:00:01Z", "status": "PENDING", "status_message": ""},
        {"timestamp": "2020-01-01T00:00:05Z", "status": "READY", "status_message": "up"},
    ]
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(
        200, request=httpx.Request("GET", BASE), json=_deployment_json("PENDING", history=history)
    )
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    # Top-level status is PENDING but history tail is READY -> reached.
    assert client.wait_for_deployment_status("my-deploy", "READY", poll_interval=0.0) is True


def test_wait_for_deployment_status_timeout() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(
        200, request=httpx.Request("GET", BASE), json=_deployment_json("PENDING")
    )
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.wait_for_deployment_status("my-deploy", "READY", timeout=0, poll_interval=0.0) is False


# ---------------------------------------------------------------------------
# Provider polling
# ---------------------------------------------------------------------------


def test_wait_for_provider_status_reaches_ready() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.side_effect = [
        httpx.Response(200, request=httpx.Request("GET", BASE), json=_provider_json("PENDING")),
        httpx.Response(200, request=httpx.Request("GET", BASE), json=_provider_json("READY")),
    ]
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.wait_for_provider_status("my-provider", "READY", poll_interval=0.0) is True


def test_wait_for_provider_status_error_returns_false() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(
        200, request=httpx.Request("GET", BASE), json=_provider_json("ERROR", status_message="boom")
    )
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.wait_for_provider_status("my-provider", "READY", poll_interval=0.0) is False


def test_wait_for_provider_status_not_found_returns_false() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(404, request=httpx.Request("GET", BASE), json={"detail": "x"})
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.wait_for_provider_status("my-provider", "READY", poll_interval=0.0) is False


def test_wait_for_provider_status_timeout() -> None:
    http = MagicMock(spec=httpx.Client)
    http.request.return_value = httpx.Response(200, request=httpx.Request("GET", BASE), json=_provider_json("PENDING"))
    client = ModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert client.wait_for_provider_status("my-provider", "READY", timeout=0, poll_interval=0.0) is False


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_create_model() -> None:
    http = AsyncMock(spec=httpx.AsyncClient)
    http.request.return_value = httpx.Response(201, request=httpx.Request("POST", BASE), json=_model_json())
    client = AsyncModelsClient(base_url=BASE, workspace="default", http_client=http)

    out = (await client.create_model(body=CreateModelEntityRequest(name="llama"))).data()
    assert out.name == "llama"


@pytest.mark.asyncio
async def test_async_resolve_model_reference_fetches_model_and_provider() -> None:
    http = AsyncMock(spec=httpx.AsyncClient)
    http.request.side_effect = [
        httpx.Response(
            200,
            request=httpx.Request("GET", BASE),
            json=_model_json("judge", model_providers=["default/provider"]),
        ),
        httpx.Response(
            200,
            request=httpx.Request("GET", BASE),
            json=_provider_json(
                "READY",
                name="provider",
                host_url="http://nim.example.test:8000",
                served_models=[
                    {
                        "model_entity_id": "default/judge",
                        "served_model_name": "publisher/upstream-judge",
                    }
                ],
            ),
        ),
    ]
    client = AsyncModelsClient(base_url=BASE, workspace="default", http_client=http)

    result = await client.resolve_model_reference("default/judge")

    assert [call.args[1] for call in http.request.call_args_list] == [
        f"{BASE}/apis/models/v2/workspaces/default/models/judge",
        f"{BASE}/apis/models/v2/workspaces/default/providers/provider",
    ]
    assert result.name == "judge"
    assert result.url == f"{BASE}/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1"
    assert result.host_url == "http://nim.example.test:8000"
    assert result.served_model_name == "publisher/upstream-judge"


@pytest.mark.asyncio
async def test_async_wait_for_deployment_status_ready() -> None:
    http = AsyncMock(spec=httpx.AsyncClient)
    http.request.side_effect = [
        httpx.Response(200, request=httpx.Request("GET", BASE), json=_deployment_json("PENDING")),
        httpx.Response(200, request=httpx.Request("GET", BASE), json=_deployment_json("READY")),
    ]
    client = AsyncModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert await client.wait_for_deployment_status("my-deploy", "READY", poll_interval=0.0) is True


@pytest.mark.asyncio
async def test_async_wait_for_provider_status_ready() -> None:
    http = AsyncMock(spec=httpx.AsyncClient)
    http.request.side_effect = [
        httpx.Response(200, request=httpx.Request("GET", BASE), json=_provider_json("PENDING")),
        httpx.Response(200, request=httpx.Request("GET", BASE), json=_provider_json("READY")),
    ]
    client = AsyncModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert await client.wait_for_provider_status("my-provider", "READY", poll_interval=0.0) is True


@pytest.mark.asyncio
async def test_async_wait_for_provider_status_error_returns_false() -> None:
    http = AsyncMock(spec=httpx.AsyncClient)
    http.request.return_value = httpx.Response(
        200, request=httpx.Request("GET", BASE), json=_provider_json("ERROR", status_message="boom")
    )
    client = AsyncModelsClient(base_url=BASE, workspace="default", http_client=http)

    assert await client.wait_for_provider_status("my-provider", "READY", poll_interval=0.0) is False
