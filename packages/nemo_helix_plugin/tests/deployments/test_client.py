# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DeploymentsClient / AsyncDeploymentsClient via mocked httpx transport.

Mirrors the Secrets/Models client test pattern: a mocked ``httpx`` transport so we
assert the client builds the right request (method, URL, body, query params) and
parses the response into the right typed model, without a live server.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.deployments.client import AsyncDeploymentsClient, DeploymentsClient
from nemo_helix_plugin.deployments.types import (
    CreateDeploymentConfigRequest,
    CreateDeploymentRequest,
    CreateVolumeRequest,
    Deployment,
    DeploymentConfig,
    RequestContainer,
    Volume,
)

BASE = "http://test:8000"
WS = "default"


def _page(data: list[dict]) -> dict:
    """A plugin list response envelope with populated pagination."""
    return {
        "data": data,
        "pagination": {
            "page": 1,
            "page_size": 20,
            "current_page_size": len(data),
            "total_pages": 1,
            "total_results": len(data),
        },
        "sort": "-created_at",
        "filter": None,
    }


def _sync(json_body: dict | None, status: int, method: str, url: str) -> MagicMock:
    mock = MagicMock(spec=httpx.Client)
    kwargs: dict = {"request": httpx.Request(method, url)}
    if json_body is not None:
        kwargs["json"] = json_body
    mock.request.return_value = httpx.Response(status, **kwargs)
    return mock


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------


def test_create_volume_sends_body_and_parses_volume() -> None:
    mock = _sync(
        {"name": "weights", "workspace": WS, "size": "50Gi", "status": "PENDING"},
        201,
        "POST",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/volumes",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    vol = client.create_volume(body=CreateVolumeRequest(name="weights", size="50Gi")).data()

    assert isinstance(vol, Volume)
    assert (vol.name, vol.size, vol.status) == ("weights", "50Gi", "PENDING")
    args, kwargs = mock.request.call_args
    assert args[0] == "POST"
    assert args[1].endswith(f"/workspaces/{WS}/volumes")
    body = json.loads(kwargs["content"])
    assert body["name"] == "weights"
    assert body["size"] == "50Gi"


def test_get_volume_parses_status() -> None:
    mock = _sync(
        {"name": "weights", "workspace": WS, "status": "BOUND"},
        200,
        "GET",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/volumes/weights",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    vol = client.get_volume(name="weights").data()

    assert vol.status == "BOUND"


def test_list_volumes_iterates_and_passes_deep_object_filter() -> None:
    mock = _sync(
        _page([{"name": "a", "workspace": WS, "status": "BOUND"}]),
        200,
        "GET",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/volumes",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    # Per-field filtering uses the deepObject form: the client passes the literal
    # bracket key `filter[status]` straight through, which is what the plugin's
    # parse_deep_object parser reads. A bare `filter=...`/`status=...` key is
    # silently dropped by that parser.
    vols = list(client.list_volumes(query_params={"filter[status]": "BOUND"}).items())

    assert [(v.name, v.status) for v in vols] == [("a", "BOUND")]
    assert mock.request.call_args.kwargs["params"] == {"filter[status]": "BOUND"}


def test_delete_volume_returns_none_and_calls_delete() -> None:
    mock = _sync(None, 204, "DELETE", f"{BASE}/apis/deployments/v2/workspaces/{WS}/volumes/weights")
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    assert client.delete_volume(name="weights").data() is None
    assert mock.request.call_args.args[0] == "DELETE"


def test_get_volume_not_found_raises() -> None:
    mock = _sync(
        {"detail": "Volume default/missing not found"},
        404,
        "GET",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/volumes/missing",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    with pytest.raises(NotFoundError):
        client.get_volume(name="missing").data()


# ---------------------------------------------------------------------------
# Deployment configs
# ---------------------------------------------------------------------------


def test_create_deployment_config_serializes_set_fields() -> None:
    mock = _sync(
        {"name": "srv", "workspace": WS, "restartPolicy": "Always"},
        201,
        "POST",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployment-configs",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    cfg = client.create_deployment_config(
        body=CreateDeploymentConfigRequest(
            name="srv",
            restartPolicy="OnFailure",
            containers=[RequestContainer(name="main", image="vllm:latest")],
        )
    ).data()

    assert isinstance(cfg, DeploymentConfig)
    assert cfg.name == "srv"
    # The shared client serializes request bodies with exclude_unset (snake_case,
    # no by_alias); the plugin request models are populate_by_name so snake_case is
    # accepted on the wire. Only explicitly-set fields are sent.
    body = json.loads(mock.request.call_args.kwargs["content"])
    assert body["name"] == "srv"
    assert body["restart_policy"] == "OnFailure"
    assert body["containers"][0]["image"] == "vllm:latest"
    assert "init_containers" not in body  # unset field omitted


def test_get_deployment_config_parses_config() -> None:
    mock = _sync(
        {"name": "srv", "workspace": WS, "restartPolicy": "Always", "backoffLimit": 6},
        200,
        "GET",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployment-configs/srv",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    cfg = client.get_deployment_config(name="srv").data()

    assert (cfg.name, cfg.restart_policy, cfg.backoff_limit) == ("srv", "Always", 6)


def test_list_deployment_configs_iterates() -> None:
    mock = _sync(
        _page([{"name": "srv", "workspace": WS}]),
        200,
        "GET",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployment-configs",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    cfgs = list(client.list_deployment_configs().items())

    assert [c.name for c in cfgs] == ["srv"]


def test_delete_deployment_config_hits_hyphenated_path() -> None:
    mock = _sync(None, 204, "DELETE", f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployment-configs/srv")
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    client.delete_deployment_config(name="srv").data()

    assert mock.request.call_args.args[1].endswith("/deployment-configs/srv")


# ---------------------------------------------------------------------------
# Deployments
# ---------------------------------------------------------------------------


def test_create_deployment_parses_deployment() -> None:
    mock = _sync(
        {"name": "srv", "workspace": WS, "deployment_config": "srv", "status": "PENDING"},
        201,
        "POST",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployments",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    dep = client.create_deployment(body=CreateDeploymentRequest(name="srv", deployment_config="srv")).data()

    assert isinstance(dep, Deployment)
    assert (dep.name, dep.deployment_config, dep.status) == ("srv", "srv", "PENDING")


def test_get_deployment_parses_deployment() -> None:
    mock = _sync(
        {"name": "srv", "workspace": WS, "deployment_config": "srv", "status": "READY"},
        200,
        "GET",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployments/srv",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    dep = client.get_deployment(name="srv").data()

    assert dep.status == "READY"


def test_list_deployments_passes_status_in() -> None:
    mock = _sync(
        _page([{"name": "srv", "workspace": WS, "deployment_config": "srv", "status": "READY"}]),
        200,
        "GET",
        f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployments",
    )
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    deps = list(client.list_deployments(query_params={"status_in": "PENDING,STARTING,READY"}).items())

    assert [d.name for d in deps] == ["srv"]
    assert mock.request.call_args.kwargs["params"] == {"status_in": "PENDING,STARTING,READY"}


def test_delete_deployment_returns_none() -> None:
    mock = _sync(None, 204, "DELETE", f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployments/srv")
    client = DeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    assert client.delete_deployment(name="srv").data() is None


# ---------------------------------------------------------------------------
# Async surface (smoke — same transport, awaited)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_create_and_list_volumes() -> None:
    mock = MagicMock(spec=httpx.AsyncClient)
    mock.request = AsyncMock(
        return_value=httpx.Response(
            201,
            request=httpx.Request("POST", f"{BASE}/apis/deployments/v2/workspaces/{WS}/volumes"),
            json={"name": "weights", "workspace": WS, "status": "PENDING"},
        )
    )
    client = AsyncDeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    vol = (await client.create_volume(body=CreateVolumeRequest(name="weights"))).data()
    assert vol.name == "weights"

    mock.request = AsyncMock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("GET", f"{BASE}/apis/deployments/v2/workspaces/{WS}/volumes"),
            json=_page([{"name": "weights", "workspace": WS, "status": "BOUND"}]),
        )
    )
    response = await client.list_volumes()
    names = [v.name async for v in response.items()]
    assert names == ["weights"]


@pytest.mark.asyncio
async def test_async_delete_deployment_returns_none() -> None:
    mock = MagicMock(spec=httpx.AsyncClient)
    mock.request = AsyncMock(
        return_value=httpx.Response(
            204, request=httpx.Request("DELETE", f"{BASE}/apis/deployments/v2/workspaces/{WS}/deployments/srv")
        )
    )
    client = AsyncDeploymentsClient(base_url=BASE, workspace=WS, http_client=mock)

    assert (await client.delete_deployment(name="srv")).data() is None
