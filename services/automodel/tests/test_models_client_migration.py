# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-config resolution through the typed Models client, over a mocked httpx transport.

Driving a real ``AsyncNemoClient`` -> ``AsyncModelsClient`` chain asserts the
wire contract (method, path, parsed model and error mapping) rather than
restating the call the implementation happens to make.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.jobs.exceptions import PlatformJobCompilationError
from nemo_platform_plugin.models.client import AsyncModelsClient
from nemo_platform_plugin.models.types import ModelDeploymentConfig
from nmp.automodel.app.jobs.compiler import _resolve_deployment_config_ref
from nmp.customization_common.service.platform_client import AsyncCustomizationPlatformClients

BASE = "http://test:8000"


def _config_json(name: str = "config", workspace: str = "other", **model_spec: object) -> dict:
    spec = {"model_name": "llama", "model_namespace": workspace, "lora_enabled": True}
    spec.update(model_spec)
    return {
        "id": f"config-{name}",
        "name": name,
        "workspace": workspace,
        "entity_version": 1,
        "engine": "nim",
        "model_spec": spec,
        "executor_config": {"gpu": 1},
        "created_at": "2020-01-01T00:00:00Z",
        "updated_at": "2020-01-01T00:00:00Z",
    }


def _recording_transport(
    status: int = 200, payload: dict | None = None
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, request=request, json=payload if payload is not None else _config_json())

    return httpx.MockTransport(handler), seen


def _platform(transport: httpx.MockTransport) -> AsyncCustomizationPlatformClients:
    client = AsyncNemoClient(base_url=BASE, workspace="default", http_client=httpx.AsyncClient(transport=transport))
    return AsyncCustomizationPlatformClients(files=AsyncMock(), models=AsyncModelsClient.from_client(client))


@pytest.mark.asyncio
async def test_resolve_deployment_config_hits_the_deployment_config_endpoint() -> None:
    transport, seen = _recording_transport()

    result = await _resolve_deployment_config_ref("other/config", "default", _platform(transport))

    assert isinstance(result, ModelDeploymentConfig)
    assert (result.workspace, result.name) == ("other", "config")
    assert result.model_spec.lora_enabled is True
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert str(seen[0].url) == f"{BASE}/apis/models/v2/workspaces/other/deployment-configs/config"


@pytest.mark.asyncio
async def test_resolve_deployment_config_defaults_to_the_job_workspace() -> None:
    transport, seen = _recording_transport(payload=_config_json(workspace="default"))

    await _resolve_deployment_config_ref("config", "default", _platform(transport))

    assert str(seen[0].url) == f"{BASE}/apis/models/v2/workspaces/default/deployment-configs/config"


@pytest.mark.asyncio
async def test_resolve_deployment_config_maps_404_to_compilation_error() -> None:
    transport, _ = _recording_transport(404, {"detail": "missing"})

    with pytest.raises(PlatformJobCompilationError, match="does not exist in workspace 'other'"):
        await _resolve_deployment_config_ref("other/config", "default", _platform(transport))
