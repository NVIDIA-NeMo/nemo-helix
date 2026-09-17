# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed resource dispatch and transport auth.

Covers two behaviours that only surface through the compatibility layer:

- ``sdk.inference.*`` must hand back a resource client matching the owning
  client's sync/async flavour.
- Raw calls through the exposed ``_client`` transport must stay authenticated.
"""

from __future__ import annotations

import httpx
import pytest
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient

BASE = "http://test:8000"


# ---------------------------------------------------------------------------
# sdk.inference.* sync/async dispatch
# ---------------------------------------------------------------------------


def test_inference_namespace_returns_sync_clients_for_sync_client() -> None:
    from nemo_platform_plugin.models.client import ModelsClient
    from nemo_platform_plugin.virtual_models.client import VirtualModelsClient

    client = NemoClient(base_url=BASE)

    assert isinstance(client.inference.providers, ModelsClient)
    assert isinstance(client.inference.deployments, ModelsClient)
    assert isinstance(client.inference.deployment_configs, ModelsClient)
    assert isinstance(client.inference.virtual_models, VirtualModelsClient)


def test_inference_namespace_returns_async_clients_for_async_client() -> None:
    """An async client must never wrap its AsyncClient in a sync resource."""
    from nemo_platform_plugin.models.client import AsyncModelsClient
    from nemo_platform_plugin.virtual_models.client import AsyncVirtualModelsClient

    client = AsyncNemoClient(base_url=BASE)

    assert isinstance(client.inference.providers, AsyncModelsClient)
    assert isinstance(client.inference.deployments, AsyncModelsClient)
    assert isinstance(client.inference.deployment_configs, AsyncModelsClient)
    assert isinstance(client.inference.virtual_models, AsyncVirtualModelsClient)


@pytest.mark.parametrize(
    ("client_factory", "transport_type"),
    [(NemoClient, httpx.Client), (AsyncNemoClient, httpx.AsyncClient)],
)
def test_inference_resources_transport_matches_flavour(client_factory, transport_type) -> None:
    client = client_factory(base_url=BASE)

    assert isinstance(client.inference.providers._http, transport_type)
    assert isinstance(client.inference.virtual_models._http, transport_type)


def test_convenience_properties_return_sync_clients_for_sync_client() -> None:
    from nemo_platform_plugin.agent_hardener.client import AgentHardenerClient
    from nemo_platform_plugin.agents.client import AgentsClient
    from nemo_platform_plugin.auditor.client import AuditorClient
    from nemo_platform_plugin.data_designer.client import DataDesignerClient
    from nemo_platform_plugin.evaluator.client import EvaluatorClient
    from nemo_platform_plugin.files.client import FilesClient
    from nemo_platform_plugin.guardrail.client import GuardrailClient
    from nemo_platform_plugin.jobs.client import JobsClient
    from nemo_platform_plugin.models.client import ModelsClient
    from nemo_platform_plugin.projects.client import ProjectsClient
    from nemo_platform_plugin.secrets.client import SecretsClient
    from nemo_platform_plugin.workspaces.client import WorkspacesClient

    client = NemoClient(base_url=BASE)
    expected_resources = [
        ("files", FilesClient),
        ("models", ModelsClient),
        ("workspaces", WorkspacesClient),
        ("secrets", SecretsClient),
        ("jobs", JobsClient),
        ("agents", AgentsClient),
        ("auditor", AuditorClient),
        ("guardrail", GuardrailClient),
        ("evaluator", EvaluatorClient),
        ("projects", ProjectsClient),
        ("data_designer", DataDesignerClient),
        ("agent_hardener", AgentHardenerClient),
    ]

    for attr, expected_type in expected_resources:
        resource = getattr(client, attr)
        assert isinstance(resource, expected_type)
        assert isinstance(resource._http, httpx.Client)


def test_convenience_properties_return_async_clients_for_async_client() -> None:
    from nemo_platform_plugin.agent_hardener.client import AsyncAgentHardenerClient
    from nemo_platform_plugin.agents.client import AsyncAgentsClient
    from nemo_platform_plugin.auditor.client import AsyncAuditorClient
    from nemo_platform_plugin.data_designer.client import AsyncDataDesignerClient
    from nemo_platform_plugin.evaluator.client import AsyncEvaluatorClient
    from nemo_platform_plugin.files.client import AsyncFilesClient
    from nemo_platform_plugin.guardrail.client import AsyncGuardrailClient
    from nemo_platform_plugin.jobs.client import AsyncJobsClient
    from nemo_platform_plugin.models.client import AsyncModelsClient
    from nemo_platform_plugin.projects.client import AsyncProjectsClient
    from nemo_platform_plugin.secrets.client import AsyncSecretsClient
    from nemo_platform_plugin.workspaces.client import AsyncWorkspacesClient

    client = AsyncNemoClient(base_url=BASE)
    expected_resources = [
        ("files", AsyncFilesClient),
        ("models", AsyncModelsClient),
        ("workspaces", AsyncWorkspacesClient),
        ("secrets", AsyncSecretsClient),
        ("jobs", AsyncJobsClient),
        ("agents", AsyncAgentsClient),
        ("auditor", AsyncAuditorClient),
        ("guardrail", AsyncGuardrailClient),
        ("evaluator", AsyncEvaluatorClient),
        ("projects", AsyncProjectsClient),
        ("data_designer", AsyncDataDesignerClient),
        ("agent_hardener", AsyncAgentHardenerClient),
    ]

    for attr, expected_type in expected_resources:
        resource = getattr(client, attr)
        assert isinstance(resource, expected_type)
        assert isinstance(resource._http, httpx.AsyncClient)


@pytest.mark.parametrize("client_factory", [NemoClient, AsyncNemoClient])
def test_unknown_attribute_still_raises(client_factory) -> None:
    client = client_factory(base_url=BASE)

    with pytest.raises(AttributeError, match="nope"):
        _ = client.nope


# ---------------------------------------------------------------------------
# Auth on the raw _client transport
# ---------------------------------------------------------------------------


def test_raw_client_calls_are_authenticated() -> None:
    """Plugin resources using owner._client bypass send() but keep auth."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    client = NemoClient(base_url=BASE, auth="tok")
    client._http._transport = httpx.MockTransport(handler)

    transport = client._client
    assert isinstance(transport, httpx.Client)
    transport.get(f"{BASE}/apis/anything")

    assert seen[0].headers["Authorization"] == "Bearer tok"


@pytest.mark.asyncio
async def test_raw_async_client_calls_are_authenticated() -> None:
    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    client = AsyncNemoClient(base_url=BASE, auth="tok")
    client._http._transport = httpx.MockTransport(handler)

    transport = client._client
    assert isinstance(transport, httpx.AsyncClient)
    await transport.get(f"{BASE}/apis/anything")

    assert seen[0].headers["Authorization"] == "Bearer tok"


def test_transport_auth_does_not_override_explicit_header() -> None:
    """send() and per-call overrides stay authoritative over transport auth."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    client = NemoClient(base_url=BASE, auth="tok")
    client._http._transport = httpx.MockTransport(handler)

    transport = client._client
    assert isinstance(transport, httpx.Client)
    transport.get(f"{BASE}/apis/anything", headers={"Authorization": "Bearer explicit"})

    assert seen[0].headers["Authorization"] == "Bearer explicit"


def test_no_auth_configured_leaves_transport_unauthenticated() -> None:
    client = NemoClient(base_url=BASE)

    assert client._http.auth is None


def test_shared_transport_keeps_auth_across_from_client() -> None:
    """Resource clients built from a parent share its authenticated transport."""
    from nemo_platform_plugin.client.auth import TokenProviderAuth
    from nemo_platform_plugin.models.client import ModelsClient

    parent = NemoClient(base_url=BASE, auth="tok")
    child = ModelsClient.from_client(parent)

    assert child._http is parent._http
    assert isinstance(child._http.auth, TokenProviderAuth)
