# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for evaluator metric reference resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from nemo_evals.jobs.metric_resolution import HelixMetricModelResolver, HelixMetricSecretResolver
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from nhx_evals_sdk.values import ModelRef
from nhx_evals_sdk.values.common import SecretRef
from pytest_mock import MockerFixture


@pytest.mark.asyncio
async def test_resolver_returns_the_secret_value(make_secrets_client) -> None:
    client = make_secrets_client({("default", "judge-key"): "sk-live-abc"})
    resolver = HelixMetricSecretResolver(client, workspace="default")

    assert await resolver.resolve_secret(SecretRef(root="judge-key")) == "sk-live-abc"


@pytest.mark.asyncio
async def test_unqualified_ref_resolves_in_the_request_workspace(make_secrets_client) -> None:
    """An unqualified name must not silently fall back to the client's own default workspace."""
    client = make_secrets_client({("team-a", "judge-key"): "sk-team-a"})
    resolver = HelixMetricSecretResolver(client, workspace="team-a")

    assert await resolver.resolve_secret(SecretRef(root="judge-key")) == "sk-team-a"
    assert client.lookups == [("team-a", "judge-key")]


@pytest.mark.asyncio
async def test_qualified_ref_resolves_in_the_named_workspace(make_secrets_client) -> None:
    """A workspace-qualified ref is honoured; the Secrets service authorizes it, not this resolver."""
    client = make_secrets_client({("team-b", "judge-key"): "sk-team-b"})
    resolver = HelixMetricSecretResolver(client, workspace="team-a")

    assert await resolver.resolve_secret(SecretRef(root="team-b/judge-key")) == "sk-team-b"
    assert client.lookups == [("team-b", "judge-key")]


@pytest.mark.asyncio
async def test_missing_secret_resolves_to_none_rather_than_raising(make_secrets_client) -> None:
    """`None` is the protocol's answer for "no such secret"; the metric turns it into the error."""
    client = make_secrets_client()
    resolver = HelixMetricSecretResolver(client, workspace="default")

    assert await resolver.resolve_secret(SecretRef(root="absent")) is None


@pytest.mark.asyncio
async def test_malformed_ref_is_rejected(make_secrets_client) -> None:
    client = make_secrets_client()
    resolver = HelixMetricSecretResolver(client, workspace="default")

    with pytest.raises(ValueError):
        await resolver.resolve_secret(SecretRef(root="too/many/parts"))
    assert client.lookups == []


JUDGE_ROUTE = "/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1"


@pytest.mark.parametrize(
    ("client_base_url", "env", "expected_base_url"),
    [
        ("http://localhost:8080", {"NHX_INTERNAL_BASE_URL": "http://nemo-api:8080"}, "http://nemo-api:8080"),
        ("http://127.0.0.1:8080", {"NHX_INTERNAL_BASE_URL": "http://nemo-api:8080/"}, "http://nemo-api:8080"),
        ("http://localhost:8080", {"NEMO_INTERNAL_BASE_URL": "http://nemo-api:8080"}, "http://nemo-api:8080"),
        ("http://localhost:8080", {}, "http://localhost:8080"),
        ("http://nemo-api:8080", {"NHX_INTERNAL_BASE_URL": "http://other:8080"}, "http://nemo-api:8080"),
    ],
)
async def test_model_resolver_mints_job_reachable_gateway_urls(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    client_base_url: str,
    env: dict[str, str],
    expected_base_url: str,
) -> None:
    """Under embedded auth the API resolves over loopback; jobs need the in-cluster service URL instead."""
    monkeypatch.delenv("NEMO_INTERNAL_BASE_URL", raising=False)
    monkeypatch.delenv("NHX_INTERNAL_BASE_URL", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    entity = SimpleNamespace(workspace="default", name="judge", model_providers=[])
    mocker.patch.object(
        AsyncModelsClient, "get_model", AsyncMock(return_value=mocker.Mock(data=mocker.Mock(return_value=entity)))
    )
    models_client = AsyncModelsClient.from_client(
        AsyncNemoClient(base_url=client_base_url, http_client=AsyncMock(spec=httpx.AsyncClient))
    )

    model = await HelixMetricModelResolver(models_client).resolve_model(ModelRef(root="default/judge"))

    assert model.url == f"{expected_base_url}{JUDGE_ROUTE}"
