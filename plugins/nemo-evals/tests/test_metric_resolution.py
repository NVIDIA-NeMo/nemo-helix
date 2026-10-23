# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for evaluator metric reference resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from nemo_evals.jobs.metric_resolution import HelixMetricModelResolver, HelixMetricSecretResolver
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.client import AsyncModelsClient, UnroutableModelError
from nemo_helix_plugin.models.refs import ResolvedModelReference
from nemo_helix_plugin.virtual_models.client import AsyncVirtualModelsClient
from nhx_evals_sdk.values import ModelRef
from nhx_evals_sdk.values.common import SecretRef
from pytest_mock import MockerFixture


@pytest.mark.asyncio
async def test_resolver_returns_the_secret_value(make_secrets_service) -> None:
    secrets = make_secrets_service({("default", "judge-key"): "sk-live-abc"})
    resolver = HelixMetricSecretResolver(secrets.client(), workspace="default")

    assert await resolver.resolve_secret(SecretRef(root="judge-key")) == "sk-live-abc"


@pytest.mark.asyncio
async def test_unqualified_ref_resolves_in_the_request_workspace(make_secrets_service) -> None:
    """An unqualified name must not silently fall back to the client's own default workspace."""
    secrets = make_secrets_service({("team-a", "judge-key"): "sk-team-a"})
    resolver = HelixMetricSecretResolver(secrets.client(), workspace="team-a")

    assert await resolver.resolve_secret(SecretRef(root="judge-key")) == "sk-team-a"
    assert secrets.lookups == [("team-a", "judge-key")]


@pytest.mark.asyncio
async def test_qualified_ref_resolves_in_the_named_workspace(make_secrets_service) -> None:
    """A workspace-qualified ref is honoured; the Secrets service authorizes it, not this resolver."""
    secrets = make_secrets_service({("team-b", "judge-key"): "sk-team-b"})
    resolver = HelixMetricSecretResolver(secrets.client(), workspace="team-a")

    assert await resolver.resolve_secret(SecretRef(root="team-b/judge-key")) == "sk-team-b"
    assert secrets.lookups == [("team-b", "judge-key")]


@pytest.mark.asyncio
async def test_missing_secret_resolves_to_none_rather_than_raising(make_secrets_service) -> None:
    """`None` is the protocol's answer for "no such secret"; the metric turns it into the error."""
    secrets = make_secrets_service()
    resolver = HelixMetricSecretResolver(secrets.client(), workspace="default")

    assert await resolver.resolve_secret(SecretRef(root="absent")) is None


@pytest.mark.asyncio
async def test_malformed_ref_is_rejected(make_secrets_service) -> None:
    secrets = make_secrets_service()
    resolver = HelixMetricSecretResolver(secrets.client(), workspace="default")

    with pytest.raises(ValueError):
        await resolver.resolve_secret(SecretRef(root="too/many/parts"))
    assert secrets.lookups == []


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
    passthrough = SimpleNamespace(autoprovisioned=True, default_model_entity=f"{entity.workspace}/{entity.name}")
    mocker.patch.object(
        AsyncVirtualModelsClient,
        "get_virtual_model",
        AsyncMock(return_value=mocker.Mock(data=mocker.Mock(return_value=passthrough))),
    )
    models_client = AsyncModelsClient.from_client(
        AsyncNemoClient(base_url=client_base_url, http_client=AsyncMock(spec=httpx.AsyncClient))
    )

    model = await HelixMetricModelResolver(models_client).resolve_model(ModelRef(root="default/judge"))

    assert model.url == f"{expected_base_url}{JUDGE_ROUTE}"


async def test_model_resolver_checks_the_virtual_model_a_ref_names(mocker: MockerFixture) -> None:
    """A judge ref to a VirtualModel is read-checked as a VirtualModel, not only as a model."""
    resolved = ResolvedModelReference(
        url=f"http://nemo-api:8080{JUDGE_ROUTE.replace('judge', 'cb-judge')}",
        name="cb-judge",
        host_url=None,
        virtual_model=True,
    )
    mocker.patch.object(AsyncModelsClient, "resolve_model_reference", AsyncMock(return_value=resolved))
    models_client = AsyncModelsClient.from_client(
        AsyncNemoClient(base_url="http://nemo-api:8080", http_client=AsyncMock(spec=httpx.AsyncClient))
    )
    checked: list[str] = []

    async def authorize(method: str, path: str) -> None:
        checked.append(path)

    model = await HelixMetricModelResolver(models_client, authorize=authorize).resolve_model(
        ModelRef(root="default/cb-judge")
    )

    assert checked == [
        "/apis/models/v2/workspaces/default/models/cb-judge",
        "/apis/inference-gateway/v2/workspaces/default/virtual-models/cb-judge",
    ]
    assert model.url.endswith("/model/cb-judge/-/v1")
    assert model.served_model_name is None


@pytest.mark.parametrize(
    ("allowed", "virtual_model", "expected_checks"),
    [
        ({"virtual-models"}, True, ["models", "virtual-models"]),
        ({"virtual-models"}, False, ["models", "virtual-models"]),
    ],
    ids=["virtual-model", "entity-served-by-its-virtual-model"],
)
async def test_model_resolver_accepts_read_access_to_either_kind(
    mocker: MockerFixture, allowed: set[str], virtual_model: bool, expected_checks: list[str]
) -> None:
    """Every model is called through a VirtualModel of the same name, so reading either one is enough."""
    resolved = ResolvedModelReference(
        url="http://nemo-api:8080/apis/inference-gateway/v2/workspaces/default/model/mm/-/v1",
        name="mm",
        host_url=None,
        virtual_model=virtual_model,
    )
    mocker.patch.object(AsyncModelsClient, "resolve_model_reference", AsyncMock(return_value=resolved))
    models_client = AsyncModelsClient.from_client(
        AsyncNemoClient(base_url="http://nemo-api:8080", http_client=AsyncMock(spec=httpx.AsyncClient))
    )
    checked: list[str] = []

    async def authorize(method: str, path: str) -> None:
        kind = path.split("/")[-2]
        checked.append(kind)
        if kind not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")

    await HelixMetricModelResolver(models_client, authorize=authorize).resolve_model(ModelRef(root="default/mm"))

    assert checked == expected_checks


async def test_model_resolver_says_why_an_existing_model_cannot_be_called(mocker: MockerFixture) -> None:
    """The model exists, so the generic "not found" advice would send the caller looking for the wrong fix."""
    unroutable = UnroutableModelError("default/mm", "http://nemo-api:8080/model/mm")
    mocker.patch.object(AsyncModelsClient, "resolve_model_reference", AsyncMock(side_effect=unroutable))
    models_client = AsyncModelsClient.from_client(
        AsyncNemoClient(base_url="http://nemo-api:8080", http_client=AsyncMock(spec=httpx.AsyncClient))
    )

    with pytest.raises(ValueError, match="no VirtualModel and no provider"):
        await HelixMetricModelResolver(models_client).resolve_model(ModelRef(root="default/mm"))


async def test_model_resolver_refuses_a_caller_who_can_read_neither(mocker: MockerFixture) -> None:
    resolve = mocker.patch.object(AsyncModelsClient, "resolve_model_reference", AsyncMock())
    models_client = AsyncModelsClient.from_client(
        AsyncNemoClient(base_url="http://nemo-api:8080", http_client=AsyncMock(spec=httpx.AsyncClient))
    )

    async def authorize(method: str, path: str) -> None:
        raise HTTPException(status_code=403, detail=f"Forbidden: {path}")

    with pytest.raises(HTTPException, match="/models/mm"):
        await HelixMetricModelResolver(models_client, authorize=authorize).resolve_model(ModelRef(root="default/mm"))
    resolve.assert_not_awaited()


@pytest.mark.parametrize(("allowed", "expected_host"), [({"virtual-models"}, None), ({"models"}, "http://nim:8000")])
async def test_only_a_caller_who_may_read_the_entity_gets_its_provider_address(
    mocker: MockerFixture, allowed: set[str], expected_host: str | None
) -> None:
    """The provider's address bypasses the gateway and its middleware, so reading the VirtualModel isn't enough."""
    resolved = ResolvedModelReference(
        url="http://nemo-api:8080/apis/inference-gateway/v2/workspaces/default/model/mm/-/v1",
        name="mm",
        host_url="http://nim:8000",
        served_model_name="org/mm",
    )
    mocker.patch.object(AsyncModelsClient, "resolve_model_reference", AsyncMock(return_value=resolved))
    models_client = AsyncModelsClient.from_client(
        AsyncNemoClient(base_url="http://nemo-api:8080", http_client=AsyncMock(spec=httpx.AsyncClient))
    )

    async def authorize(method: str, path: str) -> None:
        if path.split("/")[-2] not in allowed:
            raise HTTPException(status_code=403, detail="Forbidden")

    model = await HelixMetricModelResolver(models_client, authorize=authorize).resolve_model(
        ModelRef(root="default/mm")
    )

    assert model.url.endswith("/model/mm/-/v1")
    assert model.host_url == expected_host
    assert model.served_model_name == (None if expected_host is None else "org/mm")
