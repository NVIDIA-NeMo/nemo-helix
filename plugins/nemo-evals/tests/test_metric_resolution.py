# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for evaluator metric reference resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit

import httpx
import pytest
from nemo_agents_plugin.jobs import gateway_proxy
from nemo_evals.jobs.metric_resolution import (
    HelixMetricModelResolver,
    HelixMetricSecretResolver,
    gateway_metric_models_as_job,
)
from nemo_helix_plugin.client.auth import StaticToken
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.models.client import AsyncModelsClient
from nhx_evals_sdk.metrics.llm_judge import LLMJudgeMetric
from nhx_evals_sdk.metrics.ragas.metrics import TopicAdherenceMetric
from nhx_evals_sdk.values import Model, ModelRef
from nhx_evals_sdk.values.common import SecretRef
from nhx_evals_sdk.values.scores import JSONScoreParser, RangeScore
from pytest_httpserver import HTTPServer
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
    models_client = AsyncModelsClient.from_client(
        AsyncNemoClient(base_url=client_base_url, http_client=AsyncMock(spec=httpx.AsyncClient))
    )

    model = await HelixMetricModelResolver(models_client).resolve_model(ModelRef(root="default/judge"))

    assert model.url == f"{expected_base_url}{JUDGE_ROUTE}"


def _url(model: Model | ModelRef | None) -> str:
    assert isinstance(model, Model)
    return model.url


def _judge(model: Model) -> LLMJudgeMetric:
    return LLMJudgeMetric(
        model=model, scores=[RangeScore(name="q", minimum=0, maximum=1, parser=JSONScoreParser(json_path="q"))]
    )


def test_gateway_models_point_at_the_job_proxy_for_the_block_and_are_restored(
    monkeypatch: pytest.MonkeyPatch, httpserver: HTTPServer
) -> None:
    """Only credential-less gateway models move; RAGAS rebuilds its client config from the moved model."""
    monkeypatch.delenv("NHX_PRINCIPAL", raising=False)
    monkeypatch.setenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", "/job/proof-token")
    monkeypatch.setenv("NHX_BASE_URL", httpserver.url_for(""))
    monkeypatch.setattr(
        gateway_proxy, "resolve_workload_exchange_provider", Mock(return_value=StaticToken("job-access-token"))
    )
    resolved = f"http://nemo-api:8080{JUDGE_ROUTE}"
    ragas = TopicAdherenceMetric(metric_mode="f1", judge_model=Model(url=resolved, name="judge"))
    judge = _judge(Model(url=resolved, name="judge"))
    keyed = _judge(Model(url=resolved, name="judge", api_key_secret="judge-key"))
    external = _judge(Model(url="https://judge.example.test/v1", name="judge"))

    with gateway_metric_models_as_job([ragas, judge, keyed, external]):
        proxied = urlsplit(_url(judge.model))
        assert (proxied.hostname, proxied.path) == ("127.0.0.1", JUDGE_ROUTE)
        assert _url(ragas.judge_model) == _url(judge.model)
        assert ragas._llm_model is not None and ragas._llm_model["base_url"].startswith(
            f"http://127.0.0.1:{proxied.port}/"
        )
        assert _url(keyed.model) == resolved
        assert _url(external.model) == "https://judge.example.test/v1"

    assert _url(ragas.judge_model) == _url(judge.model) == resolved
    assert ragas._llm_model is not None and ragas._llm_model["base_url"].startswith("http://nemo-api:8080/")


def test_gateway_models_are_left_alone_when_the_job_has_no_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """An auth-disabled platform's jobs call the gateway as resolved, with no proxy."""
    monkeypatch.delenv("NHX_PRINCIPAL", raising=False)
    monkeypatch.delenv("NHX_WORKLOAD_IDENTITY_TOKEN_FILE", raising=False)
    resolved = f"http://nemo-api:8080{JUDGE_ROUTE}"
    judge = _judge(Model(url=resolved, name="judge"))

    with gateway_metric_models_as_job([judge]):
        assert _url(judge.model) == resolved
