# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nemo_platform_plugin import nooa_model_client
from nemo_platform_plugin.nooa_model_client import (
    ConfiguredModelClients,
    ConfiguredModelRefs,
    activate_model_clients,
    configured_model_refs,
    get_configured_model_refs,
    get_default_model,
    get_fast_model,
    resolve_model_clients,
)


def _model_entity(**attrs):
    defaults = dict(
        workspace="default",
        name="model",
        backend_format="OPENAI_CHAT",
        model_providers=[],
        api_endpoint=None,
    )
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def _mock_models_client(models, providers=None):
    """Build a mock ``AsyncModelsClient`` for model resolution tests.

    ``models`` maps ``(workspace, name)`` to the Model Entity ``get_model`` should
    return. ``get_model`` is an AsyncMock; its awaited result must expose a
    ``.data()`` accessor mirroring ``NemoResponse``.
    """
    client = MagicMock()
    client._auth = None
    client.default_headers = {}
    client.get_model_entity_route_openai_url.return_value = "http://platform/model/example/-/v1"
    providers = providers or {}

    async def _get_model(**kwargs):
        entity = models.get((kwargs.get("workspace"), kwargs.get("name")))
        return SimpleNamespace(data=lambda: entity)

    async def _get_provider(**kwargs):
        provider = providers.get((kwargs.get("workspace"), kwargs.get("name")))
        return SimpleNamespace(data=lambda: provider)

    client.get_model = AsyncMock(side_effect=_get_model)
    client.get_provider = AsyncMock(side_effect=_get_provider)
    return client


def _patch_models_client(models, providers=None):
    client = _mock_models_client(models, providers=providers)
    ctx = patch(
        "nemo_platform_plugin.nooa_model_client.AsyncModelsClient.from_client",
        return_value=client,
    )
    return ctx, client


def test_configured_model_refs_uses_default_for_missing_fast(monkeypatch):
    monkeypatch.setattr(
        nooa_model_client,
        "get_context",
        lambda: SimpleNamespace(default_model="default/quality", fast_model=None),
    )

    assert configured_model_refs() == ConfiguredModelRefs(
        default="default/quality",
        fast="default/quality",
    )


def test_configured_model_refs_requires_default(monkeypatch):
    monkeypatch.setattr(
        nooa_model_client,
        "get_context",
        lambda: SimpleNamespace(default_model=None, fast_model="default/fast"),
    )

    with pytest.raises(ValueError, match="No default model"):
        configured_model_refs()


def test_configured_model_refs_rejects_unqualified_default(monkeypatch):
    """An unqualified NEMO_DEFAULT_MODEL must fail here, not inside a running job."""
    monkeypatch.setattr(
        nooa_model_client,
        "get_context",
        lambda: SimpleNamespace(default_model="nvidia-nemotron-mini-4b-instruct", fast_model=None),
    )

    with pytest.raises(ValueError, match="must use workspace/name format") as excinfo:
        configured_model_refs()

    message = str(excinfo.value)
    assert "default/nvidia-nemotron-mini-4b-instruct" in message
    assert "NEMO_DEFAULT_MODEL" in message


def test_configured_model_refs_rejects_unqualified_fast(monkeypatch):
    monkeypatch.setattr(
        nooa_model_client,
        "get_context",
        lambda: SimpleNamespace(default_model="default/quality", fast_model="bare-fast"),
    )

    with pytest.raises(ValueError, match="NEMO_FAST_MODEL"):
        configured_model_refs()


def test_configured_model_refs_rejects_extra_path_segments(monkeypatch):
    monkeypatch.setattr(
        nooa_model_client,
        "get_context",
        lambda: SimpleNamespace(default_model="default/group/model", fast_model=None),
    )

    with pytest.raises(ValueError, match="NEMO_DEFAULT_MODEL") as excinfo:
        configured_model_refs()

    message = str(excinfo.value)
    assert "default/group/model" in message
    assert "must use workspace/name format" in message


async def test_resolve_model_clients_deduplicates_same_model(monkeypatch):
    model_entity = _model_entity(
        workspace="default",
        name="gpt-4-1",
        backend_format="OPENAI_CHAT",
    )
    models = {("default", "gpt-4-1"): model_entity}
    ctx, client = _patch_models_client(models)
    client.get_model_entity_route_openai_url.return_value = "http://platform/model/gpt-4-1/-/v1"
    client.default_headers = {"x-test": "value"}
    completion_client = MagicMock()
    factory = MagicMock(return_value=completion_client)
    with ctx:
        monkeypatch.setattr(nooa_model_client, "CompletionClient", factory)

        result = await resolve_model_clients(
            MagicMock(),
            ConfiguredModelRefs(default="default/gpt-4-1", fast="default/gpt-4-1"),
        )

    assert result.default is completion_client
    assert result.fast is completion_client
    client.get_model.assert_awaited_once_with(name="gpt-4-1", workspace="default")
    factory.assert_called_once_with(
        "openai/gpt-4-1",
        api_base="http://platform/model/gpt-4-1/-/v1",
        api_key="not-needed",
        base_model="openai/gpt-4-1",
        extra_headers={"x-test": "value", "accept-encoding": "identity"},
        drop_params=True,
        _skip_responses_api_bridge=True,
    )
    assert client.default_headers == {"x-test": "value"}


async def test_resolve_model_clients_uses_anthropic_route_shape(monkeypatch):
    model_entity = _model_entity(
        workspace="default",
        name="claude-sonnet-4",
        backend_format="ANTHROPIC_MESSAGES",
    )
    models = {("default", "claude-sonnet-4"): model_entity}
    ctx, client = _patch_models_client(models)
    client.get_model_entity_route_openai_url.return_value = "http://platform/model/claude-sonnet-4/-/v1"
    factory = MagicMock(return_value=MagicMock())
    with ctx:
        monkeypatch.setattr(nooa_model_client, "CompletionClient", factory)

        await resolve_model_clients(
            MagicMock(),
            ConfiguredModelRefs(default="default/claude-sonnet-4", fast="default/claude-sonnet-4"),
        )

    factory.assert_called_once_with(
        "anthropic/claude-sonnet-4",
        api_base="http://platform/model/claude-sonnet-4/-",
        api_key="not-needed",
        base_model="anthropic/claude-sonnet-4",
        extra_headers={"accept-encoding": "identity"},
        drop_params=True,
    )


async def test_resolve_model_clients_rejects_unsupported_backend_format():
    model_entity = _model_entity(
        workspace="default",
        name="responses-only",
        backend_format="OPENAI_RESPONSES",
    )
    models = {("default", "responses-only"): model_entity}
    ctx, _client = _patch_models_client(models)

    with ctx:
        with pytest.raises(ValueError, match="unsupported backend format 'OPENAI_RESPONSES'"):
            await resolve_model_clients(
                MagicMock(),
                ConfiguredModelRefs(default="default/responses-only", fast="default/responses-only"),
            )


async def test_resolve_model_clients_requires_workspace_qualified_refs(monkeypatch):
    ctx, _client = _patch_models_client({})
    with ctx:
        with pytest.raises(ValueError, match="workspace/name"):
            await resolve_model_clients(
                MagicMock(),
                ConfiguredModelRefs(default="unqualified", fast="unqualified"),
            )


async def test_resolve_model_clients_uses_provider_served_name(monkeypatch):
    model_entity = _model_entity(
        workspace="default",
        name="gpt-5-6-sol",
        backend_format="OPENAI_CHAT",
        model_providers=["default/openai"],
        api_endpoint=SimpleNamespace(model_id="gpt-5-6-sol"),
    )
    provider = SimpleNamespace(
        served_models=[
            SimpleNamespace(
                model_entity_id="default/gpt-5-6-sol",
                served_model_name="gpt-5.6-sol",
            )
        ]
    )
    ctx, client = _patch_models_client({("default", "gpt-5-6-sol"): model_entity}, {("default", "openai"): provider})
    client.get_model_entity_route_openai_url.return_value = "http://platform/model/gpt-5-6-sol/-/v1"
    factory = MagicMock(return_value=MagicMock())
    async_sdk = MagicMock()
    with ctx:
        monkeypatch.setattr(nooa_model_client, "CompletionClient", factory)

        await resolve_model_clients(
            async_sdk,
            ConfiguredModelRefs(default="default/gpt-5-6-sol", fast="default/gpt-5-6-sol"),
        )

    client.get_provider.assert_awaited_once_with(name="openai", workspace="default")
    factory.assert_called_once_with(
        "openai/gpt-5.6-sol",
        api_base="http://platform/model/gpt-5-6-sol/-/v1",
        api_key="not-needed",
        base_model="openai/gpt-5.6-sol",
        extra_headers={"accept-encoding": "identity"},
        drop_params=True,
        _skip_responses_api_bridge=True,
    )


async def test_resolve_model_clients_refreshes_auth_for_completion_calls(monkeypatch):
    class _TokenProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def get_access_token(self) -> str:
            self.calls += 1
            return f"token-{self.calls}"

    model_entity = _model_entity(
        workspace="default",
        name="gpt-4-1",
        backend_format="OPENAI_CHAT",
    )
    ctx, client = _patch_models_client({("default", "gpt-4-1"): model_entity})
    client.default_headers = {"authorization": "Bearer stale", "x-test": "value"}
    token_provider = _TokenProvider()
    client._auth = token_provider
    captured_headers: list[dict[str, str]] = []

    async def _recording_acall(
        self,
        messages: list[dict[str, object]],
        tools=None,
        output_model=None,
        cache_control_injection_points=None,
        **kwargs,
    ):
        captured_headers.append(kwargs["extra_headers"])
        return MagicMock()

    monkeypatch.setattr(nooa_model_client.CompletionClient, "acall", _recording_acall)

    with ctx:
        result = await resolve_model_clients(
            MagicMock(),
            ConfiguredModelRefs(default="default/gpt-4-1", fast="default/gpt-4-1"),
        )

    await result.default.acall([{"role": "user", "content": "hello"}], extra_headers={"x-call": "one"})
    await result.default.acall([{"role": "user", "content": "again"}])

    assert captured_headers == [
        {
            "x-test": "value",
            "accept-encoding": "identity",
            "x-call": "one",
            "Authorization": "Bearer token-1",
        },
        {
            "x-test": "value",
            "accept-encoding": "identity",
            "Authorization": "Bearer token-2",
        },
    ]
    assert token_provider.calls == 2


def test_active_model_clients_are_scoped():
    default = MagicMock()
    fast = MagicMock()
    refs = ConfiguredModelRefs(default="default/quality", fast="default/fast")
    pair = ConfiguredModelClients(default=default, fast=fast, refs=refs)

    with activate_model_clients(pair):
        assert get_default_model() is default
        assert get_fast_model() is fast
        assert get_configured_model_refs() == refs

    with pytest.raises(RuntimeError, match="were not activated"):
        get_default_model()


async def test_model_clients_close_each_distinct_client_once():
    default = MagicMock()
    default.aclose = AsyncMock()
    fast = MagicMock()
    fast.aclose = AsyncMock()

    await ConfiguredModelClients(default=default, fast=fast).aclose()

    default.aclose.assert_awaited_once()
    fast.aclose.assert_awaited_once()


async def test_model_clients_close_shared_client_once():
    shared = MagicMock()
    shared.aclose = AsyncMock()

    await ConfiguredModelClients(default=shared, fast=shared).aclose()

    shared.aclose.assert_awaited_once()


async def test_model_clients_close_fast_after_default_close_fails():
    default = MagicMock()
    default.aclose = AsyncMock(side_effect=RuntimeError("default close failed"))
    fast = MagicMock()
    fast.aclose = AsyncMock()

    with pytest.raises(RuntimeError, match="default close failed"):
        await ConfiguredModelClients(default=default, fast=fast).aclose()

    default.aclose.assert_awaited_once()
    fast.aclose.assert_awaited_once()


async def test_resolve_model_clients_closes_constructed_client_after_failure(monkeypatch):
    default_entity = _model_entity(
        workspace="default",
        name="quality",
        backend_format="OPENAI_CHAT",
        api_endpoint=SimpleNamespace(model_id="quality"),
    )

    async def _flaky_get_model(**kwargs):
        if kwargs.get("name") == "quality":
            return SimpleNamespace(data=lambda: default_entity)
        raise RuntimeError("fast resolution failed")

    ctx, client = _patch_models_client({})
    client.get_model = AsyncMock(side_effect=_flaky_get_model)
    constructed = MagicMock()
    constructed.aclose = AsyncMock()
    with ctx:
        monkeypatch.setattr(nooa_model_client, "_completion_client", MagicMock(return_value=constructed))

        with pytest.raises(RuntimeError, match="fast resolution failed"):
            await resolve_model_clients(
                MagicMock(),
                ConfiguredModelRefs(default="default/quality", fast="default/fast"),
            )

    constructed.aclose.assert_awaited_once()
