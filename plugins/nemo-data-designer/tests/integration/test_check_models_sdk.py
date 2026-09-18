# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the SDK model health check.

Covers :func:`nemo_data_designer_plugin.sdk.check_models.check_models_config`
and ``DataDesignerResource.check_models`` against an in-process mock platform.

The probe is patched throughout, for the reason documented in
``test_check_models_cli``: the engine's HTTP client does not carry the tests'
ASGI transport, so no real generation can complete here.

These exercise ``check_models_config`` with both SDKs supplied, which is what
the CLI does. ``DataDesignerResource.check_models`` passes only the sync SDK
and lets ``sync_to_async_sdk`` build the async sibling, and that rebuilt SDK
drops the in-process test transport — the same harness limitation that keeps
``test_validate_sdk`` off the sync resource.
"""

from __future__ import annotations

import logging

import data_designer.config as dd
import nemo_data_designer_plugin.testing.utils as u
import pytest
from data_designer.engine.models.errors import ModelAuthenticationError
from data_designer.interface.data_designer import DataDesigner
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource
from nemo_data_designer_plugin.sdk.check_models import check_models_config
from nemo_data_designer_plugin.sdk.resources import AsyncDataDesignerResource

pytestmark = pytest.mark.integration

# Shape mirrors the real engine log that names the alias being probed.
_PROBE_LOG = "👀 Checking 'nano-v3' in provider named 'p' for model alias 'text'..."


def _builder(provider: str = u.OPEN_PROVIDER_NAME) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[dd.ModelConfig(alias="text", model=u.ENABLED_MODEL_NAME, provider=provider)]
    )
    builder.add_column(dd.LLMTextColumnConfig(name="x", prompt="hi", model_alias="text"))
    return builder


def _patch_probe(
    monkeypatch: pytest.MonkeyPatch,
    outcome: Exception | None,
    *,
    emit_log: bool = False,
) -> list[bool]:
    calls: list[bool] = []

    def _probe(*args, **kwargs) -> None:
        calls.append(True)
        if emit_log:
            logging.getLogger("data_designer.engine.models.registry").info(_PROBE_LOG)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(DataDesigner, "check_models", _probe)
    return calls


async def test_check_models_surfaces_engine_logs_without_caller_setup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The engine names each alias as it probes it, and that is the only place
    the alias appears — the error it raises on failure does not carry it. An SDK
    caller should get those lines from the resource's own logging setup, without
    having to configure logging themselves.
    """
    _patch_probe(monkeypatch, None, emit_log=True)
    # pytest attaches its own root handler, which would make the resource think
    # the caller already configured logging. Clear it to model a bare caller.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        report = await AsyncDataDesignerResource(client_context.async_sdk).check_models(_builder())

    assert report.ok is True
    assert "model alias 'text'" in capsys.readouterr().err


async def test_check_models_returns_ok_report(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch, None)

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        report = await check_models_config(
            _builder(),
            sdk=client_context.sdk,
            async_sdk=client_context.async_sdk,
            workspace=client_context.sdk.workspace or u.WORKSPACE_NAME,
        )

    assert report.ok is True
    assert report.errors == []


async def test_check_models_reports_typed_model_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch, ModelAuthenticationError("bad key"))

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        report = await check_models_config(
            _builder(),
            sdk=client_context.sdk,
            async_sdk=client_context.async_sdk,
            workspace=client_context.sdk.workspace or u.WORKSPACE_NAME,
        )

    assert report.ok is False
    assert [(e.error_type, e.message) for e in report.errors] == [("ModelAuthenticationError", "bad key")]


async def test_check_models_reports_resolution_failure_without_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_probe(monkeypatch, None)

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        report = await check_models_config(
            _builder(provider="default/does-not-exist"),
            sdk=client_context.sdk,
            async_sdk=client_context.async_sdk,
            workspace=client_context.sdk.workspace or u.WORKSPACE_NAME,
        )

    assert report.ok is False
    assert calls == []
    assert "does-not-exist" in report.errors[0].message


async def test_check_models_probes_without_a_sync_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """An async-only caller cannot build a sync SDK — rebuilding one from an
    async SDK is deliberately unsupported (see ``data_designer_nemo.sdk_translation``).
    The probe still runs, against a probe-only engine context, so the async
    resource has real parity with the sync one rather than a silent skip.
    """
    calls = _patch_probe(monkeypatch, None)

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        report = await AsyncDataDesignerResource(client_context.async_sdk).check_models(_builder())

    assert calls == [True]
    assert report.ok is True


async def test_check_models_without_sync_sdk_surfaces_model_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_probe(monkeypatch, ModelAuthenticationError("bad key"))

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
    ):
        report = await AsyncDataDesignerResource(client_context.async_sdk).check_models(_builder())

    assert report.ok is False
    assert [(e.error_type, e.message) for e in report.errors] == [("ModelAuthenticationError", "bad key")]


async def test_check_models_probes_a_seeded_config_without_a_sync_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resource-provider construction eagerly looks up a reader for the
    configured seed type, so a seeded config is the case a probe-only context
    could plausibly break on. The seed is never read — only registered.
    """
    calls = _patch_probe(monkeypatch, None)

    builder = _builder()
    # FilesetFileSeedSource is registered by the plugin, so it is not in the
    # library's static seed-source union.
    builder.with_seed_dataset(FilesetFileSeedSource(path=u.FILESET_FILE_SEED_SOURCE_PATH))  # ty: ignore[invalid-argument-type]

    with (
        u.make_mock_client_context() as client_context,
        u.setup_mock_providers(client_context),
        u.setup_mock_file(client_context),
    ):
        report = await AsyncDataDesignerResource(client_context.async_sdk).check_models(builder)

    assert calls == [True]
    assert report.ok is True, [e.message for e in report.errors]
