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
and lets ``NemoClient.to_async`` build the async sibling, and that rebuilt client
drops the in-process test transport — the same harness limitation that keeps
``test_validate_sdk`` off the sync resource.
"""

from __future__ import annotations

import logging

import data_designer.config as dd
import data_designer.interface.data_designer as data_designer_interface
import nemo_data_designer_plugin.testing.utils as u
import pytest
from data_designer.engine.models.errors import ModelAuthenticationError
from data_designer.interface.data_designer import DataDesigner
from data_designer_nemo.fileset_file_seed_source import FilesetFileSeedSource
from nemo_data_designer_plugin.sdk.check_models import check_models_config
from nemo_data_designer_plugin.sdk.resources import AsyncDataDesignerResource
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("mock_providers")]

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
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    async_data_designer: AsyncDataDesignerResource,
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

    report = await async_data_designer.check_models(_builder())

    assert report.ok is True
    assert "model alias 'text'" in capsys.readouterr().err


async def test_check_models_returns_ok_report(
    monkeypatch: pytest.MonkeyPatch, client: NemoClient, async_client: AsyncNemoClient
) -> None:
    _patch_probe(monkeypatch, None)

    report = await check_models_config(
        _builder(),
        client=client,
        async_client=async_client,
        workspace=client.workspace or u.WORKSPACE_NAME,
    )

    assert report.ok is True
    assert report.errors == []


@pytest.mark.parametrize(
    ("outcome", "expected_type"),
    [
        (ModelAuthenticationError("bad key"), "ModelAuthenticationError"),
        # TimeoutError is a builtin, not a DataDesignerError, so it only lands in
        # a report because _ENGINE_ERRORS lists it separately. Upstream raises it
        # when the 180s health-check budget is exhausted.
        (TimeoutError("bad key"), "TimeoutError"),
    ],
    ids=["typed-model-error", "health-check-timeout"],
)
async def test_check_models_reports_engine_error(
    monkeypatch: pytest.MonkeyPatch,
    outcome: Exception,
    expected_type: str,
    client: NemoClient,
    async_client: AsyncNemoClient,
) -> None:
    _patch_probe(monkeypatch, outcome)

    report = await check_models_config(
        _builder(),
        client=client,
        async_client=async_client,
        workspace=client.workspace or u.WORKSPACE_NAME,
    )

    assert report.ok is False
    assert [(e.error_type, e.message) for e in report.errors] == [(expected_type, "bad key")]


async def test_check_models_reports_resolution_failure_without_probing(
    monkeypatch: pytest.MonkeyPatch, client: NemoClient, async_client: AsyncNemoClient
) -> None:
    calls = _patch_probe(monkeypatch, None)

    report = await check_models_config(
        _builder(provider="default/does-not-exist"),
        client=client,
        async_client=async_client,
        workspace=client.workspace or u.WORKSPACE_NAME,
    )

    assert report.ok is False
    assert calls == []
    assert "does-not-exist" in report.errors[0].message


async def test_check_models_probes_without_a_sync_sdk(
    monkeypatch: pytest.MonkeyPatch, async_data_designer: AsyncDataDesignerResource
) -> None:
    """An async-only caller cannot build a sync SDK — rebuilding one from an
    async client is deliberately unsupported (only ``NemoClient.to_async`` exists).
    The probe still runs, against a probe-only engine context, so the async
    resource has real parity with the sync one rather than a silent skip.
    """
    calls = _patch_probe(monkeypatch, None)

    report = await async_data_designer.check_models(_builder())

    assert calls == [True]
    assert report.ok is True


async def test_check_models_without_sync_sdk_surfaces_model_errors(
    monkeypatch: pytest.MonkeyPatch, async_data_designer: AsyncDataDesignerResource
) -> None:
    _patch_probe(monkeypatch, ModelAuthenticationError("bad key"))

    report = await async_data_designer.check_models(_builder())

    assert report.ok is False
    assert [(e.error_type, e.message) for e in report.errors] == [("ModelAuthenticationError", "bad key")]


def _patch_readiness(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Patch only the network probe, leaving resource-provider construction real.

    ``_patch_probe`` replaces ``DataDesigner.check_models`` wholesale, which
    also skips ``_create_resource_provider`` — and with it the eager seed-reader
    lookup that the probe-only context exists to satisfy. Patching one level
    down keeps that construction in the test's path.
    """
    calls: list[bool] = []

    def _readiness(*args, **kwargs) -> None:
        calls.append(True)

    monkeypatch.setattr(data_designer_interface, "run_readiness_check", _readiness)
    return calls


@pytest.mark.usefixtures("mock_file")
async def test_check_models_probes_a_seeded_config_without_a_sync_sdk(
    monkeypatch: pytest.MonkeyPatch, async_data_designer: AsyncDataDesignerResource
) -> None:
    """Resource-provider construction eagerly looks up a reader for the
    configured seed type, so a seeded config is the case a probe-only context
    could plausibly break on. The seed is never read — only registered.

    Patches the readiness probe rather than ``check_models`` so the real
    ``_create_resource_provider`` runs: that lookup is the whole reason
    ``CheckModelsSeedReader`` exists, and patching a level up would skip it.
    """
    calls = _patch_readiness(monkeypatch)

    builder = _builder()
    # FilesetFileSeedSource is registered by the plugin, so it is not in the
    # library's static seed-source union.
    builder.with_seed_dataset(FilesetFileSeedSource(path=u.FILESET_FILE_SEED_SOURCE_PATH))  # ty: ignore[invalid-argument-type]

    report = await async_data_designer.check_models(builder)

    assert calls == [True]
    assert report.ok is True, [e.message for e in report.errors]


async def test_check_models_restores_the_library_logger_level(
    monkeypatch: pytest.MonkeyPatch, async_data_designer: AsyncDataDesignerResource
) -> None:
    """A check_models call that attaches the fallback library handler must not
    leave the ``data_designer`` logger pinned at INFO afterward — the level is
    saved and restored, mirroring ``_engine_logs.forward_engine_logs``.
    """
    lib_logger = logging.getLogger("data_designer")
    # A bare caller: no root or library handler configured, so the fallback
    # stream handler attaches, as it would in a real CLI process.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.setattr(lib_logger, "handlers", [])
    monkeypatch.setattr(lib_logger, "level", logging.WARNING)

    _patch_probe(monkeypatch, None, emit_log=True)

    report = await async_data_designer.check_models(_builder())

    assert report.ok is True
    # The level we set before the call must survive it (not be left at INFO).
    assert lib_logger.level == logging.WARNING
