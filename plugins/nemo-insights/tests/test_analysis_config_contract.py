# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for carrying scheduled-analysis models into Platform state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_insights_plugin import cli
from nemo_insights_plugin.entities import AnalysisConfig, AnalysisConfigStatus
from nemo_insights_plugin.sdk_resources.analysis_configs import (
    _build_enable_body,
    _build_status_update_body,
    _build_update_body,
)
from nemo_insights_plugin.service import InsightsService
from nemo_platform_plugin.entity_client import NemoEntityNotFoundError, get_entity_client


def _app(entity_client: AsyncMock) -> FastAPI:
    app = FastAPI()
    for spec in InsightsService().get_routers():
        app.include_router(spec.router, prefix=spec.prefix)
    app.dependency_overrides[get_entity_client] = lambda: entity_client
    return app


def test_enable_request_requires_and_persists_model_pair() -> None:
    entity_client = AsyncMock()
    entity_client.get.side_effect = NemoEntityNotFoundError("missing")
    entity_client.create.side_effect = lambda config: config

    response = TestClient(_app(entity_client)).post(
        "/v2/workspaces/default/analysis-configs/calculator-agent/enable",
        json={
            "default_model": "default/gpt-5",
            "fast_model": "default/gpt-5-mini",
        },
    )

    assert response.status_code == 200
    created = entity_client.create.await_args.args[0]
    assert created.default_model == "default/gpt-5"
    assert created.fast_model == "default/gpt-5-mini"
    assert response.json()["default_model"] == "default/gpt-5"
    assert response.json()["fast_model"] == "default/gpt-5-mini"


def test_reenable_refreshes_persisted_model_pair() -> None:
    entity_client = AsyncMock()
    existing = AnalysisConfig(
        name="calculator-agent",
        workspace="default",
        agent="calculator-agent",
        enabled=False,
        default_model="default/old-model",
        fast_model="default/old-model",
    )
    entity_client.get.return_value = existing
    entity_client.update.side_effect = lambda config: config

    response = TestClient(_app(entity_client)).post(
        "/v2/workspaces/default/analysis-configs/calculator-agent/enable",
        json={
            "default_model": "default/gpt-5",
            "fast_model": "default/gpt-5-mini",
        },
    )

    assert response.status_code == 200
    updated = entity_client.update.await_args.args[0]
    assert updated.enabled is True
    assert updated.default_model == "default/gpt-5"
    assert updated.fast_model == "default/gpt-5-mini"


def test_enable_sdk_body_contains_both_model_refs() -> None:
    assert _build_enable_body(
        default_model="default/gpt-5",
        fast_model="default/gpt-5-mini",
    ).model_dump(mode="json") == {
        "default_model": "default/gpt-5",
        "fast_model": "default/gpt-5-mini",
    }


def test_update_config_sdk_body_omits_none_enabled() -> None:
    assert _build_update_body(enabled=None).model_dump(mode="json", exclude_unset=True) == {}
    assert _build_update_body(enabled=False).model_dump(mode="json", exclude_unset=True) == {"enabled": False}


def test_update_run_status_sdk_body_omits_unset_fields_and_keeps_explicit_null_timestamps() -> None:
    empty = _build_status_update_body()
    partial = _build_status_update_body(
        status=AnalysisConfigStatus.RUNNING,
        last_submitted_job="job-123",
        last_error="",
    )
    clear_timestamps = _build_status_update_body(
        last_successful_run_at=None,
        last_attempted_at=None,
        last_completed_at=None,
    )

    assert empty.model_dump(mode="json", exclude_unset=True) == {}
    assert partial.model_dump(mode="json", exclude_unset=True) == {
        "status": "running",
        "last_submitted_job": "job-123",
        "last_error": "",
    }
    assert clear_timestamps.model_dump(mode="json", exclude_unset=True) == {
        "last_successful_run_at": None,
        "last_attempted_at": None,
        "last_completed_at": None,
    }


@pytest.mark.asyncio
async def test_enable_cli_sends_locally_configured_models(monkeypatch: pytest.MonkeyPatch) -> None:
    analysis_configs = SimpleNamespace(
        enable=AsyncMock(
            return_value=AnalysisConfig(
                name="calculator-agent",
                workspace="default",
                agent="calculator-agent",
                default_model="default/gpt-5",
                fast_model="default/gpt-5-mini",
            )
        )
    )
    client = SimpleNamespace(
        insights=SimpleNamespace(analysis_configs=analysis_configs),
        close=AsyncMock(),
    )
    monkeypatch.setattr(cli, "make_client", lambda base_url: client)
    monkeypatch.setattr(
        cli,
        "configured_model_refs",
        lambda: SimpleNamespace(default="default/gpt-5", fast="default/gpt-5-mini"),
    )

    await cli._analysis_config_command(
        action="enable",
        agent="calculator-agent",
        workspace="default",
        base_url="http://localhost:8080",
    )

    analysis_configs.enable.assert_awaited_once_with(
        workspace="default",
        agent="calculator-agent",
        default_model="default/gpt-5",
        fast_model="default/gpt-5-mini",
    )
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_enable_cli_checks_local_models_before_constructing_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_models():
        raise ValueError("No default model is configured. Run `nemo setup` and select agent models.")

    make_client = AsyncMock()
    monkeypatch.setattr(cli, "configured_model_refs", missing_models)
    monkeypatch.setattr(cli, "make_client", make_client)

    with pytest.raises(ValueError, match="No default model is configured"):
        await cli._analysis_config_command(
            action="enable",
            agent="calculator-agent",
            workspace="default",
            base_url="http://localhost:8080",
        )

    make_client.assert_not_called()
