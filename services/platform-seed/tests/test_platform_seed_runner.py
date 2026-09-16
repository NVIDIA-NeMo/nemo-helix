# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for platform seed runner."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.models.types import CreateModelProviderRequest
from nhx.platform_seed.config import PlatformSeedConfig
from nhx.platform_seed.tasks.seed import run_platform_seed


@pytest.fixture
def entity_client():
    return AsyncMock()


@pytest.fixture
def sdk():
    return AsyncMock()


@pytest.fixture
def config_disabled():
    return PlatformSeedConfig(enabled=False)


@pytest.fixture
def config_enabled():
    """Returns a config with seeding enabled, but all specific seeds disabled.
    Tests should mutate this config object setting the specific seed under test to enabled.
    """
    return PlatformSeedConfig(
        enabled=True,
        auth_enabled=False,
        guardrails_enabled=False,
        model_provider_enabled=False,
        guardrails_config_store_path=Path("/tmp/config-store"),
    )


@pytest.mark.asyncio
async def test_run_platform_seed_disabled(config_disabled, entity_client, sdk):
    """When config.enabled is False, no seed steps run."""
    result = await run_platform_seed(entity_client, sdk, config_disabled)
    assert result.auth_ok is False
    assert result.guardrails_ok is False
    assert result.models_ok is False
    assert result.plugin_results == {}
    assert result.errors == []


@pytest.mark.asyncio
async def test_run_platform_seed_guardrails_only(config_enabled, entity_client, sdk):
    """Guardrails seed is called when guardrails_enabled is True."""
    config_enabled.guardrails_enabled = True

    with patch("nhx.guardrails.app.utils.config_store.populate_config_store", new_callable=AsyncMock) as mock_populate:
        result = await run_platform_seed(entity_client, sdk, config_enabled)
        mock_populate.assert_called_once_with(entity_client, config_enabled.guardrails_config_store_path)
        assert result.auth_ok is False
        assert result.guardrails_ok is True
        assert result.errors == []


@pytest.mark.asyncio
async def test_run_platform_seed_guardrails_failure(config_enabled, entity_client, sdk):
    """Guardrails seed failure is recorded and other steps still run."""
    config_enabled.guardrails_enabled = True

    with patch(
        "nhx.guardrails.app.utils.config_store.populate_config_store",
        new_callable=AsyncMock,
        side_effect=RuntimeError("bad"),
    ):
        result = await run_platform_seed(entity_client, sdk, config_enabled)

    assert result.guardrails_ok is False
    assert len(result.errors) == 1
    assert "Guardrails" in result.errors[0]


@pytest.mark.asyncio
async def test_run_platform_seed_plugin_seed_jobs(config_enabled, entity_client, sdk):
    """Discovered plugin seed jobs are run with injected sdk and entity client."""
    mock_seed_cls = MagicMock()
    mock_seed_instance = MagicMock()
    mock_seed_instance.run = AsyncMock()
    mock_seed_cls.return_value = mock_seed_instance

    with patch(
        "nhx.platform_seed.tasks.seed.run.discover_seed_jobs",
        return_value={"example": mock_seed_cls},
    ):
        result = await run_platform_seed(entity_client, sdk, config_enabled)

    mock_seed_instance.run.assert_awaited_once_with()
    assert mock_seed_instance.sdk is sdk
    assert mock_seed_instance.entities_client is entity_client
    assert result.plugin_results == {"example": True}


@pytest.mark.asyncio
async def test_run_platform_seed_plugin_seed_job_failure(config_enabled, entity_client, sdk):
    """Plugin seed failures are recorded without failing the whole run."""
    mock_seed_cls = MagicMock()
    mock_seed_instance = MagicMock()
    mock_seed_instance.run = AsyncMock(side_effect=RuntimeError("boom"))
    mock_seed_cls.return_value = mock_seed_instance

    with patch(
        "nhx.platform_seed.tasks.seed.run.discover_seed_jobs",
        return_value={"example": mock_seed_cls},
    ):
        result = await run_platform_seed(entity_client, sdk, config_enabled)

    assert result.plugin_results == {"example": False}
    assert any("Plugin seed job 'example' failed" in error for error in result.errors)


@pytest.mark.asyncio
async def test_run_platform_seed_plugin_seed_jobs_can_be_disabled_per_plugin(
    config_enabled, entity_client, sdk, monkeypatch
):
    """Only plugin seed jobs with enabled env toggles should run."""
    mock_example_seed_cls = MagicMock()
    mock_example_seed_instance = MagicMock()
    mock_example_seed_instance.run = AsyncMock()
    mock_example_seed_cls.return_value = mock_example_seed_instance

    mock_other_seed_cls = MagicMock()
    mock_other_seed_instance = MagicMock()
    mock_other_seed_instance.run = AsyncMock()
    mock_other_seed_cls.return_value = mock_other_seed_instance

    monkeypatch.setenv("NHX_PLATFORM_SEED_EXAMPLE_ENABLED", "false")

    with patch(
        "nhx.platform_seed.tasks.seed.run.discover_seed_jobs",
        return_value={"example": mock_example_seed_cls, "other-plugin": mock_other_seed_cls},
    ):
        result = await run_platform_seed(entity_client, sdk, config_enabled)

    mock_example_seed_instance.run.assert_not_called()
    mock_other_seed_instance.run.assert_awaited_once_with()
    assert result.plugin_results == {"other-plugin": True}
    assert result.errors == []


@pytest.mark.asyncio
async def test_run_platform_seed_auth_called(config_enabled, entity_client, sdk):
    """Auth seed is called when auth_enabled is True."""
    config_enabled.auth_enabled = True

    with patch("nhx.core.auth.app.seeding.run_seeding", new_callable=AsyncMock) as mock_run_seeding:
        mock_run_seeding.return_value = True
        result = await run_platform_seed(entity_client, sdk, config_enabled)
        mock_run_seeding.assert_called_once_with(entity_client)
        assert result.auth_ok is True
        assert result.errors == []


@pytest.mark.asyncio
async def test_run_platform_seed_model_provider_called(config_enabled, entity_client, sdk):
    """Model provider seed is called when model_provider_enabled is True."""
    config_enabled.model_provider_enabled = True
    models = MagicMock()
    models.create_provider = AsyncMock()

    with patch("nhx.platform_seed.tasks.seed.run.client_from_platform", return_value=models):
        result = await run_platform_seed(entity_client, sdk, config_enabled)

    models.create_provider.assert_awaited_once_with(
        workspace="system",
        body=CreateModelProviderRequest(
            name="nvidia-build",
            host_url="https://integrate.api.nvidia.com",
            api_key_secret_name="ngc-api-key",
        ),
    )
    assert result.models_ok is True
    assert result.errors == []


@pytest.mark.asyncio
async def test_run_platform_seed_model_provider_conflict(config_enabled, entity_client, sdk):
    """ConflictError when creating provider is handled gracefully (not a failure)."""
    config_enabled.model_provider_enabled = True
    models = MagicMock()
    response = httpx.Response(409, request=httpx.Request("POST", "http://models/providers"))
    models.create_provider = AsyncMock(side_effect=ConflictError(response))

    with patch("nhx.platform_seed.tasks.seed.run.client_from_platform", return_value=models):
        result = await run_platform_seed(entity_client, sdk, config_enabled)

    assert result.models_ok is True
    assert result.errors == []
