# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for execution-time Platform secret resolution."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from nemo_deployments_plugin.entities import Container, DeploymentConfig, EnvVar, SecretRef
from nemo_deployments_plugin.secrets import (
    SecretResolutionError,
    platform_ngc_secret_ref,
    resolve_deployment_config_secrets,
    resolve_deployment_secret_env,
)
from nemo_helix_plugin.client.errors import NemoClientError, NotFoundError


def _config(secret_ref: SecretRef, *, env_name: str = "NGC_API_KEY") -> DeploymentConfig:
    return DeploymentConfig(
        name="cfg",
        workspace="default",
        containers=[
            Container(
                name="server",
                image="nvcr.io/nim/test:latest",
                env=[EnvVar(name=env_name, secretRef=secret_ref)],
            )
        ],
    )


def _not_found() -> NotFoundError:
    request = httpx.Request("GET", "http://test/secrets")
    return NotFoundError(httpx.Response(404, request=request, text="missing"))


def test_env_var_rejects_plaintext_and_secret_reference_together() -> None:
    with pytest.raises(ValueError, match="only one"):
        EnvVar(
            name="NGC_API_KEY",
            value="must-not-be-stored",
            secretRef=SecretRef(workspace="system", name="ngc-api-key"),
        )


def test_platform_ngc_secret_ref_contains_reference_and_fallback_name_only() -> None:
    config = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )
    with patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=config):
        secret_ref = platform_ngc_secret_ref()

    assert secret_ref == SecretRef(
        workspace="system",
        name="ngc-api-key",
    )


@pytest.mark.asyncio
async def test_resolve_deployment_config_secrets_keeps_stored_config_reference_only() -> None:
    original = _config(SecretRef(workspace="system", name="ngc-api-key"))
    response = MagicMock()
    response.data.return_value = SimpleNamespace(value="resolved-value")
    secrets = AsyncMock()
    secrets.access_secret.return_value = response

    platform = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )
    with (
        patch("nemo_deployments_plugin.secrets.client_from_platform", return_value=secrets),
        patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=platform),
    ):
        resolved = await resolve_deployment_config_secrets(MagicMock(), original)

    assert original.containers[0].env[0].value is None
    assert original.containers[0].env[0].secret_ref == SecretRef(workspace="system", name="ngc-api-key")
    assert "resolved-value" not in original.model_dump_json(by_alias=True)
    assert resolved.containers[0].env[0].value == "resolved-value"
    assert resolved.containers[0].env[0].secret_ref is None


@pytest.mark.asyncio
async def test_resolve_deployment_config_secrets_uses_environment_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(SecretRef(workspace="system", name="ngc-api-key"))
    secrets = AsyncMock()
    secrets.access_secret.side_effect = _not_found()
    monkeypatch.setenv("NGC_API_KEY", "fallback-value")

    platform = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )
    with (
        patch("nemo_deployments_plugin.secrets.client_from_platform", return_value=secrets),
        patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=platform),
    ):
        resolved = await resolve_deployment_config_secrets(MagicMock(), config)

    assert resolved.containers[0].env[0].value == "fallback-value"


@pytest.mark.asyncio
async def test_resolve_deployment_config_secrets_omits_unresolved_ngc_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(SecretRef(workspace="system", name="ngc-api-key"))
    secrets = AsyncMock()
    secrets.access_secret.side_effect = _not_found()
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    platform = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )
    with (
        patch("nemo_deployments_plugin.secrets.client_from_platform", return_value=secrets),
        patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=platform),
    ):
        resolved = await resolve_deployment_config_secrets(MagicMock(), config)

    assert all(item.name != "NGC_API_KEY" for item in resolved.containers[0].env)
    assert config.containers[0].env[0].secret_ref == SecretRef(workspace="system", name="ngc-api-key")


@pytest.mark.asyncio
async def test_secret_access_error_does_not_fall_back_to_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(SecretRef(workspace="system", name="ngc-api-key"))
    secrets = AsyncMock()
    secrets.access_secret.side_effect = NemoClientError("access denied")
    monkeypatch.setenv("NGC_API_KEY", "must-not-be-used")
    platform = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )

    with (
        patch("nemo_deployments_plugin.secrets.client_from_platform", return_value=secrets),
        patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=platform),
        pytest.raises(SecretResolutionError, match="access failed"),
    ):
        await resolve_deployment_config_secrets(MagicMock(), config)


@pytest.mark.asyncio
async def test_resolve_deployment_config_secrets_resolves_arbitrary_reference() -> None:
    config = _config(
        SecretRef(workspace="default", name="app-token"),
        env_name="APP_TOKEN",
    )
    response = MagicMock()
    response.data.return_value = SimpleNamespace(value="app-token-value")
    secrets = AsyncMock()
    secrets.access_secret.return_value = response
    platform = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )

    with (
        patch("nemo_deployments_plugin.secrets.client_from_platform", return_value=secrets),
        patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=platform),
    ):
        resolved = await resolve_deployment_config_secrets(MagicMock(), config)

    assert resolved.containers[0].env[0].value == "app-token-value"
    assert resolved.containers[0].env[0].secret_ref is None
    # The stored config is never mutated with a plaintext value.
    assert config.containers[0].env[0].value is None


@pytest.mark.asyncio
async def test_resolve_deployment_config_secrets_missing_arbitrary_reference_raises() -> None:
    config = _config(
        SecretRef(workspace="default", name="app-token"),
        env_name="APP_TOKEN",
    )
    secrets = AsyncMock()
    secrets.access_secret.side_effect = _not_found()
    platform = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )

    with (
        patch("nemo_deployments_plugin.secrets.client_from_platform", return_value=secrets),
        patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=platform),
        pytest.raises(SecretResolutionError, match="could not be resolved"),
    ):
        await resolve_deployment_config_secrets(MagicMock(), config)


@pytest.mark.asyncio
async def test_resolve_deployment_secret_env_collects_values_across_containers() -> None:
    config = DeploymentConfig(
        name="cfg",
        workspace="default",
        initContainers=[
            Container(
                name="init",
                image="busybox",
                env=[EnvVar(name="INIT_TOKEN", secretRef=SecretRef(workspace="default", name="init-token"))],
            )
        ],
        containers=[
            Container(
                name="server",
                image="nvcr.io/nim/test:latest",
                env=[
                    EnvVar(name="APP_TOKEN", secretRef=SecretRef(workspace="default", name="app-token")),
                    EnvVar(name="PLAINTEXT", value="not-a-secret"),
                ],
            )
        ],
    )

    async def _access_secret(*, name: str, workspace: str) -> MagicMock:
        response = MagicMock()
        response.data.return_value = SimpleNamespace(value=f"{workspace}/{name}-value")
        return response

    secrets = AsyncMock()
    secrets.access_secret.side_effect = _access_secret
    platform = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )

    with (
        patch("nemo_deployments_plugin.secrets.client_from_platform", return_value=secrets),
        patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=platform),
    ):
        secret_env = await resolve_deployment_secret_env(MagicMock(), config)

    assert secret_env == {
        "INIT_TOKEN": "default/init-token-value",
        "APP_TOKEN": "default/app-token-value",
    }
    # The config's secret_ref env vars remain intact (mounted via envFrom, not plaintext).
    assert config.containers[0].env[0].secret_ref == SecretRef(workspace="default", name="app-token")


@pytest.mark.asyncio
async def test_resolve_deployment_secret_env_empty_when_no_secret_refs() -> None:
    config = DeploymentConfig(
        name="cfg",
        workspace="default",
        containers=[Container(name="server", image="busybox", env=[EnvVar(name="PLAIN", value="v")])],
    )
    secret_env = await resolve_deployment_secret_env(MagicMock(), config)
    assert secret_env == {}


@pytest.mark.asyncio
async def test_resolve_deployment_secret_env_omits_unresolved_ngc_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(SecretRef(workspace="system", name="ngc-api-key"))
    secrets = AsyncMock()
    secrets.access_secret.side_effect = _not_found()
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    platform = SimpleNamespace(
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )

    with (
        patch("nemo_deployments_plugin.secrets.client_from_platform", return_value=secrets),
        patch("nemo_deployments_plugin.secrets.get_platform_config", return_value=platform),
    ):
        secret_env = await resolve_deployment_secret_env(MagicMock(), config)

    assert secret_env == {}
