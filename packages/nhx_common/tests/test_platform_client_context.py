# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR
from nhx.common.config import HelixConfig
from nhx.common.platform_client_context import build_platform_client_context, build_platform_runtime_context
from nhx.common.service import DependencyProvider


def test_runtime_context_applies_explicit_base_url_before_building_runtime() -> None:
    platform_config = HelixConfig(
        base_url="https://platform.example.test",
        service_discovery={"entities": "http://entities.internal:8080"},
    )

    runtime_context = build_platform_runtime_context(
        platform_config=platform_config,
        base_url="https://override.example.test",
    )

    assert runtime_context.platform_config is platform_config
    assert runtime_context.base_url == "https://override.example.test"
    assert runtime_context.endpoint.connect_base_url == "https://override.example.test"
    assert (
        runtime_context.runtime.resolve_url("https://override.example.test/health/ready")
        == "https://override.example.test/health/ready"
    )
    assert (
        runtime_context.runtime.resolve_url("https://override.example.test/apis/entities/v2/workspaces/default")
        == "http://entities.internal:8080/apis/entities/v2/workspaces/default"
    )


def test_client_context_reuses_supplied_runtime_context() -> None:
    runtime_context = build_platform_runtime_context(
        platform_config=HelixConfig(base_url="https://platform.example.test"),
    )

    client_context = build_platform_client_context(runtime_context=runtime_context)

    assert client_context.runtime_context is runtime_context
    assert client_context.endpoint is runtime_context.endpoint
    assert client_context.runtime is runtime_context.runtime
    assert client_context.base_url == runtime_context.base_url


def test_client_context_workload_identity_auth_uses_context_base_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subject_token_file = tmp_path / "workload-token"
    subject_token_file.write_text("subject-token\n", encoding="utf-8")
    provider = object()
    monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(subject_token_file))

    client_context = build_platform_client_context(
        platform_config=HelixConfig(base_url="https://platform.example.test"),
    )

    with patch(
        "nemo_helix_plugin.client.oidc_factory.resolve_workload_exchange_provider",
        return_value=provider,
    ) as resolve_provider:
        assert client_context.nemo_client_auth() is provider

    resolve_provider.assert_called_once_with(
        base_url="https://platform.example.test",
        subject_token_file=subject_token_file,
    )


def test_dependency_provider_caches_runtime_context_for_platform_config() -> None:
    platform_config = HelixConfig(base_url="https://platform.example.test")

    with patch("nhx.common.service.base.get_platform_config", return_value=platform_config):
        provider = DependencyProvider()
        runtime_context = provider.get_runtime_context()

    assert provider.get_runtime_context() is runtime_context
    assert runtime_context.platform_config is platform_config
    assert runtime_context.endpoint.connect_base_url == "https://platform.example.test"
