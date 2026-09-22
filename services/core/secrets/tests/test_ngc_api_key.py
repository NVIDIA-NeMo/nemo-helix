# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NGC API key secret helpers."""

import pytest
from nhx.common.config import HelixConfig
from nhx.core.secrets.api.v2.secrets.ngc_api_key import (
    get_default_ngc_api_key,
    is_default_ngc_api_key,
)
from nhx.core.secrets.entities import HelixSecret


@pytest.fixture
def platform_config() -> HelixConfig:
    """Default HelixConfig for NGC API key secret (system workspace, ngc-api-key name)."""
    return HelixConfig(  # type: ignore[abstract]
        ngc_api_key_secret="system/ngc-api-key",
        ngc_api_key_env_var="NGC_API_KEY",
    )


class TestIsDefaultNgcApiKey:
    """Tests for is_default_ngc_api_key."""

    def test_returns_true_when_workspace_and_name_match_config(self, platform_config: HelixConfig) -> None:
        assert is_default_ngc_api_key(platform_config, "system", "ngc-api-key") is True

    def test_returns_false_when_workspace_mismatch(self, platform_config: HelixConfig) -> None:
        assert is_default_ngc_api_key(platform_config, "default", "ngc-api-key") is False

    def test_returns_false_when_name_mismatch(self, platform_config: HelixConfig) -> None:
        assert is_default_ngc_api_key(platform_config, "system", "other-secret") is False

    def test_returns_false_when_both_mismatch(self, platform_config: HelixConfig) -> None:
        assert is_default_ngc_api_key(platform_config, "default", "other-secret") is False


class TestGetDefaultNgcApiKey:
    """Tests for get_default_ngc_api_key."""

    def test_returns_secret_with_env_value(
        self, platform_config: HelixConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NGC_API_KEY", "test-ngc-key-value")
        secret = get_default_ngc_api_key(platform_config)
        assert isinstance(secret, HelixSecret)
        assert secret.name == "ngc-api-key"
        assert secret.workspace == "system"
        assert secret.description == "Default NGC API key secret for the platform"
        assert secret._data == "test-ngc-key-value"

    def test_raises_value_error_when_env_var_unset(
        self, platform_config: HelixConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="NGC API key not found in environment variable NGC_API_KEY"):
            get_default_ngc_api_key(platform_config)

    def test_uses_config_env_var_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUSTOM_NGC_KEY", "custom-key")
        config = HelixConfig(  # type: ignore[abstract]
            ngc_api_key_secret="system/ngc-api-key",
            ngc_api_key_env_var="CUSTOM_NGC_KEY",
        )
        secret = get_default_ngc_api_key(config)
        assert secret._data == "custom-key"
