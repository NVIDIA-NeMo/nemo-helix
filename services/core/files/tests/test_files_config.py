# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FilesConfig (allowed_external_hosts and related)."""

import pytest
from nmp.common.config import Configuration
from nmp.core.files.config import FilesConfig


class TestFilesConfigAllowedExternalHosts:
    """Tests for FilesConfig.allowed_external_hosts and get_allowed_external_hosts()."""

    def test_allowed_external_hosts_default(self) -> None:
        """Test allowed_external_hosts default value and get_allowed_external_hosts() parsing."""
        config = FilesConfig()
        assert config.allowed_external_hosts == (
            "https://api.ngc.nvidia.com,https://huggingface.co,https://api.github.com"
        )
        assert config.get_allowed_external_hosts() == [
            "https://api.ngc.nvidia.com",
            "https://huggingface.co",
            "https://api.github.com",
        ]

    def test_allowed_external_hosts_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test NMP_FILES_ALLOWED_EXTERNAL_HOSTS env var sets allowed_external_hosts."""
        monkeypatch.setenv(
            "NMP_FILES_ALLOWED_EXTERNAL_HOSTS",
            "https://proxy.corp.example.com,https://api.ngc.nvidia.com",
        )
        config = FilesConfig()
        assert config.allowed_external_hosts == ("https://proxy.corp.example.com,https://api.ngc.nvidia.com")
        assert config.get_allowed_external_hosts() == [
            "https://proxy.corp.example.com",
            "https://api.ngc.nvidia.com",
        ]

    def test_allowed_external_hosts_from_yaml(self) -> None:
        """Test files.allowed_external_hosts loaded from YAML via global_settings_to_service_config."""
        settings = {
            "files": {
                "allowed_external_hosts": "https://hf.example.com,https://ngc.example.com",
            },
        }
        config = Configuration.global_settings_to_service_config(settings, FilesConfig)
        assert config.allowed_external_hosts == "https://hf.example.com,https://ngc.example.com"
        assert config.get_allowed_external_hosts() == [
            "https://hf.example.com",
            "https://ngc.example.com",
        ]

    def test_get_allowed_external_hosts_strips_whitespace(self) -> None:
        """Test get_allowed_external_hosts() strips whitespace and skips empty segments."""
        config = FilesConfig(allowed_external_hosts="  https://a.com , https://b.com  , , https://c.com  ")
        assert config.get_allowed_external_hosts() == [
            "https://a.com",
            "https://b.com",
            "https://c.com",
        ]

    def test_allowed_external_hosts_invalid_url_raises(self) -> None:
        """Test that invalid URL in allowed_external_hosts raises during validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="IPv4 or IPv6"):
            FilesConfig(allowed_external_hosts="https://valid.com,http://[zzz]")

    def test_allowed_external_hosts_missing_netloc_raises(self) -> None:
        """Test that URL without scheme/netloc raises ValueError from model validator."""
        with pytest.raises(ValueError, match="must be a valid URL"):
            FilesConfig(allowed_external_hosts="https://valid.com,not-a-url")


class TestFilesConfigHuggingFaceRetries:
    """Tests for Hugging Face retry configuration."""

    def test_hf_retry_defaults(self) -> None:
        """Test Hugging Face retry default values."""
        config = FilesConfig()
        assert config.hf_retry_attempts == 4
        assert config.hf_retry_initial_delay_seconds == 0.5
        assert config.hf_retry_max_delay_seconds == 5.0

    def test_hf_retry_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test NMP_FILES_HF_RETRY_* env vars set Hugging Face retry config."""
        monkeypatch.setenv("NMP_FILES_HF_RETRY_ATTEMPTS", "7")
        monkeypatch.setenv("NMP_FILES_HF_RETRY_INITIAL_DELAY_SECONDS", "1")
        monkeypatch.setenv("NMP_FILES_HF_RETRY_MAX_DELAY_SECONDS", "30")

        config = FilesConfig()

        assert config.hf_retry_attempts == 7
        assert config.hf_retry_initial_delay_seconds == 1.0
        assert config.hf_retry_max_delay_seconds == 30.0

    def test_hf_retry_from_yaml(self) -> None:
        """Test files.hf_retry_* values load from YAML settings."""
        settings = {
            "files": {
                "hf_retry_attempts": 6,
                "hf_retry_initial_delay_seconds": 2.5,
                "hf_retry_max_delay_seconds": 20,
            },
        }

        config = Configuration.global_settings_to_service_config(settings, FilesConfig)

        assert config.hf_retry_attempts == 6
        assert config.hf_retry_initial_delay_seconds == 2.5
        assert config.hf_retry_max_delay_seconds == 20.0

    def test_hf_retry_invalid_values_raise(self) -> None:
        """Test invalid Hugging Face retry values fail validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            FilesConfig(hf_retry_attempts=0)

        with pytest.raises(ValidationError):
            FilesConfig(hf_retry_initial_delay_seconds=-1)

        with pytest.raises(ValidationError):
            FilesConfig(hf_retry_max_delay_seconds=-1)
