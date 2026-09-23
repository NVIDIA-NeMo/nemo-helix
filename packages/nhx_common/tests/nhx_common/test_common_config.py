# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nhx.common.config module."""

import pytest
from nhx.common.config import (
    AuthConfig,
    CommonServiceConfig,
    Configuration,
    DatabaseConfig,
    HelixConfig,
    get_common_service_config,
    get_platform_config,
)
from nhx.common.platform_endpoint import UDS_BASE_URL


class TestHelixConfig:
    """Tests for HelixConfig."""

    def test_platform_config_defaults(self):
        """Test HelixConfig has sensible defaults."""
        config = HelixConfig()

        assert config.base_url == "http://localhost:8080"
        assert config.get_service_url("jobs") == "http://localhost:8080"
        assert config.get_service_url("files") == "http://localhost:8080"
        assert config.get_service_url("models") == "http://localhost:8080"
        assert config.get_service_url("secrets") == "http://localhost:8080"
        assert config.image_pull_secrets == []

    def test_platform_config_global_settings_key(self):
        """Test HelixConfig global_settings_key."""
        assert HelixConfig.global_settings_key() == "platform"

    def test_platform_config_to_shared_envvars(self):
        """Test to_shared_envvars returns expected dict."""
        config = HelixConfig()
        envvars = config.to_shared_envvars()

        assert "NHX_BASE_URL" in envvars
        assert "NHX_JOBS_URL" in envvars

    def test_service_url_env_var_populates_service_discovery(self, monkeypatch):
        """NHX_<SERVICE>_URL env vars are merged into service_discovery."""
        monkeypatch.setenv("NHX_FILES_URL", "http://files:8000")
        config = HelixConfig()

        assert config.get_service_url("files") == "http://files:8000"
        assert config.service_discovery["files"] == "http://files:8000"

    def test_service_url_env_var_overrides_config_file(self, monkeypatch):
        """Env var NHX_*_URL overrides or adds to file service_discovery."""
        monkeypatch.setenv("NHX_JOBS_URL", "http://jobs-from-env:9000")
        settings = {
            "platform": {
                "service_discovery": {"files": "http://files-from-file:8080"},
            },
        }
        config = Configuration.global_settings_to_service_config(settings, HelixConfig)

        assert config.get_service_url("files") == "http://files-from-file:8080"
        assert config.get_service_url("jobs") == "http://jobs-from-env:9000"
        assert config.service_discovery["files"] == "http://files-from-file:8080"
        assert config.service_discovery["jobs"] == "http://jobs-from-env:9000"

    def test_base_url_env_var_not_in_service_discovery(self, monkeypatch):
        """NHX_BASE_URL sets base_url and is not added to service_discovery."""
        monkeypatch.setenv("NHX_BASE_URL", "http://base-from-env:7000")
        config = HelixConfig()

        assert config.base_url == "http://base-from-env:7000"
        assert "base" not in config.service_discovery

    def test_get_services_parses_comma_separated_list(self):
        """get_services returns parsed list; empty or blank services gives empty list."""
        assert HelixConfig(services="").get_services() == []
        assert HelixConfig(services="auth,jobs,models").get_services() == [
            "auth",
            "jobs",
            "models",
        ]
        assert HelixConfig(services="  auth , jobs  , models  ").get_services() == [
            "auth",
            "jobs",
            "models",
        ]

    def test_is_service_local_matches_whole_names_only(self):
        """_is_service_local uses get_services() so substrings do not match (e.g. 'job' not in 'auth,jobs,models')."""
        config = HelixConfig(services="auth,jobs,models")
        assert config._is_service_local("jobs") is True
        assert config._is_service_local("auth") is True
        assert config._is_service_local("models") is True
        assert config._is_service_local("job") is False
        assert config._is_service_local("entities") is False

    def test_get_service_url_local_uses_common_service_config_host_port(self):
        """When service is local, get_service_url returns URL from CommonServiceConfig (get_host_url)."""
        config = HelixConfig(services="auth,jobs")
        url = config.get_service_url("auth")
        common = get_common_service_config()
        assert url == common.get_host_url()
        assert config.get_service_url("entities") == "http://localhost:8080"  # not local, base_url

    def test_get_service_url_local_overrides_gateway_base_url(self):
        """Local services bypass the gateway even when base_url points at the gateway."""
        config = HelixConfig(
            base_url="https://nemo-gateway:8080",
            services="auth,jobs",
            service_discovery={},
        )
        common = get_common_service_config()

        assert config.get_service_url("auth") == common.get_host_url()
        assert config.get_service_url("jobs") == common.get_host_url()
        assert config.get_service_url("files") == "https://nemo-gateway:8080"

    def test_get_service_url_local_overrides_service_discovery(self):
        """When a service is both local and in service_discovery, prefer local URL (CommonServiceConfig)."""
        config = HelixConfig(
            services="auth,jobs",
            service_discovery={
                "auth": "http://auth-remote:8080",
                "jobs": "http://jobs-remote:9000",
                "entities": "http://entities-remote:8080",
            },
        )
        common = get_common_service_config()
        # Local services: return CommonServiceConfig host URL, not service_discovery
        assert config.get_service_url("auth") == common.get_host_url()
        assert config.get_service_url("jobs") == common.get_host_url()
        # Not local but in service_discovery: use service_discovery
        assert config.get_service_url("entities") == "http://entities-remote:8080"

    def test_get_service_url_unknown_falls_back_to_base_url(self):
        """When service is not local and not in service_discovery, get_service_url returns base_url."""
        config = HelixConfig(
            base_url="http://platform:8080",
            services="auth,jobs",
            service_discovery={"entities": "http://entities-remote:8080"},
        )
        # Unknown service: not in services list, not in service_discovery
        assert config.get_service_url("models") == "http://platform:8080"
        assert config.get_service_url("files") == "http://platform:8080"
        # In service_discovery only: use service_discovery
        assert config.get_service_url("entities") == "http://entities-remote:8080"

    def test_create_service_pattern(self):
        """create_service_pattern returns a regex matching /apis/{service-name}/ (lowercase + dashes only).

        Mirrors and extends the docstring examples; pattern disallows uppercase, digits, and slashes.
        """
        config = HelixConfig()
        pattern = config.create_service_pattern()

        assert pattern is not None
        # Docstring examples: match and capture service name
        assert pattern.search("/apis/jobs/v2").group(1) == "jobs"
        assert pattern.search("/apis/my-service/v2").group(1) == "my-service"
        assert pattern.search("/apis/jobs/v2/workspaces/ws1/jobs/123").group(1) == "jobs"
        # Other valid names
        assert pattern.search("/apis/models/").group(1) == "models"
        assert pattern.search("http://host/apis/entities/v2/workspaces").group(1) == "entities"
        assert pattern.search("/apis/data-designer/v2/thing").group(1) == "data-designer"
        # Reject uppercase, digits, slash in name
        assert pattern.search("/apis/Jobs/") is None
        assert pattern.search("/apis/MyService/") is None
        assert pattern.search("/apis/my-Service/") is None
        assert pattern.search("/apis/jobs2/") is None
        assert pattern.search("/apis/svc1/") is None
        # Only first segment captured (no slash in service name)
        match = pattern.search("/apis/my/service/")
        assert match is not None
        assert match.group(1) == "my"


class TestAuthConfig:
    """Tests for shared auth configuration helpers."""

    def test_get_pdp_url_for_embedded_provider_normalizes_base_url(self):
        config = AuthConfig(policy_decision_point_base_url="http://localhost:8080/")

        assert config.get_pdp_url("allow") == "http://localhost:8080/apis/auth/v2/authz/allow"
        assert config.auth_url == "http://localhost:8080/apis/auth/v2/authz/allow"

    def test_get_pdp_url_for_opa_provider_normalizes_base_url(self):
        config = AuthConfig(
            policy_decision_point_base_url="http://opa:8181/",
            policy_decision_point_provider="opa",
        )

        assert config.get_pdp_url("has_permissions") == "http://opa:8181/v1/data/authz/has_permissions"

    def test_get_pdp_url_for_uds_endpoint_uses_connect_base_url(self):
        config = AuthConfig(policy_decision_point_base_url="unix:///tmp/nemo-helix.sock")

        assert config.get_pdp_url("allow") == f"{UDS_BASE_URL}/apis/auth/v2/authz/allow"


class TestCommonServiceConfig:
    """Tests for CommonServiceConfig."""

    def test_common_service_config_defaults(self):
        """Test CommonServiceConfig has sensible defaults."""
        config = CommonServiceConfig()

        assert config.log_level == "INFO"
        assert config.log_format == "plain"
        assert config.host == "127.0.0.1"
        assert config.port == 8080

    def test_common_service_config_global_settings_key(self):
        """Test CommonServiceConfig global_settings_key."""
        assert CommonServiceConfig.global_settings_key() == "service"

    def test_common_service_config_log_level_from_env(self, monkeypatch):
        """Test LOG_LEVEL env var configures log_level."""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        config = CommonServiceConfig()

        assert config.log_level == "DEBUG"

    def test_common_service_config_log_level_values(self, monkeypatch):
        """Test LOG_LEVEL accepts all valid values."""
        for level in ("DEBUG", "INFO", "WARN", "ERROR"):
            monkeypatch.setenv("LOG_LEVEL", level)
            config = CommonServiceConfig()
            assert config.log_level == level

    def test_common_service_config_log_format_from_env(self, monkeypatch):
        """Test LOG_FORMAT env var configures log_format."""
        monkeypatch.setenv("LOG_FORMAT", "json")
        config = CommonServiceConfig()

        assert config.log_format == "json"

    def test_common_service_config_log_format_values(self, monkeypatch):
        """Test LOG_FORMAT accepts all valid values."""
        for fmt in ("json", "plain"):
            monkeypatch.setenv("LOG_FORMAT", fmt)
            config = CommonServiceConfig()
            assert config.log_format == fmt


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_database_config_defaults(self):
        """Test DatabaseConfig has sensible defaults."""
        config = DatabaseConfig()

        assert config.dialect == "postgresql"
        assert config.host == ""
        assert config.path == ""
        assert config.name == ""
        assert config.port is None
        assert config.user is None
        assert config.password is None
        assert config.connections_limit == 10
        assert config.echo is False

    def test_database_config_sqlite_url(self):
        """Test SQLite database URL generation."""
        config = DatabaseConfig(dialect="sqlite", path="/tmp/test.db")
        url = config.sqlalchemy_database_url()

        assert "sqlite" in url
        assert "/tmp/test.db" in url

    def test_database_config_sqlite_requires_path(self):
        """Test SQLite config with explicit host but no path raises error."""
        # When connection params ARE set but path is missing, it should error
        config = DatabaseConfig(dialect="sqlite", host="localhost")

        with pytest.raises(ValueError, match="requires 'path'"):
            config.sqlalchemy_database_url()

    def test_database_config_postgres_url(self):
        """Test PostgreSQL database URL generation."""
        config = DatabaseConfig(
            dialect="postgresql",
            host="localhost",
            name="testdb",
            user="user",
            password="pass",
            port=5432,
        )
        url = config.sqlalchemy_database_url()

        assert "postgresql" in url
        assert "localhost" in url
        assert "testdb" in url

    def test_database_config_postgres_requires_name(self):
        """Test PostgreSQL config requires name."""
        config = DatabaseConfig(dialect="postgresql", host="localhost")

        with pytest.raises(ValueError, match="requires 'name'"):
            config.sqlalchemy_database_url()


class TestConfiguration:
    """Tests for Configuration class."""

    def test_get_platform_config(self):
        """Test Configuration.get_platform_config returns HelixConfig."""
        config = Configuration.get_platform_config()

        assert isinstance(config, HelixConfig)

    def test_global_settings_to_service_config_missing_key(self):
        """Test global_settings_to_service_config with missing key uses defaults."""
        config = Configuration.global_settings_to_service_config({}, HelixConfig)

        assert isinstance(config, HelixConfig)
        # Should use defaults
        assert config.base_url == "http://localhost:8080"

    def test_global_settings_to_service_config_with_values(self):
        """Test global_settings_to_service_config with provided values."""
        settings = {"platform": {"base_url": "http://custom:9000"}}
        config = Configuration.global_settings_to_service_config(settings, HelixConfig)

        assert config.base_url == "http://custom:9000"

    def test_global_settings_to_service_config_partial_overlay(self):
        """Test that only specified keys are overridden; unspecified keys keep code defaults."""
        settings = {"platform": {"base_url": "http://custom:9000"}}
        config = Configuration.global_settings_to_service_config(settings, HelixConfig)

        assert config.base_url == "http://custom:9000"
        # Unspecified keys keep defaults; get_service_url falls back to base_url
        assert config.get_service_url("jobs") == "http://custom:9000"
        assert config.get_service_url("models") == "http://custom:9000"
        assert config.runtime.value == "docker"

    def test_global_settings_to_service_config_nested_merge(self):
        """Test that only specified nested keys override; sibling keys keep defaults."""
        settings = {"platform": {"docker": {"reserved_gpu_device_ids": "none"}}}
        config = Configuration.global_settings_to_service_config(settings, HelixConfig)

        assert config.base_url == "http://localhost:8080"
        assert config.docker.reserved_gpu_device_ids == "none"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_platform_config_function(self):
        """Test get_platform_config convenience function."""
        config = get_platform_config()

        assert isinstance(config, HelixConfig)

    def test_get_common_service_config_function(self):
        """Test get_common_service_config convenience function."""
        config = get_common_service_config()

        assert isinstance(config, CommonServiceConfig)
