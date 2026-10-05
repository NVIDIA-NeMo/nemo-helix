# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the config module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from nemo_helix_ext.config.config import (
    Config,
    ConfigParams,
    get_context,
)
from nemo_helix_ext.config.models import DEFAULT_BASE_URL, ConfigFile, NoAuthUser, OAuthUser
from nemo_helix_plugin.client.constants import WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file for testing."""
    config_data = {
        "current_context": "production",
        "clusters": [
            {
                "name": "prod-cluster",
                "base_url": "https://api.prod.example.com",
            },
            {
                "name": "dev-cluster",
                "base_url": "https://api.dev.example.com",
            },
            {
                "name": "local-cluster",
                "base_url": "http://localhost:8000",
                "certificate_authority": "/tmp/local-ca.pem",
            },
        ],
        "users": [
            {
                "name": "prod-admin",
                "type": "api-key",
                "api_key": "prod-key-123",
            },
            {
                "name": "dev-admin",
                "type": "api-key",
                "api_key": "dev-key-456",
            },
            {
                "name": "local-user",
                "type": "no-auth",
            },
        ],
        "contexts": [
            {
                "name": "production",
                "cluster": "prod-cluster",
                "user": "prod-admin",
                "workspace": "prod-workspace",
                "preferences": {
                    "output_format": "yaml",
                    "timestamp_format": "iso8601",
                    "color_output": True,
                },
            },
            {
                "name": "development",
                "cluster": "dev-cluster",
                "user": "dev-admin",
                "workspace": "dev-workspace",
                "preferences": {
                    "output_format": "json",
                    "timestamp_format": "iso8601",
                    "color_output": False,
                },
            },
            {
                "name": "local",
                "cluster": "local-cluster",
                "user": "local-user",
                "workspace": "local-workspace",
                "preferences": {
                    "output_format": "table",
                    "timestamp_format": "relative",
                    "color_output": True,
                },
            },
        ],
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


class TestConfigFromFile:
    """Test loading configuration from a file."""

    def test_load_from_file_basic(self, temp_config_file: Path):
        """Test basic loading from config file."""
        config = get_context(config_path=temp_config_file)

        assert config.context_name == "production"
        assert config.cluster.name == "prod-cluster"
        assert str(config.cluster.base_url) == "https://api.prod.example.com/"
        assert config.workspace == "prod-workspace"
        assert config.preferences.output_format == "yaml"
        assert config.preferences.timestamp_format == "iso8601"
        assert config.preferences.color_output is True

    def test_load_different_context_from_file(self, temp_config_file: Path):
        """Test loading a specific context from file."""
        params: ConfigParams = {"current_context": "development"}
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "development"
        assert config.cluster.name == "dev-cluster"
        assert str(config.cluster.base_url) == "https://api.dev.example.com/"
        assert config.workspace == "dev-workspace"
        assert config.preferences.output_format == "json"

    def test_load_local_context_from_file(self, temp_config_file: Path):
        """Test loading the local context from file."""
        params: ConfigParams = {"current_context": "local"}
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "local"
        assert config.cluster.name == "local-cluster"
        assert str(config.cluster.base_url) == "http://localhost:8000/"
        assert config.cluster.certificate_authority == "/tmp/local-ca.pem"
        assert config.workspace == "local-workspace"
        assert config.preferences.output_format == "table"
        assert config.preferences.timestamp_format == "relative"


class TestConfigFromEnvVars:
    """Test loading configuration from environment variables."""

    def test_env_var_override_context(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """Test environment variable overriding context."""
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "development")

        config = get_context(config_path=temp_config_file)

        assert config.context_name == "development"
        assert config.cluster.name == "dev-cluster"

    def test_env_var_override_workspace(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """Test environment variable overriding workspace."""
        monkeypatch.setenv("NHX_WORKSPACE", "custom-workspace")

        config = get_context(config_path=temp_config_file)

        # Context should still be production (from file)
        assert config.context_name == "production"
        # But workspace should be overridden
        assert config.workspace == "custom-workspace"

    def test_env_var_override_preferences(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """Test environment variables overriding preferences."""
        monkeypatch.setenv("NHX_OUTPUT_FORMAT", "json")
        monkeypatch.setenv("NHX_COLOR_OUTPUT", "false")

        config = get_context(config_path=temp_config_file)

        assert config.context_name == "production"
        assert config.preferences.output_format == "json"
        assert config.preferences.color_output is False

    def test_env_var_multiple_overrides(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """Test multiple environment variable overrides at once."""
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "local")
        monkeypatch.setenv("NHX_WORKSPACE", "test-workspace")
        monkeypatch.setenv("NHX_OUTPUT_FORMAT", "json")
        monkeypatch.setenv("NHX_TIMESTAMP_FORMAT", "iso8601")

        config = get_context(config_path=temp_config_file)

        assert config.context_name == "local"
        assert config.workspace == "test-workspace"
        assert config.preferences.output_format == "json"
        assert config.preferences.timestamp_format == "iso8601"

    def test_env_var_access_token_overrides_context_user(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """NHX_ACCESS_TOKEN should override user auth from config context."""
        monkeypatch.setenv("NHX_ACCESS_TOKEN", "env-access-token-123")

        config = get_context(config_path=temp_config_file)

        assert isinstance(config.user, OAuthUser)
        assert config.user.token.get_secret_value() == "env-access-token-123"
        assert config.user.refresh_token is None
        assert not hasattr(config.user, "token_endpoint")

    def test_env_var_access_token_ignores_legacy_api_key_env(
        self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """NHX_ACCESS_TOKEN is used even when legacy NHX_API_KEY is present."""
        monkeypatch.setenv("NHX_ACCESS_TOKEN", "env-access-token-123")
        monkeypatch.setenv("NHX_API_KEY", "env-api-key-456")

        config = get_context(config_path=temp_config_file)

        assert isinstance(config.user, OAuthUser)
        assert config.user.token.get_secret_value() == "env-access-token-123"


class TestConfigFromSDK:
    """Test loading configuration using SDK ConfigParams."""

    def test_sdk_override_context(self, temp_config_file: Path):
        """Test SDK params overriding context."""
        params: ConfigParams = {"current_context": "development"}
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "development"
        assert config.cluster.name == "dev-cluster"

    def test_sdk_override_workspace(self, temp_config_file: Path):
        """Test SDK params overriding workspace."""
        params: ConfigParams = {"workspace": "sdk-workspace"}
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "production"
        assert config.workspace == "sdk-workspace"

    def test_sdk_override_certificate_authority(self, temp_config_file: Path, tmp_path: Path):
        """Test SDK params overriding cluster certificate authority."""
        ca_bundle = str(tmp_path / "override-ca.pem")
        params: ConfigParams = {"current_context": "local", "certificate_authority": ca_bundle}

        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "local"
        assert config.cluster.certificate_authority == ca_bundle

    def test_sdk_certificate_authority_override_does_not_mutate_stored_cluster(
        self, temp_config_file: Path, tmp_path: Path
    ):
        """Test runtime certificate authority override does not mutate loaded config."""
        ca_bundle = str(tmp_path / "override-ca.pem")
        params: ConfigParams = {"current_context": "local", "certificate_authority": ca_bundle}
        config = Config.load(config_path=temp_config_file, overrides=params)

        resolved = config.resolve()
        stored_cluster = next(cluster for cluster in config._config_file.clusters if cluster.name == "local-cluster")

        assert resolved.cluster.certificate_authority == ca_bundle
        assert stored_cluster.certificate_authority == "/tmp/local-ca.pem"

        config.certificate_authority = None
        assert config.resolve().cluster.certificate_authority == "/tmp/local-ca.pem"

    def test_reload_preserves_certificate_authority_override(self, temp_config_file: Path, tmp_path: Path):
        """Test reload preserves runtime certificate authority override."""
        ca_bundle = str(tmp_path / "override-ca.pem")
        params: ConfigParams = {"current_context": "local", "certificate_authority": ca_bundle}
        config = Config.load(config_path=temp_config_file, overrides=params)

        config.reload()

        resolved = config.resolve()
        assert resolved.context_name == "local"
        assert resolved.cluster.certificate_authority == ca_bundle

    def test_sdk_override_preferences(self, temp_config_file: Path):
        """Test SDK params overriding preferences."""
        params: ConfigParams = {
            "output_format": "json",
            "timestamp_format": "relative",
            "color_output": False,
        }
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.preferences.output_format == "json"
        assert config.preferences.timestamp_format == "relative"
        assert config.preferences.color_output is False

    def test_sdk_override_multiple_params(self, temp_config_file: Path):
        """Test SDK params with multiple overrides."""
        params: ConfigParams = {
            "current_context": "local",
            "workspace": "custom-workspace",
            "output_format": "table",
        }
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "local"
        assert config.workspace == "custom-workspace"
        assert config.preferences.output_format == "table"


class TestOverridePrecedence:
    """Test that overrides follow correct precedence: file < env vars < SDK params."""

    def test_sdk_overrides_env_var(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that SDK params take precedence over environment variables."""
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "development")
        monkeypatch.setenv("NHX_WORKSPACE", "env-workspace")

        # SDK params should override env vars
        params: ConfigParams = {"current_context": "local", "workspace": "sdk-workspace"}
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "local"  # SDK wins over env
        assert config.workspace == "sdk-workspace"  # SDK wins over env

    def test_env_var_overrides_file(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that environment variables override file values."""
        # File has production context
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "development")

        config = get_context(config_path=temp_config_file)

        assert config.context_name == "development"  # Env wins over file

    def test_full_precedence_chain(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """Test complete precedence chain: SDK > env > file."""
        # File has: production context, prod-workspace, yaml output
        # Env sets: development context, env-workspace
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "development")
        monkeypatch.setenv("NHX_WORKSPACE", "env-workspace")
        monkeypatch.setenv("NHX_OUTPUT_FORMAT", "json")

        # SDK overrides context and workspace but not output format
        params: ConfigParams = {"current_context": "local", "workspace": "sdk-workspace"}
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "local"  # SDK wins
        assert config.workspace == "sdk-workspace"  # SDK wins
        assert config.preferences.output_format == "json"  # Env wins

    def test_partial_overrides(self, temp_config_file: Path):
        """Test that only specified params are overridden."""
        # Override only workspace, everything else should come from file/context
        params: ConfigParams = {"workspace": "override-workspace"}
        config = get_context(config_path=temp_config_file, overrides=params)

        assert config.context_name == "production"  # From file
        assert config.workspace == "override-workspace"  # Overridden
        assert config.cluster.name == "prod-cluster"  # From context
        assert config.preferences.output_format == "yaml"  # From context


class TestConfigWithoutFile:
    """Test configuration without a config file using only env vars and SDK params."""

    def test_config_from_env_vars_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test configuring entirely through environment variables without a config file."""
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NHX_ACCESS_TOKEN", "test-token-123")
        monkeypatch.setenv("NHX_WORKSPACE", "test-workspace")
        monkeypatch.setenv("NHX_OUTPUT_FORMAT", "json")

        config = get_context()

        assert config.context_name == "default"
        assert config.cluster.name == "default-cluster"
        assert str(config.cluster.base_url) == "https://api.example.com/"
        assert config.user is not None
        assert isinstance(config.user, OAuthUser)
        assert config.user.token.get_secret_value() == "test-token-123"
        assert config.workspace == "test-workspace"
        assert config.preferences.output_format == "json"

    def test_config_from_env_access_token_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """NHX_ACCESS_TOKEN should work without a config file."""
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NHX_ACCESS_TOKEN", "env-access-token-123")

        config = get_context()

        assert config.user is not None
        assert isinstance(config.user, OAuthUser)
        assert config.user.token.get_secret_value() == "env-access-token-123"
        assert config.user.refresh_token is None
        assert not hasattr(config.user, "token_endpoint")

    def test_legacy_workload_token_env_is_ignored(self, monkeypatch: pytest.MonkeyPatch):
        """NEMO_WORKLOAD_TOKEN is no longer a runtime auth bootstrap source."""
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NEMO_WORKLOAD_TOKEN", "workload-token-123")

        config = get_context()

        assert isinstance(config.user, NoAuthUser)

    def test_legacy_workload_token_file_env_is_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """NEMO_WORKLOAD_TOKEN_FILE is no longer read as a bearer-token file."""
        token_path = tmp_path / "workload.token"
        token_path.write_text("workload-token-from-file\n", encoding="utf-8")
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NEMO_WORKLOAD_TOKEN_FILE", str(token_path))

        config = get_context()

        assert isinstance(config.user, NoAuthUser)

    def test_missing_legacy_workload_token_file_is_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Legacy workload token file env no longer triggers config-time file reads."""
        token_path = tmp_path / "missing-workload.token"
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NEMO_WORKLOAD_TOKEN_FILE", str(token_path))

        config = get_context()

        assert isinstance(config.user, NoAuthUser)

    def test_workload_identity_token_file_is_not_static_access_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The workload identity token file env var is handled by the client factory, not Config."""
        token_path = tmp_path / "workload.token"
        token_path.write_text("subject-token\n", encoding="utf-8")
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(token_path))

        config = get_context()

        assert isinstance(config.user, NoAuthUser)

    def test_nhx_access_token_precedes_workload_token_env(self, monkeypatch: pytest.MonkeyPatch):
        """NHX_ACCESS_TOKEN remains the highest-precedence token env var."""
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NHX_ACCESS_TOKEN", "preferred-token")
        monkeypatch.setenv("NEMO_WORKLOAD_TOKEN", "workload-token-123")
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, "/var/run/secrets/nemo-helix/workload/token")

        config = get_context()

        assert isinstance(config.user, OAuthUser)
        assert config.user.token.get_secret_value() == "preferred-token"

    def test_runtime_access_token_source_label_uses_config_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Runtime token source labels should share Config's token env precedence."""
        missing_token_path = tmp_path / "missing-workload.token"
        monkeypatch.setenv("NHX_ACCESS_TOKEN", "preferred-token")
        monkeypatch.setenv("NEMO_WORKLOAD_TOKEN", "workload-token-123")
        monkeypatch.setenv("NEMO_WORKLOAD_TOKEN_FILE", str(missing_token_path))
        monkeypatch.setenv(WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR, str(missing_token_path))

        assert Config.runtime_access_token_source_label() == "NHX_ACCESS_TOKEN environment override"

    def test_config_from_env_access_token_ignores_legacy_api_key_env(self, monkeypatch: pytest.MonkeyPatch):
        """NHX_ACCESS_TOKEN is used when legacy NHX_API_KEY is present without config."""
        monkeypatch.setenv("NHX_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("NHX_ACCESS_TOKEN", "env-access-token-123")
        monkeypatch.setenv("NHX_API_KEY", "env-api-key-456")

        config = get_context()

        assert config.user is not None
        assert isinstance(config.user, OAuthUser)
        assert config.user.token.get_secret_value() == "env-access-token-123"

    def test_config_from_sdk_params_only(self, tmp_path: Path):
        """Test configuring entirely through SDK ConfigParams without a config file."""
        params: ConfigParams = {
            "base_url": "https://api.sdk.com",
            "access_token": "sdk-token-456",
            "workspace": "sdk-workspace",
            "output_format": "table",
        }

        config = get_context(overrides=params)

        assert config.context_name == "default"
        assert str(config.cluster.base_url) == "https://api.sdk.com/"
        assert config.workspace == "sdk-workspace"
        assert config.preferences.output_format == "table"

    def test_config_from_sdk_params_only_with_certificate_authority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test SDK certificate authority override without a config file."""
        default_config_path = tmp_path / "missing-config.yaml"
        monkeypatch.delenv("NHX_CONFIG_FILE", raising=False)
        monkeypatch.setattr(Config, "get_default_config_path", classmethod(lambda _cls: default_config_path))
        ca_bundle = str(tmp_path / "sdk-ca.pem")
        params: ConfigParams = {
            "base_url": "https://api.sdk.com",
            "certificate_authority": ca_bundle,
        }

        config = Config.load(overrides=params)
        resolved = config.resolve()

        assert str(resolved.cluster.base_url) == "https://api.sdk.com/"
        assert resolved.cluster.certificate_authority == ca_bundle

    def test_config_without_file_no_auth(self, tmp_path: Path):
        """Test configuration without auth (NoAuth) and without config file."""
        params: ConfigParams = {
            "base_url": "http://localhost:8000",
            "workspace": "local",
        }

        config = get_context(overrides=params)

        assert config.user is not None
        assert isinstance(config.user, NoAuthUser)

    def test_config_without_file_defaults_to_localhost(self, tmp_path: Path):
        """Test that base_url defaults to localhost:8080 when no config file exists."""
        context = get_context()
        assert str(context.cluster.base_url) == DEFAULT_BASE_URL + "/"


class TestConfigFilePathOverride:
    """Test config file path override via environment variable."""

    def test_config_file_path_from_env_var(self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that NHX_CONFIG_FILE env var overrides default path."""
        # Set the env var to point to our temp config file
        monkeypatch.setenv("NHX_CONFIG_FILE", str(temp_config_file))

        # Get config without specifying path - should use env var
        config = get_context()

        # Should load from the temp config file (which has production context)
        assert config.context_name == "production"
        assert config.cluster.name == "prod-cluster"
        assert config.workspace == "prod-workspace"

    def test_explicit_path_overrides_env_var(
        self, temp_config_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that explicit config_path parameter takes precedence over env var."""
        # Create another config file
        alt_config_data = {
            "current_context": "alt",
            "clusters": [
                {
                    "name": "alt-cluster",
                    "base_url": "https://api.alt.com",
                }
            ],
            "users": [
                {
                    "name": "alt-user",
                }
            ],
            "contexts": [
                {
                    "name": "alt",
                    "cluster": "alt-cluster",
                    "user": "alt-user",
                    "workspace": "alt-workspace",
                    "preferences": {},
                }
            ],
        }
        alt_config_path = tmp_path / "alt_config.yaml"
        with open(alt_config_path, "w") as f:
            yaml.dump(alt_config_data, f)

        # Set env var to temp_config_file
        monkeypatch.setenv("NHX_CONFIG_FILE", str(temp_config_file))

        # But explicitly pass alt_config_path - should use explicit path
        config = get_context(config_path=alt_config_path)

        assert config.context_name == "alt"
        assert config.cluster.name == "alt-cluster"


class TestConfigErrors:
    """Test error handling in config module."""

    def test_invalid_context_name(self, temp_config_file: Path):
        """Test error when invalid context name is provided."""
        params: ConfigParams = {"current_context": "nonexistent"}

        with pytest.raises(ValueError, match="Context 'nonexistent' not found"):
            get_context(config_path=temp_config_file, overrides=params)

    def test_missing_cluster_reference(self, temp_config_file: Path):
        """Test error when context references missing cluster."""
        # Modify file to have invalid cluster reference
        with open(temp_config_file) as f:
            config_data = yaml.safe_load(f)
        config_data["contexts"][0]["cluster"] = "nonexistent-cluster"
        with open(temp_config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(
            ValueError,
            match="Context 'production' references non-existent cluster 'nonexistent-cluster'",
        ):
            get_context(config_path=temp_config_file)

    def test_config_file_not_found(self, tmp_path: Path):
        """Test error when explicit config file path doesn't exist."""
        nonexistent_path = tmp_path / "does_not_exist.yaml"

        with pytest.raises(FileNotFoundError, match=f"Config file not found at {nonexistent_path}"):
            get_context(config_path=nonexistent_path)

    def test_config_file_parse_error(self, tmp_path: Path):
        """Test error when config file has invalid YAML syntax."""
        invalid_yaml_path = tmp_path / "invalid.yaml"

        # Write invalid YAML
        with open(invalid_yaml_path, "w") as f:
            f.write("invalid: yaml: syntax:\n  - bad indentation\n  this is wrong")

        with pytest.raises(ValueError, match="Error parsing config file"):
            get_context(config_path=invalid_yaml_path)

    def test_config_file_env_var_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test error when NHX_CONFIG_FILE points to nonexistent file."""
        nonexistent_path = tmp_path / "does_not_exist.yaml"

        # Set env var to nonexistent path
        monkeypatch.setenv("NHX_CONFIG_FILE", str(nonexistent_path))

        with pytest.raises(FileNotFoundError, match=f"Config file not found at {nonexistent_path}"):
            get_context()


class TestConfigWrite:
    """Test Config.write() method for creating/updating config files."""

    def test_write_with_result_reports_context_and_creation(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"

        created = Config.write_with_result(
            {"base_url": "https://staging.example.com"},
            context_name="staging",
            config_path=config_path,
        )

        assert created.context_name == "staging"
        assert created.created is True
        assert created.config.get_config_file().current_context == "staging"

        updated = Config.write_with_result(
            {"workspace": "updated-workspace"},
            context_name="staging",
            config_path=config_path,
        )

        assert updated.context_name == "staging"
        assert updated.created is False
        assert updated.config.resolve().workspace == "updated-workspace"

    def test_write_can_leave_current_context_unset(self, tmp_path: Path):
        result = Config.write_with_result(
            {"base_url": "https://staging.example.com"},
            context_name="staging",
            config_path=tmp_path / "config.yaml",
            set_current_if_unset=False,
        )

        assert result.created is True
        assert result.config.get_config_file().current_context is None

    def test_write_without_context_name_falls_back_to_default_when_current_context_unset(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        # Create a named context without activating it (current_context stays None).
        Config.write_with_result(
            {"base_url": "https://staging.example.com"},
            context_name="staging",
            config_path=config_path,
            set_current_if_unset=False,
        )
        assert Config.load(config_path=config_path).get_config_file().current_context is None

        # A subsequent write without context_name must not raise (regression: config.resolve() raised
        # ValueError when contexts existed but current_context was None). It targets DEFAULT_CONTEXT.
        result = Config.write_with_result({"base_url": "https://default.example.com"}, config_path=config_path)

        assert result.context_name == "default"

    def test_write_rejects_unknown_context_selected_by_environment(
        self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "prodction")

        with pytest.raises(
            ValueError,
            match="Context 'prodction' not found. Available contexts: production, development, local",
        ):
            Config.write_with_result({"workspace": "updated-workspace"}, config_path=temp_config_file)

        config = Config.load(config_path=temp_config_file)
        assert "prodction" not in {context.name for context in config.get_config_file().contexts}

    def test_write_allows_explicit_context_when_environment_selects_unknown_context(
        self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "prodction")

        result = Config.write_with_result(
            {"base_url": "https://new.example.com", "workspace": "new-workspace"},
            context_name="new-context",
            config_path=temp_config_file,
        )

        assert result.context_name == "new-context"
        assert result.created is True

    def test_write_creates_new_config_file(self, tmp_path: Path):
        """Test that write() creates a new config file when none exists."""
        config_path = tmp_path / "new_config.yaml"
        assert not config_path.exists()

        config = Config.write(
            {"base_url": "http://test.example.com", "workspace": "test-ws"},
            context_name="default",
            config_path=config_path,
        )

        assert config_path.exists()
        config_file = config.get_config_file()
        assert config_file.current_context == "default"
        assert len(config_file.clusters) == 1
        assert config_file.clusters[0].name == "default-cluster"
        assert str(config_file.clusters[0].base_url) == "http://test.example.com/"
        assert len(config_file.contexts) == 1
        assert config_file.contexts[0].workspace == "test-ws"

    def test_write_creates_config_at_env_var_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that write() creates config when NHX_CONFIG_FILE points to non-existent file."""
        config_path = tmp_path / "subdir" / "new_config.yaml"
        assert not config_path.exists()

        monkeypatch.setenv("NHX_CONFIG_FILE", str(config_path))

        config = Config.write(
            {"base_url": "http://env-var.example.com", "workspace": "env-ws"},
            context_name="env-context",
        )

        assert config_path.exists()
        config_file = config.get_config_file()
        assert config_file.current_context == "env-context"
        assert config_file.clusters[0].name == "env-context-cluster"
        assert str(config_file.clusters[0].base_url) == "http://env-var.example.com/"

    def test_write_updates_existing_config(self, temp_config_file: Path):
        """Test that write() updates existing config without losing data."""
        # Get original config
        original = Config.load(config_path=temp_config_file)
        original_contexts = len(original.get_config_file().contexts)

        # Write update to existing context
        config = Config.write(
            {"workspace": "updated-workspace"},
            context_name="production",
            config_path=temp_config_file,
        )

        config_file = config.get_config_file()
        # Should still have same number of contexts
        assert len(config_file.contexts) == original_contexts
        # Production context should be updated
        prod_ctx = next(c for c in config_file.contexts if c.name == "production")
        assert prod_ctx.workspace == "updated-workspace"

    def test_write_creates_new_context_in_existing_file(self, temp_config_file: Path):
        """Test that write() can add a new context to existing file."""
        original = Config.load(config_path=temp_config_file)
        original_contexts = len(original.get_config_file().contexts)

        # Add new context
        config = Config.write(
            {"base_url": "http://staging.example.com", "workspace": "staging-ws"},
            context_name="staging",
            config_path=temp_config_file,
        )

        config_file = config.get_config_file()
        assert len(config_file.contexts) == original_contexts + 1
        staging_ctx = next(c for c in config_file.contexts if c.name == "staging")
        assert staging_ctx.workspace == "staging-ws"

    def test_write_with_access_token(self, tmp_path: Path):
        """Test that write() correctly sets OAuth access token authentication."""
        config_path = tmp_path / "config.yaml"
        config = Config.write(
            {"base_url": "http://test.example.com", "access_token": "test-token-123"},
            context_name="default",
            config_path=config_path,
        )

        config_file = config.get_config_file()
        user = config_file.users[0]
        assert isinstance(user, OAuthUser)
        assert user.token.get_secret_value() == "test-token-123"

    def test_write_with_access_token_none_clears_oauth_user(self, tmp_path: Path):
        """Test that write() clears OAuth credentials when access_token is explicitly None."""
        config_path = tmp_path / "config.yaml"
        Config.write(
            {
                "base_url": "http://test.example.com",
                "access_token": "test-token-123",
                "refresh_token": "test-refresh-123",
            },
            context_name="default",
            config_path=config_path,
        )

        config = Config.write(
            {"access_token": None, "refresh_token": None},
            context_name="default",
            config_path=config_path,
        )

        user = config.get_config_file().users[0]
        assert isinstance(user, NoAuthUser)

        with open(config_path) as f:
            data = yaml.safe_load(f)

        stored_user = data["users"][0]
        assert stored_user["type"] == "no-auth"
        assert "token" not in stored_user
        assert "refresh_token" not in stored_user

    def test_write_without_access_token_preserves_oauth_user(self, tmp_path: Path):
        """Test that unrelated config writes preserve existing OAuth credentials."""
        config_path = tmp_path / "config.yaml"
        Config.write(
            {
                "base_url": "http://test.example.com",
                "access_token": "test-token-123",
                "refresh_token": "test-refresh-123",
            },
            context_name="default",
            config_path=config_path,
        )

        config = Config.write(
            {"workspace": "updated-workspace"},
            context_name="default",
            config_path=config_path,
        )

        user = config.get_config_file().users[0]
        assert isinstance(user, OAuthUser)
        assert user.token.get_secret_value() == "test-token-123"
        assert user.refresh_token is not None
        assert user.refresh_token.get_secret_value() == "test-refresh-123"

    def test_write_without_access_token_creates_noauth_user(self, tmp_path: Path):
        """Test that write() creates NoAuthUser when no access_token provided."""
        config_path = tmp_path / "config.yaml"
        config = Config.write(
            {"base_url": "http://test.example.com"},
            context_name="default",
            config_path=config_path,
        )

        config_file = config.get_config_file()
        user = config_file.users[0]
        assert isinstance(user, NoAuthUser)

    def test_write_sets_preferences(self, tmp_path: Path):
        """Test that write() correctly sets preferences."""
        config_path = tmp_path / "config.yaml"
        config = Config.write(
            {
                "base_url": "http://test.example.com",
                "output_format": "json",
                "timestamp_format": "relative",
            },
            context_name="default",
            config_path=config_path,
        )

        config_file = config.get_config_file()
        ctx = config_file.contexts[0]
        assert ctx.preferences.output_format == "json"
        assert ctx.preferences.timestamp_format == "relative"

    def test_write_sets_current_context_for_new_file(self, tmp_path: Path):
        """Test that write() sets current_context when creating new file."""
        config_path = tmp_path / "config.yaml"
        config = Config.write(
            {"base_url": "http://test.example.com"},
            context_name="my-context",
            config_path=config_path,
        )

        config_file = config.get_config_file()
        assert config_file.current_context == "my-context"

    def test_write_preserves_current_context(self, temp_config_file: Path):
        """Test that write() preserves existing current_context when updating."""
        # Add new context, should not change current_context
        config = Config.write(
            {"base_url": "http://new.example.com"},
            context_name="new-context",
            config_path=temp_config_file,
        )

        config_file = config.get_config_file()
        # Original current_context was "production"
        assert config_file.current_context == "production"

    def test_write_can_override_current_context(self, temp_config_file: Path):
        """Test that write() can explicitly change current_context via params."""
        config = Config.write(
            {"current_context": "development"},
            context_name="production",
            config_path=temp_config_file,
        )

        config_file = config.get_config_file()
        assert config_file.current_context == "development"

    def test_write_without_context_name_uses_current_context_from_existing_file(self, temp_config_file: Path):
        """Test that write() uses current context from existing file when context_name not provided."""
        # temp_config_file has current_context="production"
        config = Config.write(
            {"workspace": "new-workspace"},
            context_name=None,  # Not providing context_name
            config_path=temp_config_file,
        )

        # Should have updated the "production" context (the current context)
        config_file = config.get_config_file()
        prod_ctx = next(c for c in config_file.contexts if c.name == "production")
        assert prod_ctx.workspace == "new-workspace"

    def test_write_without_context_name_respects_env_var_override(
        self, temp_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that write() respects NHX_CURRENT_CONTEXT env var when context_name not provided."""
        # temp_config_file has current_context="production" in the file
        # But we override with env var to use "development"
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "development")

        config = Config.write(
            {"workspace": "env-override-workspace"},
            context_name=None,  # Not providing context_name - should use env var
            config_path=temp_config_file,
        )

        # Should have updated the "development" context (from env var), not "production" (from file)
        config_file = config.get_config_file()
        dev_ctx = next(c for c in config_file.contexts if c.name == "development")
        assert dev_ctx.workspace == "env-override-workspace"

        # Production context should be unchanged
        prod_ctx = next(c for c in config_file.contexts if c.name == "production")
        assert prod_ctx.workspace == "prod-workspace"

    def test_write_without_context_name_uses_default_when_no_file_exists(self, tmp_path: Path):
        """Test that write() uses DEFAULT_CONTEXT when no config file exists and context_name not provided."""
        config_path = tmp_path / "new_config.yaml"
        assert not config_path.exists()

        config = Config.write(
            {"base_url": "http://test.example.com"},
            context_name=None,  # Not providing context_name
            config_path=config_path,
        )

        config_file = config.get_config_file()
        # Should use "default" as the context name
        assert config_file.current_context == "default"
        assert len(config_file.contexts) == 1
        assert config_file.contexts[0].name == "default"

    def test_first_write_respects_current_context_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        config_path = tmp_path / "new_config.yaml"
        monkeypatch.setenv("NHX_CURRENT_CONTEXT", "staging")

        result = Config.write_with_result(
            {"base_url": "https://staging.example.com"},
            config_path=config_path,
        )

        assert result.context_name == "staging"
        assert result.config.get_config_file().current_context == "staging"
        assert result.config.resolve().context_name == "staging"


class TestConfigFilePermissions:
    """Test that config files are created with secure permissions."""

    def test_save_creates_directory_with_700_permissions(self, tmp_path: Path):
        """Test that save() creates config directory with owner-only access (700)."""
        config_dir = tmp_path / "nhx"
        config_path = config_dir / "config.yaml"
        assert not config_dir.exists()

        Config.write(
            {"base_url": "http://test.example.com"},
            context_name="default",
            config_path=config_path,
        )

        assert config_dir.exists()
        dir_mode = config_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700, f"Expected 700, got {oct(dir_mode)}"

    def test_save_creates_file_with_600_permissions(self, tmp_path: Path):
        """Test that save() creates config file with owner read/write only (600)."""
        config_path = tmp_path / "nhx" / "config.yaml"

        Config.write(
            {"base_url": "http://test.example.com"},
            context_name="default",
            config_path=config_path,
        )

        assert config_path.exists()
        file_mode = config_path.stat().st_mode & 0o777
        assert file_mode == 0o600, f"Expected 600, got {oct(file_mode)}"

    def test_save_tolerates_unowned_parent_directory(self, tmp_path: Path):
        """Saving under a world-writable parent (e.g. /tmp) should not fail on dir chmod."""
        config_path = tmp_path / "config.yaml"

        real_chmod = os.chmod

        def chmod_side_effect(path, mode):
            if Path(path) == tmp_path:
                raise PermissionError("Operation not permitted")
            real_chmod(path, mode)

        with patch("nemo_helix_ext.config.config.os.chmod", side_effect=chmod_side_effect):
            Config.write(
                {"base_url": "http://test.example.com"},
                context_name="default",
                config_path=config_path,
            )

        assert config_path.exists()
        file_mode = config_path.stat().st_mode & 0o777
        assert file_mode == 0o600, f"Expected 600, got {oct(file_mode)}"

    def test_save_raises_when_file_chmod_fails(self, tmp_path: Path):
        """Credential file permission failures must remain fatal."""
        config_path = tmp_path / "config.yaml"

        real_chmod = os.chmod

        def chmod_side_effect(path, mode):
            if Path(path) == config_path:
                raise PermissionError("Operation not permitted")
            real_chmod(path, mode)

        with (
            patch("nemo_helix_ext.config.config.os.chmod", side_effect=chmod_side_effect),
            pytest.raises(PermissionError),
        ):
            Config.write(
                {"base_url": "http://test.example.com"},
                context_name="default",
                config_path=config_path,
            )


class TestUserTypeDiscriminator:
    """Test User type discriminator and deserialization."""

    def test_user_without_type_defaults_to_no_auth(self):
        """User without explicit type field should deserialize to NoAuthUser."""
        config_file = ConfigFile.model_validate(
            {
                "clusters": [{"name": "test", "base_url": "http://localhost"}],
                "users": [{"name": "test-user"}],  # No type field
                "contexts": [{"name": "test", "cluster": "test", "user": "test-user"}],
            }
        )

        user = config_file.users[0]
        assert isinstance(user, NoAuthUser)
        assert user.type == "no-auth"

    def test_user_with_explicit_no_auth_type(self):
        """User with explicit type='no-auth' should deserialize to NoAuthUser."""
        config_file = ConfigFile.model_validate(
            {
                "clusters": [{"name": "test", "base_url": "http://localhost"}],
                "users": [{"name": "test-user", "type": "no-auth"}],
                "contexts": [{"name": "test", "cluster": "test", "user": "test-user"}],
            }
        )

        user = config_file.users[0]
        assert isinstance(user, NoAuthUser)

    def test_user_with_oauth_type(self):
        """User with type='oauth' should deserialize to OAuthUser."""
        config_file = ConfigFile.model_validate(
            {
                "clusters": [{"name": "test", "base_url": "http://localhost"}],
                "users": [{"name": "test-user", "type": "oauth", "token": "secret123", "refresh_token": None}],
                "contexts": [{"name": "test", "cluster": "test", "user": "test-user"}],
            }
        )

        user = config_file.users[0]
        assert isinstance(user, OAuthUser)
        assert user.token.get_secret_value() == "secret123"

    def test_mixed_user_types_in_config(self):
        """Config with multiple user types should deserialize each correctly."""
        config_file = ConfigFile.model_validate(
            {
                "clusters": [{"name": "test", "base_url": "http://localhost"}],
                "users": [
                    {"name": "no-auth-user"},  # No type -> NoAuthUser
                    {
                        "name": "oauth-user",
                        "type": "oauth",
                        "token": "token",
                        "refresh_token": "refresh",
                    },
                ],
                "contexts": [{"name": "test", "cluster": "test", "user": "no-auth-user"}],
            }
        )

        assert isinstance(config_file.users[0], NoAuthUser)
        assert isinstance(config_file.users[1], OAuthUser)


class TestLegacyApiKeyMigration:
    """Test migration of legacy api-key users to oauth users."""

    def test_legacy_api_key_user_migrates_to_oauth_user(self, tmp_path: Path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "current_context": "default",
                    "clusters": [{"name": "default", "base_url": "https://api.example.com"}],
                    "users": [{"name": "default", "type": "api-key", "api_key": "legacy-token-123"}],
                    "contexts": [{"name": "default", "cluster": "default", "user": "default"}],
                }
            )
        )

        config = Config.load(config_path=config_path)
        user = config.get_config_file().users[0]

        assert isinstance(user, OAuthUser)
        assert user.token.get_secret_value() == "legacy-token-123"
        assert user.refresh_token is None

    def test_legacy_email_api_key_migrates_to_unsigned_jwt(self, tmp_path: Path):
        from nemo_helix_ext.auth.helpers import decode_jwt_claims

        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "current_context": "default",
                    "clusters": [{"name": "default", "base_url": "https://api.example.com"}],
                    "users": [{"name": "default", "type": "api-key", "api_key": "admin@example.com"}],
                    "contexts": [{"name": "default", "cluster": "default", "user": "default"}],
                }
            )
        )

        config = Config.load(config_path=config_path)
        user = config.get_config_file().users[0]

        assert isinstance(user, OAuthUser)
        claims = decode_jwt_claims(user.token.get_secret_value())
        assert claims["sub"] == "admin@example.com"
        assert claims["email"] == "admin@example.com"


class TestNoAuthUserGetClientConfig:
    """Test NoAuthUser.get_client_config() behavior."""

    def test_get_client_config_returns_empty(self):
        """NoAuthUser should return empty config."""
        user = NoAuthUser(name="test")

        config = user.get_client_config()

        assert config == {}
