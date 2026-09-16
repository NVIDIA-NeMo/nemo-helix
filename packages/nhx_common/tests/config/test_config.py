# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
from nhx.common.config import (
    Configuration,
    DatabaseConfig,
    DockerConfig,
    get_service_config_prefix,
    nhx_user_data_dir,
)


# Patch the default configuration file path for testing
@pytest.fixture(autouse=True)
def patch_config_file_path(monkeypatch):
    # Clear the cache before each test to ensure fresh config reads
    Configuration.clear_cache()
    # Set the environment variable to a test configuration file path
    test_dir = pathlib.Path(__file__).parent
    monkeypatch.setenv("NHX_CONFIG_FILE_PATH", str(test_dir / "fixtures" / "simple.yaml"))


def test_get_service_config_prefix():
    service_name = "test_service"
    expected_prefix = f"NHX_{service_name.upper()}_"
    assert get_service_config_prefix(service_name) == expected_prefix


def test_configuration_get_platform_config():
    platform_config = Configuration.get_platform_config()
    assert platform_config is not None
    assert hasattr(platform_config, "base_url")
    assert platform_config.base_url is not None
    assert platform_config.base_url.startswith("http")  # Assuming base_url should be a valid URL


def test_configuration_get_platform_config_with_env(monkeypatch):
    # Set an environment variable to test the Configuration
    monkeypatch.setenv("NHX_BASE_URL", "http://test-platform:8000")

    platform_config = Configuration.get_platform_config()
    assert platform_config.base_url == "http://test-platform:8000"


def test_configuration_get_platform_config_with_file(monkeypatch):
    test_dir = pathlib.Path(__file__).parent
    config_file_path = test_dir / "fixtures" / "simple.yaml"

    assert config_file_path.exists(), f"Test configuration file {config_file_path} does not exist."
    monkeypatch.setenv("NHX_CONFIG_FILE_PATH", str(config_file_path))
    platform_config = Configuration.get_platform_config()

    # Check if the base_url is set correctly from the file
    assert platform_config.base_url == "http://gateway:8000"


def test_configuration_get_platform_config_malformed_file(monkeypatch):
    test_dir = pathlib.Path(__file__).parent

    # Simulate a malformed configuration file
    config_file_path = test_dir / "fixtures" / "malformed.yaml"
    assert config_file_path.exists(), f"Test configuration file {config_file_path} does not exist."
    monkeypatch.setenv("NHX_CONFIG_FILE_PATH", str(config_file_path))
    with pytest.raises(ValueError):
        Configuration.get_platform_config()


def test_configuration_get_platform_config_list_typed_file(monkeypatch):
    test_dir = pathlib.Path(__file__).parent

    # We don't accept configurations that are a list type
    config_file_path = test_dir / "fixtures" / "list.yaml"
    assert config_file_path.exists(), f"Test configuration file {config_file_path} does not exist."
    monkeypatch.setenv("NHX_CONFIG_FILE_PATH", str(config_file_path))
    with pytest.raises(ValueError):
        Configuration.get_platform_config()


def test_database_config_database_url_default(monkeypatch, tmp_path):
    """Test database_url with no connection params returns default sqlite URL.

    Defaults to a SQLite file under the NeMo Helix user data directory (XDG-style)
    so local state survives macOS ``/tmp/`` cleanup on reboot. We override
    ``NHX_DATA_DIR`` to keep the assertion stable across developer machines.
    """
    monkeypatch.setenv("NHX_DATA_DIR", str(tmp_path))
    config = DatabaseConfig(dialect="sqlite")
    url = config.sqlalchemy_database_url()
    expected_path = nhx_user_data_dir() / "nhx-platform.db"
    assert url == f"sqlite:///{expected_path}"
    # Parent directory is created on use so SQLite can open the file.
    assert expected_path.parent.exists()


def test_database_config_database_url_sqlite_with_default_db():
    """Test database_url with SQLite and default database file."""
    config = DatabaseConfig(dialect="sqlite", path="nhx.db")
    url = config.sqlalchemy_database_url()
    assert url == "sqlite:///nhx.db"


def test_database_config_database_url_sqlite_with_memory():
    """Test database_url with SQLite and memory database."""
    config = DatabaseConfig(dialect="sqlite", path=":memory:")
    url = config.sqlalchemy_database_url()
    assert url == "sqlite:///:memory:"


def test_database_config_database_url_postgres_basic():
    """Test database_url with basic PostgreSQL configuration."""
    config = DatabaseConfig(
        dialect="postgresql", host="localhost", port=5432, user="testuser", password="testpass", name="testdb"
    )
    url = config.sqlalchemy_database_url()
    assert url == "postgresql://testuser:testpass@localhost:5432/testdb"


def test_database_config_database_url_postgres_no_auth():
    """Test database_url with PostgreSQL without authentication."""
    config = DatabaseConfig(dialect="postgresql", host="localhost", port=5432, name="testdb")
    url = config.sqlalchemy_database_url()
    assert url == "postgresql://localhost:5432/testdb"


def test_database_config_database_url_postgres_with_special_chars():
    """Test database_url with PostgreSQL and special characters in password."""
    config = DatabaseConfig(
        dialect="postgresql",
        host="db.example.com",
        port=5432,
        user="user@domain",
        password="p@ssw0rd!#$",
        name="testdb",
    )
    url = config.sqlalchemy_database_url()
    # Special characters should be URL encoded
    assert "user%40domain" in url  # @ is encoded as %40
    assert "p%40ssw0rd%21%23%24" in url  # Special chars are encoded
    assert url.startswith("postgresql://")


def test_database_config_database_url_no_port():
    """Test database_url with PostgreSQL without port specified."""
    config = DatabaseConfig(dialect="postgresql", host="localhost", user="testuser", password="testpass", name="testdb")
    url = config.sqlalchemy_database_url()
    assert url == "postgresql://testuser:testpass@localhost/testdb"


def test_database_config_database_url_only_user():
    """Test database_url with only username, no password."""
    config = DatabaseConfig(dialect="postgresql", host="localhost", port=5432, user="testuser", name="testdb")
    url = config.sqlalchemy_database_url()
    assert url == "postgresql://testuser@localhost:5432/testdb"


def test_database_config_database_url_empty_values():
    """Test database_url raises error when database name not provided."""
    config = DatabaseConfig(dialect="postgresql", host="localhost", port=None, user=None, password=None)
    with pytest.raises(ValueError, match="postgresql database requires 'name' to be set"):
        config.sqlalchemy_database_url()


def test_database_config_database_url_sqlite_absolute_path():
    """Test database_url with SQLite absolute path."""
    config = DatabaseConfig(dialect="sqlite", path="/absolute/path/to/database.db")
    url = config.sqlalchemy_database_url()
    assert url == "sqlite:////absolute/path/to/database.db"


def test_database_config_database_url_sqlite_relative_path():
    """Test database_url with SQLite relative path."""
    config = DatabaseConfig(dialect="sqlite", path="relative/path/database.db")
    url = config.sqlalchemy_database_url()
    assert url == "sqlite:///relative/path/database.db"


def test_database_config_database_url_password_only():
    """Test database_url with password but no username and no database name raises error."""
    config = DatabaseConfig(dialect="postgresql", host="localhost", port=5432, user=None, password="password")
    with pytest.raises(ValueError, match="postgresql database requires 'name' to be set"):
        config.sqlalchemy_database_url()


def test_database_config_database_url_postgres_with_invalid_path():
    """Test database_url with PostgreSQL and path field raises error."""
    config = DatabaseConfig(
        dialect="postgresql", host="localhost", port=5432, user="testuser", password="testpass", path="/invalid/path"
    )
    with pytest.raises(ValueError, match="postgresql database should not use 'path' field"):
        config.sqlalchemy_database_url()


def test_database_config_database_url_special_hostname():
    """Test database_url with special characters in hostname."""
    config = DatabaseConfig(
        dialect="postgresql",
        host="db-server.example-domain.com",
        port=5432,
        user="testuser",
        password="testpass",
        name="nhx",
    )
    url = config.sqlalchemy_database_url()
    assert url == "postgresql://testuser:testpass@db-server.example-domain.com:5432/nhx"


def test_database_config_database_url_postgres_with_database_name():
    """Test database_url with PostgreSQL and database path."""
    config = DatabaseConfig(
        dialect="postgresql", host="localhost", port=5432, user="testuser", password="testpass", name="mydb"
    )
    url = config.sqlalchemy_database_url()
    assert url == "postgresql://testuser:testpass@localhost:5432/mydb"


def test_database_config_database_url_sqlite_path_preferred_over_host():
    """Test database_url with SQLite prefers path over host when both are set."""
    config = DatabaseConfig(dialect="sqlite", host="hostname", path="/new/path/database.db")
    url = config.sqlalchemy_database_url()
    assert url == "sqlite:////new/path/database.db"


class TestDockerConfigReservedGpuDeviceIds:
    """Tests for reserved_gpu_device_ids configuration field."""

    def test_default_is_all(self):
        """Test default value is 'all' for auto-detection."""
        config = DockerConfig()
        assert config.reserved_gpu_device_ids == "all"

    @pytest.mark.parametrize(
        "input_value,expected_string,expected_ids",
        [
            pytest.param("all", "all", None, id="all_magic_string"),
            pytest.param("ALL", "all", None, id="all_uppercase"),
            pytest.param("All", "all", None, id="all_mixed_case"),
            pytest.param("  all  ", "all", None, id="all_with_whitespace"),
            pytest.param("none", "none", [], id="none_magic_string"),
            pytest.param("NONE", "none", [], id="none_uppercase"),
            pytest.param("None", "none", [], id="none_mixed_case"),
            pytest.param("  none  ", "none", [], id="none_with_whitespace"),
            pytest.param("", "", [], id="empty_string_disables_gpus"),
            pytest.param("0", "0", [0], id="single_gpu"),
            pytest.param("0,1,2,3", "0,1,2,3", [0, 1, 2, 3], id="multiple_gpus"),
            pytest.param("0, 1, 2, 3", "0, 1, 2, 3", [0, 1, 2, 3], id="multiple_gpus_with_spaces"),
            pytest.param("7", "7", [7], id="high_device_id"),
        ],
    )
    def test_valid_string_values(self, input_value, expected_string, expected_ids):
        """Test various valid string input formats."""
        config = DockerConfig(reserved_gpu_device_ids=input_value)
        assert config.reserved_gpu_device_ids == expected_string
        assert config.get_reserved_gpu_ids() == expected_ids

    @pytest.mark.parametrize(
        "invalid_value,error_match",
        [
            pytest.param("abc", "must be 'all', 'none', or comma-separated integers", id="invalid_string"),
            pytest.param("0,1,x,3", "must be 'all', 'none', or comma-separated integers", id="non_integer_in_string"),
            pytest.param("0.5", "must be 'all', 'none', or comma-separated integers", id="float_in_string"),
            pytest.param([0, 1, 2], "must be a string", id="list_type_rejected"),
            pytest.param([], "must be a string", id="empty_list_rejected"),
            pytest.param([0], "must be a string", id="single_item_list_rejected"),
            pytest.param(123, "must be a string", id="integer_not_string"),
        ],
    )
    def test_invalid_values_raise_error(self, invalid_value, error_match):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError, match=error_match):
            DockerConfig(reserved_gpu_device_ids=invalid_value)  # type: ignore[arg-type]

    def test_invalid_integer_preserves_original_exception(self):
        """Test that invalid integer parsing preserves original ValueError in chain."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            DockerConfig(reserved_gpu_device_ids="0,1,invalid,3")

        # Pydantic wraps our ValueError - dig into the error context to verify chaining
        errors = exc_info.value.errors()
        assert len(errors) == 1
        inner_error = errors[0].get("ctx", {}).get("error")
        assert inner_error is not None
        assert inner_error.__cause__ is not None
        assert isinstance(inner_error.__cause__, ValueError)
        assert "invalid" in str(inner_error.__cause__)

    def test_env_var_with_comma_separated_list(self, monkeypatch):
        """Test reserved_gpu_device_ids from environment variable with comma-separated list."""
        # Clear cached config to ensure fresh read
        monkeypatch.delenv("NHX_CONFIG_FILE_PATH", raising=False)
        monkeypatch.setenv("NHX_DOCKER_RESERVED_GPU_DEVICE_IDS", "0,1,2")

        from nhx.common.config import PlatformConfig

        # The env var handler should inject this into the docker config
        platform_config = PlatformConfig()  # type: ignore[abstract]
        assert platform_config.docker.reserved_gpu_device_ids == "0,1,2"
        assert platform_config.docker.get_reserved_gpu_ids() == [0, 1, 2]

    def test_env_var_with_all(self, monkeypatch):
        """Test reserved_gpu_device_ids from environment variable with 'all'."""
        monkeypatch.delenv("NHX_CONFIG_FILE_PATH", raising=False)
        monkeypatch.setenv("NHX_DOCKER_RESERVED_GPU_DEVICE_IDS", "all")

        from nhx.common.config import PlatformConfig

        platform_config = PlatformConfig()  # type: ignore[abstract]
        assert platform_config.docker.reserved_gpu_device_ids == "all"
        assert platform_config.docker.get_reserved_gpu_ids() is None

    def test_env_var_with_none(self, monkeypatch):
        """Test reserved_gpu_device_ids from environment variable with 'none'."""
        monkeypatch.delenv("NHX_CONFIG_FILE_PATH", raising=False)
        monkeypatch.setenv("NHX_DOCKER_RESERVED_GPU_DEVICE_IDS", "none")

        from nhx.common.config import PlatformConfig

        platform_config = PlatformConfig()  # type: ignore[abstract]
        assert platform_config.docker.reserved_gpu_device_ids == "none"
        assert platform_config.docker.get_reserved_gpu_ids() == []
