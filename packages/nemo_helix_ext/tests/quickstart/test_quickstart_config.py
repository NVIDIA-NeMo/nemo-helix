# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the quickstart config module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from nemo_helix_ext.quickstart.config import QuickstartConfig, _is_internal_tag
from nemo_helix_ext.quickstart.validators import validate_config


class TestQuickstartConfigRemove:
    """Test removal of the quickstart config file."""

    def test_remove_deletes_existing_config_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """remove() deletes the config file when it exists."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_path = QuickstartConfig.get_default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("image: nvcr.io/nhx:latest\n")
        assert config_path.exists()

        QuickstartConfig.remove()

        assert not config_path.exists()

    def test_remove_idempotent_when_file_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """remove() does not raise when the config file does not exist."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_path = QuickstartConfig.get_default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        assert not config_path.exists()

        QuickstartConfig.remove()

        assert not config_path.exists()


class TestQuickstartConfigFilePermissions:
    """Test that quickstart config files are created with secure permissions."""

    def test_save_creates_directory_with_700_permissions(self, tmp_path: Path) -> None:
        """Test that save() creates config directory with owner-only access (700)."""
        config_dir = tmp_path / "nhx"
        config_path = config_dir / "quickstart.yaml"
        assert not config_dir.exists()

        config = QuickstartConfig()
        config.save(path=config_path)

        assert config_dir.exists()
        dir_mode = config_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700, f"Expected 700, got {oct(dir_mode)}"

    def test_save_creates_file_with_600_permissions(self, tmp_path: Path) -> None:
        """Test that save() creates config file with owner read/write only (600)."""
        config_path = tmp_path / "nhx" / "quickstart.yaml"

        config = QuickstartConfig()
        config.save(path=config_path)

        assert config_path.exists()
        file_mode = config_path.stat().st_mode & 0o777
        assert file_mode == 0o600, f"Expected 600, got {oct(file_mode)}"


class TestQuickstartConfigRegistryCredentials:
    """Tests for registry host extraction and credential matching."""

    @pytest.mark.parametrize(
        ("image", "registry_host"),
        [
            ("registry.example.com/nhx-api:test", "registry.example.com"),
            ("localhost:5000/nhx-api:test", "localhost:5000"),
            ("registry.example.com/org/nhx-api:test", "registry.example.com"),
            ("namespace/nhx-api:test", ""),
            ("nhx-api:test", ""),
        ],
    )
    def test_get_registry_host(self, image: str, registry_host: str) -> None:
        config = QuickstartConfig(image=image)

        assert config.get_registry_host() == registry_host

    def test_has_registry_credentials_for_two_part_registry_image(self) -> None:
        config = QuickstartConfig(
            image="registry.example.com/nhx-api:test",
            registry_host="registry.example.com",
            registry_username="test-user",
            registry_password="test-token",  # type: ignore[arg-type]
        )

        assert config.has_registry_credentials_for_image()


class TestQuickstartConfigReservedGpuDeviceIds:
    """Tests for reserved_gpu_device_ids field (stored and loaded as comma-separated string)."""

    def test_save_and_load_reserved_gpu_device_ids_as_string(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """reserved_gpu_device_ids is stored and loaded as a string (e.g. '0,1,2')."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_path = QuickstartConfig.get_default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = QuickstartConfig(
            reserved_gpu_device_ids="0,1,2",
        )
        config.save()

        loaded = QuickstartConfig.load()
        assert loaded.reserved_gpu_device_ids == "0,1,2"
        assert config_path.read_text().count("reserved_gpu_device_ids") == 1
        assert "0,1,2" in config_path.read_text() or "'0,1,2'" in config_path.read_text()

    def test_parse_reserved_gpu_device_ids_returns_list(self):
        """parse_reserved_gpu_device_ids() returns list of ints from comma-separated string."""
        config = QuickstartConfig(reserved_gpu_device_ids="0,1,2")
        assert config.parse_reserved_gpu_device_ids() == [0, 1, 2]

    def test_parse_reserved_gpu_device_ids_none_when_unset(self):
        """parse_reserved_gpu_device_ids() returns None when field is None."""
        config = QuickstartConfig(reserved_gpu_device_ids=None)
        assert config.parse_reserved_gpu_device_ids() is None

    def test_parse_reserved_gpu_device_ids_empty_list_when_empty_string(self):
        """parse_reserved_gpu_device_ids() returns [] when field is empty string."""
        config = QuickstartConfig(reserved_gpu_device_ids="")
        assert config.parse_reserved_gpu_device_ids() == []

    def test_parse_reserved_gpu_device_ids_handles_spaces(self):
        """parse_reserved_gpu_device_ids() handles spaces around commas."""
        config = QuickstartConfig(reserved_gpu_device_ids="0, 1, 2")
        assert config.parse_reserved_gpu_device_ids() == [0, 1, 2]

    def test_parse_reserved_gpu_device_ids_returns_none_for_invalid_format(self):
        """parse_reserved_gpu_device_ids() returns None when input contains non-integers."""
        config = QuickstartConfig(reserved_gpu_device_ids="a,b,c")
        assert config.parse_reserved_gpu_device_ids() is None

    def test_parse_reserved_gpu_device_ids_returns_none_for_negative_int(self):
        """parse_reserved_gpu_device_ids() returns None when input contains negative integers."""
        config = QuickstartConfig(reserved_gpu_device_ids="-1")
        assert config.parse_reserved_gpu_device_ids() is None
        config = QuickstartConfig(reserved_gpu_device_ids="0,-1,1")
        assert config.parse_reserved_gpu_device_ids() is None

    def test_parse_reserved_gpu_device_ids_returns_none_for_uuid_like(self):
        """parse_reserved_gpu_device_ids() returns None when input contains UUID-like (non-int) values."""
        config = QuickstartConfig(reserved_gpu_device_ids="GPU-abc-123")
        assert config.parse_reserved_gpu_device_ids() is None

    # Additional invalid/unsupported cases (empty, whitespace, mixed UUID, non-numeric, "all")
    # are covered by test_gpu_config.TestParseCudaVisibleDevicesIntegers.test_returns_none_for_invalid_or_unsupported,
    # which exercises the shared parser parse_comma_separated_non_negative_integers.


class TestValidateConfigReservedGpuDeviceIds:
    """validate_config fails when use_gpu is True but reserved_gpu_device_ids is not set."""

    @pytest.fixture(autouse=True)
    def _patch_validators(self):
        """Patch docker/socket/storage/port validators so only GPU validation is under test."""
        ok = MagicMock(valid=True, message="ok")
        with (
            patch("nemo_helix_ext.quickstart.validators.validate_docker_available", return_value=ok),
            patch("nemo_helix_ext.quickstart.validators.validate_docker_socket", return_value=ok),
            patch("nemo_helix_ext.quickstart.validators.validate_storage_path", return_value=ok),
            patch("nemo_helix_ext.quickstart.validators.validate_port_available", return_value=ok),
        ):
            yield

    def test_validate_config_fails_when_use_gpu_and_no_reserved_gpu_device_ids(self):
        """use_gpu True and reserved_gpu_device_ids unset fails validation."""
        config = QuickstartConfig(use_gpu=True, reserved_gpu_device_ids=None)
        results = validate_config(config)
        failures = [r for r in results if not r.valid]
        assert any("GPU device IDs are not set" in r.message for r in failures)

    def test_validate_config_fails_when_use_gpu_and_reserved_gpu_device_ids_empty(self):
        """use_gpu True and reserved_gpu_device_ids empty string fails validation."""
        config = QuickstartConfig(use_gpu=True, reserved_gpu_device_ids="")
        results = validate_config(config)
        failures = [r for r in results if not r.valid]
        assert any("GPU device IDs are not set" in r.message for r in failures)

    def test_validate_config_fails_when_use_gpu_and_reserved_gpu_device_ids_whitespace(self):
        """use_gpu True and reserved_gpu_device_ids whitespace-only fails validation."""
        config = QuickstartConfig(use_gpu=True, reserved_gpu_device_ids="   ")
        results = validate_config(config)
        failures = [r for r in results if not r.valid]
        assert any("GPU device IDs are not set" in r.message for r in failures)

    def test_validate_config_passes_when_use_gpu_and_reserved_gpu_device_ids_set(self):
        """use_gpu and reserved_gpu_device_ids set passes GPU validation."""
        config = QuickstartConfig(use_gpu=True, reserved_gpu_device_ids="0,1")
        results = validate_config(config)
        gpu_results = [r for r in results if "GPU" in r.message]
        assert any(r.valid and "GPU device IDs are set" in r.message for r in gpu_results)


class TestQuickstartConfigBackwardCompatibility:
    """Tests for loading legacy quickstart config files."""

    def test_load_ignores_legacy_registry_credentials(self, tmp_path: Path):
        """Legacy registry fields should be ignored when loading config."""
        config_path = tmp_path / "nhx" / "quickstart.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "image": "registry.example.com/nhx-api:test",
                    "registry_user": "legacy-user",
                    "registry_password": "legacy-pass",
                }
            )
        )

        config = QuickstartConfig.load(path=config_path)

        assert config.image == "registry.example.com/nhx-api:test"
        assert not hasattr(config, "registry_user")
        assert config.registry_password is None
        assert not config.has_registry_credentials_for_image()


class TestIsInternalTag:
    """Tests for the _is_internal_tag() helper."""

    @pytest.mark.parametrize(
        "tag",
        [
            "nightly-20260223",
            "nightly-20251201",
            "26.02-k10",
            "26.02-k1",
            "25.12-k3",
        ],
    )
    def test_internal_tags(self, tag: str) -> None:
        assert _is_internal_tag(tag) is True

    @pytest.mark.parametrize(
        "tag",
        [
            "26.03",  # public GA release
            "25.10",  # public GA release
            "26.02",  # missing k-suffix
            "latest",
            "",
        ],
    )
    def test_public_or_unknown_tags(self, tag: str) -> None:
        assert _is_internal_tag(tag) is False


class TestResolveBestImage:
    """Tests for QuickstartConfig.resolve_best_image()."""

    NIGHTLY_IMAGE = "nvcr.io/nvidia/platform-api:nightly-20260223"
    MILESTONE_IMAGE = "nvcr.io/nvidia/platform-api:26.02-k10"
    PUBLIC_IMAGE = "nvcr.io/nvidia/nemo-microservices/nhx-api:26.03"

    @pytest.fixture
    def config_with_key(self) -> QuickstartConfig:
        return QuickstartConfig(ngc_api_key="test-key")  # type: ignore[arg-type]

    @pytest.fixture
    def config_without_key(self) -> QuickstartConfig:
        # Explicitly pass None so the fixture is not affected by a NGC_API_KEY
        # environment variable that may be set in the developer's shell.
        return QuickstartConfig(ngc_api_key=None)

    def _mock_sdk(self, image_tag: str) -> MagicMock:
        mock = MagicMock()
        mock.__image_tag__ = image_tag
        return mock

    # ------------------------------------------------------------------
    # Short-circuit: explicit image is always returned as-is
    # ------------------------------------------------------------------

    def test_returns_custom_image_unchanged(self) -> None:
        """If image is explicitly set, skip the lookup entirely."""
        config = QuickstartConfig(image="my-registry/my-image:v1", ngc_api_key="key")  # type: ignore[arg-type]
        assert config.resolve_best_image() == "my-registry/my-image:v1"

    # ------------------------------------------------------------------
    # No tag available
    # ------------------------------------------------------------------

    def test_returns_empty_when_sdk_not_installed(self, config_with_key: QuickstartConfig) -> None:
        """When nemo-helix is not installed, empty string is returned."""
        with patch.dict("sys.modules", {"nemo_helix._version": None}):
            assert config_with_key.resolve_best_image() == ""

    def test_returns_empty_when_image_tag_is_none(self, config_with_key: QuickstartConfig) -> None:
        """When __image_tag__ is None, empty string is returned."""
        with patch.dict("sys.modules", {"nemo_helix._version": self._mock_sdk(None)}):  # type: ignore[arg-type]
            assert config_with_key.resolve_best_image() == ""

    # ------------------------------------------------------------------
    # Internal tags — returned directly without key or access check
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "image_tag,expected_image",
        [
            ("nightly-20260223", "nvcr.io/nvidia/platform-api:nightly-20260223"),
            ("26.02-k10", "nvcr.io/nvidia/platform-api:26.02-k10"),
        ],
    )
    def test_internal_tag_returned_directly(
        self,
        image_tag: str,
        expected_image: str,
        config_without_key: QuickstartConfig,
    ) -> None:
        """Internal tags are returned directly — no NGC key or Docker access check required."""
        with patch.dict("sys.modules", {"nemo_helix._version": self._mock_sdk(image_tag)}):
            assert config_without_key.resolve_best_image() == expected_image

    # ------------------------------------------------------------------
    # Public GA tags — no key required, returned directly
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("image_tag", ["26.03", "25.10"])
    def test_public_tag_returns_empty_without_key(self, image_tag: str, config_without_key: QuickstartConfig) -> None:
        """Public GA tags on nvcr.io still require an NGC key; empty string returned without one."""
        with patch.dict("sys.modules", {"nemo_helix._version": self._mock_sdk(image_tag)}):
            assert config_without_key.resolve_best_image() == ""

    def test_public_tag_returned_with_key(self, config_with_key: QuickstartConfig) -> None:
        """Public GA tags are returned when an NGC key is present."""
        with patch.dict("sys.modules", {"nemo_helix._version": self._mock_sdk("26.03")}):
            assert config_with_key.resolve_best_image() == self.PUBLIC_IMAGE

    # ------------------------------------------------------------------
    # NHX_IMAGE_TAG env var override
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "image_tag,expected_image",
        [
            ("nightly-20260223", "nvcr.io/nvidia/platform-api:nightly-20260223"),
            ("26.02-k10", "nvcr.io/nvidia/platform-api:26.02-k10"),
            ("26.03", "nvcr.io/nvidia/nemo-microservices/nhx-api:26.03"),
        ],
    )
    def test_env_var_overrides_sdk_image_tag(
        self,
        image_tag: str,
        expected_image: str,
        config_with_key: QuickstartConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """NHX_IMAGE_TAG is used instead of the SDK-baked tag; SDK is never imported."""
        monkeypatch.setenv("NHX_IMAGE_TAG", image_tag)
        with patch.dict("sys.modules", {"nemo_helix._version": None}):
            result = config_with_key.resolve_best_image()

        assert result == expected_image

    def test_env_var_takes_precedence_over_sdk(
        self, config_with_key: QuickstartConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NHX_IMAGE_TAG takes precedence over __image_tag__ in the SDK."""
        monkeypatch.setenv("NHX_IMAGE_TAG", "nightly-20260223")
        with patch.dict("sys.modules", {"nemo_helix._version": self._mock_sdk("26.02-k10")}):
            result = config_with_key.resolve_best_image()

        assert result == self.NIGHTLY_IMAGE

    def test_public_env_var_returns_empty_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A public GA tag in NHX_IMAGE_TAG still requires an NGC key for nvcr.io."""
        monkeypatch.setenv("NHX_IMAGE_TAG", "26.03")
        config = QuickstartConfig(ngc_api_key=None)
        with patch.dict("sys.modules", {"nemo_helix._version": None}):
            assert config.resolve_best_image() == ""

    def test_public_env_var_returned_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A public GA tag in NHX_IMAGE_TAG is returned when an NGC key is present."""
        monkeypatch.setenv("NHX_IMAGE_TAG", "26.03")
        config = QuickstartConfig(ngc_api_key="test-key")  # type: ignore[arg-type]
        with patch.dict("sys.modules", {"nemo_helix._version": None}):
            assert config.resolve_best_image() == self.PUBLIC_IMAGE
