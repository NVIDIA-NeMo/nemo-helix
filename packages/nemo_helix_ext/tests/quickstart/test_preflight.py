# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quickstart preflight checks."""

from unittest.mock import MagicMock, patch

import pytest
from docker.errors import DockerException
from nemo_helix_ext.quickstart.config import QuickstartConfig
from nemo_helix_ext.quickstart.preflight import CheckStatus, PreflightChecker
from nemo_helix_ext.quickstart.validators import ValidationResult, validate_image_registry_access
from pydantic import SecretStr

USER_PASS_IMAGE = "ghcr.io/nvidia-nemo/nemo/nhx-api:latest"
GITHUB_IMAGE = "ghcr.io/nvidia-nemo/nhx-api:latest"


def _container_image_result(config: QuickstartConfig):
    checker = PreflightChecker(config)
    checker._check_image_pullable()
    assert len(checker.results) == 1
    return checker.results[0], checker._image_pull_required


def _registry_credentials_result(config: QuickstartConfig, *, pull_required: bool | None = True):
    checker = PreflightChecker(config)
    checker._image_pull_required = pull_required
    checker._check_credentials()
    assert len(checker.results) == 1
    return checker.results[0]


def _user_pass_config(image: str) -> QuickstartConfig:
    return QuickstartConfig(
        image=image,
        ngc_api_key=None,
        registry_host="ghcr.io",
        registry_username="test-user",
        registry_password=SecretStr("test-token"),
    )


def _github_config(image: str) -> QuickstartConfig:
    return QuickstartConfig(
        image=image,
        ngc_api_key=None,
        registry_host="ghcr.io",
        registry_username="github-user",
        registry_password=SecretStr("github-token"),
    )


@pytest.fixture
def user_pass_config():
    return _user_pass_config(USER_PASS_IMAGE)


@pytest.fixture
def valid_user_pass_registry_credentials():
    with patch(
        "nemo_helix_ext.quickstart.validators.validate_registry_credentials",
        return_value=ValidationResult(True, "Registry credentials are valid"),
    ) as validate_credentials:
        yield validate_credentials


def test_validate_image_registry_access_inspects_manifest():
    """Manifest validation checks registry access without pulling image layers."""
    completed = MagicMock(returncode=0, stdout="{}", stderr="")

    with patch("nemo_helix_ext.quickstart.validators.subprocess.run", return_value=completed) as run:
        result = validate_image_registry_access("registry.example.com/org/image:tag")

    assert result.valid
    assert result.message == "Image manifest is accessible"
    run.assert_called_once_with(
        ["docker", "manifest", "inspect", "registry.example.com/org/image:tag"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_validate_image_registry_access_reports_manifest_failure():
    """Manifest validation returns the Docker error when access fails."""
    completed = MagicMock(returncode=1, stdout="", stderr="unauthorized")

    with patch("nemo_helix_ext.quickstart.validators.subprocess.run", return_value=completed):
        result = validate_image_registry_access("registry.example.com/org/image:tag")

    assert not result.valid
    assert result.message == "Image manifest check failed: unauthorized"


def test_preflight_treats_private_registry_image_inspection_error_as_pull_required():
    """Private image local-inspection errors defer to the registry credential check."""
    config = QuickstartConfig(
        image=USER_PASS_IMAGE,
        ngc_api_key=None,
    )

    with patch("docker.from_env", side_effect=DockerException("local inspect failed")):
        result, pull_required = _container_image_result(config)

    assert result.status is CheckStatus.PASS
    assert result.message == ("Image ghcr.io/nvidia-nemo/nemo/nhx-api:latest will be pulled on start")
    assert result.details == "Could not inspect local image: local inspect failed"
    assert pull_required is True


def test_preflight_reraises_non_docker_image_inspection_error():
    """Unexpected non-Docker failures during image inspection should surface."""
    config = QuickstartConfig(
        image=USER_PASS_IMAGE,
        ngc_api_key=None,
    )

    with (
        patch("docker.from_env", side_effect=RuntimeError("programming error")),
        pytest.raises(RuntimeError, match="programming error"),
    ):
        _container_image_result(config)


def test_preflight_validates_user_pass_registry_access(user_pass_config, valid_user_pass_registry_credentials):
    """Username/password registries validate stored Docker credentials before pulling."""
    with (
        patch(
            "nemo_helix_ext.quickstart.validators.validate_image_registry_access",
            return_value=ValidationResult(True, "Image manifest is accessible"),
        ) as validate_access,
    ):
        result = _registry_credentials_result(user_pass_config)

    assert result.status is CheckStatus.PASS
    assert result.message == "Registry credentials validated"
    valid_user_pass_registry_credentials.assert_called_once_with("ghcr.io", "test-user", "test-token")
    validate_access.assert_called_once_with(USER_PASS_IMAGE)


def test_preflight_validates_github_registry_access():
    """GitHub Container Registry credentials are validated before pulling."""
    config = _github_config(GITHUB_IMAGE)

    with (
        patch(
            "nemo_helix_ext.quickstart.validators.validate_registry_credentials",
            return_value=ValidationResult(True, "Registry credentials are valid"),
        ) as validate_credentials,
        patch(
            "nemo_helix_ext.quickstart.validators.validate_image_registry_access",
            return_value=ValidationResult(True, "Image manifest is accessible"),
        ) as validate_access,
    ):
        result = _registry_credentials_result(config)

    assert result.status is CheckStatus.PASS
    assert result.message == "Registry credentials validated"
    validate_credentials.assert_called_once_with("ghcr.io", "github-user", "github-token")
    validate_access.assert_called_once_with(GITHUB_IMAGE)


def test_preflight_fails_when_user_pass_registry_access_fails(user_pass_config, valid_user_pass_registry_credentials):
    """Username/password registries fail preflight when Docker cannot access the manifest."""
    with (
        patch(
            "nemo_helix_ext.quickstart.validators.validate_image_registry_access",
            return_value=ValidationResult(False, "Image manifest check failed: unauthorized"),
        ),
    ):
        result = _registry_credentials_result(user_pass_config)

    assert result.status is CheckStatus.FAIL
    assert result.message == "Registry credentials invalid or image is inaccessible"
    assert result.details is not None
    assert "nemo quickstart auth --registry ghcr.io" in result.details
    valid_user_pass_registry_credentials.assert_called_once_with("ghcr.io", "test-user", "test-token")


def test_preflight_fails_when_stored_user_pass_registry_credentials_fail_validation():
    """Stored username/password registry credentials are validated before startup."""
    config = _user_pass_config(USER_PASS_IMAGE)

    with (
        patch(
            "nemo_helix_ext.quickstart.validators.validate_registry_credentials",
            return_value=ValidationResult(False, "Registry login failed: unauthorized"),
        ) as validate_credentials,
        patch("nemo_helix_ext.quickstart.validators.validate_image_registry_access") as validate_access,
    ):
        result = _registry_credentials_result(config)

    assert result.status is CheckStatus.FAIL
    assert result.message == "Registry credentials invalid"
    assert result.details is not None
    assert "nemo quickstart auth --registry ghcr.io" in result.details
    validate_credentials.assert_called_once_with("ghcr.io", "test-user", "test-token")
    validate_access.assert_not_called()


def test_preflight_fails_when_user_pass_registry_credentials_are_not_configured():
    """Username/password registries require quickstart credentials even when host Docker is logged in."""
    config = QuickstartConfig(
        image=USER_PASS_IMAGE,
        ngc_api_key=None,
    )

    with (
        patch("nemo_helix_ext.quickstart.validators.validate_registry_credentials") as validate_credentials,
        patch("nemo_helix_ext.quickstart.validators.validate_image_registry_access") as validate_access,
    ):
        result = _registry_credentials_result(config)

    assert result.status is CheckStatus.FAIL
    assert result.message == "Registry credentials are required"
    assert result.details == "Run 'nemo quickstart auth --registry ghcr.io' and try again."
    validate_credentials.assert_not_called()
    validate_access.assert_not_called()


def test_preflight_validates_ngc_credentials_when_configured():
    """Configured NGC credentials are validated before startup."""
    config = QuickstartConfig(
        image="nvcr.io/nvidia/platform-api:nightly-20260310",
        ngc_api_key=SecretStr("test-ngc-key"),
    )

    with patch(
        "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
        return_value=ValidationResult(True, "NGC credentials are valid"),
    ) as validate_ngc:
        result = _registry_credentials_result(config)

    assert result.status is CheckStatus.PASS
    assert result.message == "NGC credentials configured"
    validate_ngc.assert_called_once_with("test-ngc-key")


def test_preflight_fails_when_ngc_credentials_are_invalid():
    """Revoked NGC credentials fail preflight before Docker pull."""
    config = QuickstartConfig(
        image="nvcr.io/nvidia/platform-api:nightly-20260310",
        ngc_api_key=SecretStr("revoked-ngc-key"),
    )

    with patch(
        "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
        return_value=ValidationResult(False, "NGC login failed: unauthorized"),
    ) as validate_ngc:
        result = _registry_credentials_result(config)

    assert result.status is CheckStatus.FAIL
    assert result.message == "NGC credentials invalid"
    assert result.details == "NGC login failed: unauthorized"
    validate_ngc.assert_called_once_with("revoked-ngc-key")


def test_preflight_validates_user_pass_registry_access_when_pull_requirement_unknown(
    user_pass_config, valid_user_pass_registry_credentials
):
    """If image availability could not be verified, still validate registry access."""
    with (
        patch(
            "nemo_helix_ext.quickstart.validators.validate_image_registry_access",
            return_value=ValidationResult(True, "Image manifest is accessible"),
        ) as validate_access,
    ):
        result = _registry_credentials_result(user_pass_config, pull_required=None)

    assert result.status is CheckStatus.PASS
    assert result.message == "Registry credentials validated"
    valid_user_pass_registry_credentials.assert_called_once_with("ghcr.io", "test-user", "test-token")
    validate_access.assert_called_once_with(USER_PASS_IMAGE)


def test_preflight_skips_manifest_check_for_registry_without_auth_requirement():
    """Generic registries keep the existing no-auth path."""
    config = QuickstartConfig(image="my-registry/nhx-api:local", ngc_api_key=None)

    with patch("nemo_helix_ext.quickstart.validators.validate_image_registry_access") as validate_access:
        result = _registry_credentials_result(config)

    assert result.status is CheckStatus.PASS
    assert result.message == "No registry credentials required"
    validate_access.assert_not_called()


def test_preflight_validates_user_pass_manifest_check_when_image_will_not_be_pulled(
    valid_user_pass_registry_credentials,
):
    """Pinned private images still validate registry access for jobs backend pulls."""
    config = _user_pass_config("ghcr.io/nvidia-nemo/nemo/nhx-api:v2.0.0")

    with (
        patch(
            "nemo_helix_ext.quickstart.validators.validate_image_registry_access",
            return_value=ValidationResult(True, "Image manifest is accessible"),
        ) as validate_access,
    ):
        result = _registry_credentials_result(config, pull_required=False)

    assert result.status is CheckStatus.PASS
    assert result.message == "Registry credentials validated"
    valid_user_pass_registry_credentials.assert_called_once_with("ghcr.io", "test-user", "test-token")
    validate_access.assert_called_once_with(config.image)
