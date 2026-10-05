# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quickstart interactive prompts."""

from unittest.mock import patch

from nemo_helix_ext.quickstart.config import QuickstartConfig
from nemo_helix_ext.quickstart.prompts import (
    _registry_host_from_image,
    prompt_for_configuration,
    prompt_for_registry_credentials,
)
from nemo_helix_ext.quickstart.validators import ValidationResult
from pydantic import SecretStr


def _valid_registry_credentials_patch():
    return patch(
        "nemo_helix_ext.quickstart.validators.validate_registry_credentials",
        return_value=ValidationResult(True, "Registry credentials are valid"),
    )


def test_prompt_for_configuration_validates_new_ngc_key():
    """A newly entered NGC API key is validated immediately and stripped before saving."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="  test-key\n"),
        patch("nemo_helix_ext.quickstart.prompts.prompt_choice", side_effect=["nvidia-build", "yes"]),
        patch(
            "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
            return_value=ValidationResult(True, "NGC credentials are valid"),
        ) as validate_ngc,
    ):
        config = prompt_for_configuration(QuickstartConfig(ngc_api_key=None))

    validate_ngc.assert_called_once_with("test-key")
    assert config.ngc_api_key is not None
    assert config.ngc_api_key.get_secret_value() == "test-key"


def test_prompt_for_configuration_rejects_invalid_new_ngc_key():
    """An invalid newly entered NGC API key stops configure before later prompts."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="bad-key"),
        patch("nemo_helix_ext.quickstart.prompts.prompt_choice") as prompt_choice,
        patch(
            "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
            return_value=ValidationResult(False, "NGC login failed: unauthorized"),
        ),
    ):
        try:
            prompt_for_configuration(QuickstartConfig(ngc_api_key=None))
        except RuntimeError as exc:
            assert "NGC API Key validation failed" in str(exc)
        else:
            raise AssertionError("Expected invalid NGC key to stop configuration")

    prompt_choice.assert_not_called()


def test_prompt_for_configuration_validates_existing_ngc_key_when_kept():
    """Pressing Enter to keep an existing NGC API key still validates it."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value=""),
        patch("nemo_helix_ext.quickstart.prompts.prompt_choice", side_effect=["nvidia-build", "yes"]),
        patch(
            "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
            return_value=ValidationResult(True, "NGC credentials are valid"),
        ) as validate_ngc,
    ):
        config = prompt_for_configuration(QuickstartConfig(ngc_api_key=SecretStr("existing-key")))

    validate_ngc.assert_called_once_with("existing-key")
    assert config.ngc_api_key is not None
    assert config.ngc_api_key.get_secret_value() == "existing-key"


def test_prompt_for_configuration_validates_existing_ngc_key_when_whitespace_entered():
    """Whitespace input keeps and validates the existing NGC API key."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="   "),
        patch("nemo_helix_ext.quickstart.prompts.prompt_choice", side_effect=["nvidia-build", "yes"]),
        patch(
            "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
            return_value=ValidationResult(True, "NGC credentials are valid"),
        ) as validate_ngc,
    ):
        config = prompt_for_configuration(QuickstartConfig(ngc_api_key=SecretStr("existing-key")))

    validate_ngc.assert_called_once_with("existing-key")
    assert config.ngc_api_key is not None
    assert config.ngc_api_key.get_secret_value() == "existing-key"


def test_prompt_for_configuration_rejects_invalid_existing_ngc_key_when_kept():
    """A revoked stored NGC API key stops configure even if the user presses Enter."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value=""),
        patch("nemo_helix_ext.quickstart.prompts.prompt_choice") as prompt_choice,
        patch(
            "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
            return_value=ValidationResult(False, "NGC login failed: unauthorized"),
        ) as validate_ngc,
    ):
        try:
            prompt_for_configuration(QuickstartConfig(ngc_api_key=SecretStr("revoked-key")))
        except RuntimeError as exc:
            assert "NGC API Key validation failed" in str(exc)
        else:
            raise AssertionError("Expected invalid existing NGC key to stop configuration")

    validate_ngc.assert_called_once_with("revoked-key")
    prompt_choice.assert_not_called()


def test_prompt_for_configuration_surfaces_user_pass_registry_token_step():
    """Username/password registry images surface registry credentials during configure and validate the token."""
    image = "ghcr.io/nvidia-nemo/nemo/nhx-api:latest"
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", side_effect=["ngc-key", "registry-token"]),
        patch("nemo_helix_ext.quickstart.prompts.prompt_choice", side_effect=["yes", "nvidia-build", "yes"]),
        patch(
            "nemo_helix_ext.quickstart.prompts.prompt_text",
            side_effect=["ghcr.io", "registry-user"],
        ),
        patch(
            "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
            return_value=ValidationResult(True, "NGC credentials are valid"),
        ),
        _valid_registry_credentials_patch() as validate_registry,
    ):
        config = prompt_for_configuration(QuickstartConfig(image=image, ngc_api_key=None))

    validate_registry.assert_called_once_with("ghcr.io", "registry-user", "registry-token")
    assert config.registry_host == "ghcr.io"
    assert config.registry_username == "registry-user"
    assert config.registry_password is not None
    assert config.registry_password.get_secret_value() == "registry-token"


def test_prompt_for_configuration_surfaces_github_registry_token_step():
    """GitHub Container Registry images surface registry credentials and validate the token."""
    image = "ghcr.io/nvidia-nemo/nhx-api:latest"
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", side_effect=["ngc-key", "github-token"]),
        patch("nemo_helix_ext.quickstart.prompts.prompt_choice", side_effect=["yes", "nvidia-build", "yes"]),
        patch(
            "nemo_helix_ext.quickstart.prompts.prompt_text",
            side_effect=["ghcr.io", "github-user"],
        ),
        patch(
            "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
            return_value=ValidationResult(True, "NGC credentials are valid"),
        ),
        _valid_registry_credentials_patch() as validate_registry,
    ):
        config = prompt_for_configuration(QuickstartConfig(image=image, ngc_api_key=None))

    validate_registry.assert_called_once_with("ghcr.io", "github-user", "github-token")
    assert config.registry_host == "ghcr.io"
    assert config.registry_username == "github-user"
    assert config.registry_password is not None
    assert config.registry_password.get_secret_value() == "github-token"


def test_prompt_for_configuration_can_skip_user_pass_registry_token_step():
    """Users can skip registry auth in configure and use the dedicated command later."""
    image = "ghcr.io/nvidia-nemo/nemo/nhx-api:latest"
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="ngc-key"),
        patch("nemo_helix_ext.quickstart.prompts.prompt_choice", side_effect=["no", "nvidia-build", "yes"]),
        patch(
            "nemo_helix_ext.quickstart.validators.validate_ngc_credentials",
            return_value=ValidationResult(True, "NGC credentials are valid"),
        ),
        _valid_registry_credentials_patch() as validate_registry,
    ):
        prompt_for_configuration(QuickstartConfig(image=image, ngc_api_key=None))

    validate_registry.assert_not_called()


def test_prompt_for_registry_credentials_returns_values():
    """Prompt helper should return registry, username, and secret password."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_text", side_effect=["registry.example.com", "test-user"]),
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="test-pass"),
        _valid_registry_credentials_patch() as validate_registry,
    ):
        credentials = prompt_for_registry_credentials("registry.example.com/repo/image:tag")

    assert credentials.registry == "registry.example.com"
    assert credentials.username == "test-user"
    assert credentials.password.get_secret_value() == "test-pass"
    validate_registry.assert_called_once_with("registry.example.com", "test-user", "test-pass")


def test_prompt_for_registry_credentials_rejects_invalid_token():
    """Invalid registry credentials stop the prompt before retrying image pull."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_text", side_effect=["ghcr.io", "test-user"]),
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="bad-token"),
        patch(
            "nemo_helix_ext.quickstart.validators.validate_registry_credentials",
            return_value=ValidationResult(False, "Registry login failed: unauthorized"),
        ) as validate_registry,
    ):
        try:
            prompt_for_registry_credentials("ghcr.io/nvidia-nemo/image:tag")
        except RuntimeError as exc:
            assert "Registry credential validation failed" in str(exc)
        else:
            raise AssertionError("Expected invalid registry token to stop credential prompt")

    validate_registry.assert_called_once_with("ghcr.io", "test-user", "bad-token")


def test_prompt_for_registry_credentials_uses_default_registry():
    """Prompt helper should pass the provided registry default to prompt_text."""
    with (
        patch(
            "nemo_helix_ext.quickstart.prompts.prompt_text", side_effect=["registry.example.com", "test-user"]
        ) as text_prompt,
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="test-pass"),
        _valid_registry_credentials_patch(),
    ):
        prompt_for_registry_credentials("repo/image:tag", default_registry="registry.example.com")

    first_call = text_prompt.call_args_list[0]
    assert first_call.kwargs["default"] == "registry.example.com"


def test_prompt_for_registry_credentials_two_part_image_no_namespace_as_registry():
    """For 2-part Docker Hub images (namespace/repo:tag), default registry must not be the namespace."""
    with (
        patch(
            "nemo_helix_ext.quickstart.prompts.prompt_text",
            side_effect=["registry-1.docker.io", "user"],
        ) as text_prompt,
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="secret"),
        _valid_registry_credentials_patch(),
    ):
        prompt_for_registry_credentials("ubuntu/mysql:latest", default_registry=None)

    first_call = text_prompt.call_args_list[0]
    assert first_call.kwargs["default"] == ""  # "ubuntu" must not be used as registry default


def test_prompt_for_registry_credentials_nvcr_uses_oauthtoken():
    """When registry is nvcr.io, username should be $oauthtoken and we only prompt for password."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_text", return_value="nvcr.io") as text_prompt,
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="my-ngc-token"),
        _valid_registry_credentials_patch() as validate_registry,
    ):
        credentials = prompt_for_registry_credentials("nvcr.io/nvidia/nemo-microservices/nhx-api:latest")

    assert credentials.registry == "nvcr.io"
    assert credentials.username == "$oauthtoken"
    assert credentials.password.get_secret_value() == "my-ngc-token"
    assert text_prompt.call_count == 1  # Only Registry, not Username
    validate_registry.assert_called_once_with("nvcr.io", "$oauthtoken", "my-ngc-token")


def test_prompt_for_registry_credentials_strips_password_whitespace():
    """Password is stripped of surrounding whitespace (pasted keys often carry a trailing newline)."""
    with (
        patch("nemo_helix_ext.quickstart.prompts.prompt_text", return_value="nvcr.io"),
        patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="  my-ngc-token\n"),
        _valid_registry_credentials_patch() as validate_registry,
    ):
        credentials = prompt_for_registry_credentials("nvcr.io/nvidia/platform-api:nightly-20260310")

    assert credentials.password.get_secret_value() == "my-ngc-token"
    validate_registry.assert_called_once_with("nvcr.io", "$oauthtoken", "my-ngc-token")


# --- _registry_host_from_image ---


def test_registry_host_from_image_empty_or_no_slash():
    """Single-part or empty image has no registry in the reference."""
    assert _registry_host_from_image("") == ""
    assert _registry_host_from_image("mysql") == ""
    assert _registry_host_from_image("nginx:latest") == ""


def test_registry_host_from_image_two_parts_namespace_repo():
    """Docker Hub namespace/repo must not be treated as registry."""
    assert _registry_host_from_image("ubuntu/mysql:latest") == ""
    assert _registry_host_from_image("user/repo") == ""
    assert _registry_host_from_image("library/nginx") == ""


def test_registry_host_from_image_two_parts_with_host():
    """Two-part image with host-like first segment (domain or port) is a registry."""
    assert _registry_host_from_image("registry.example.com/myimage:tag") == "registry.example.com"
    assert _registry_host_from_image("localhost:5000/repo:v1") == "localhost:5000"
    assert _registry_host_from_image("nvcr.io/nvidia/something") == "nvcr.io"


def test_registry_host_from_image_three_or_more_parts():
    """Three+ part image: first segment is the registry host."""
    assert _registry_host_from_image("nvcr.io/nvidia/nemo-microservices/nhx-api:25.10") == "nvcr.io"
    assert _registry_host_from_image("registry.example.com/org/proj/image:tag") == "registry.example.com"
