# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for quickstart CLI commands."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from nemo_helix_ext.cli.app import app
from nemo_helix_ext.cli.commands.quickstart.cli import (
    _is_docker_desktop_sign_in_required_error,
    _is_docker_registry_auth_error,
    _preflight_needs_registry_credentials_prompt,
    _should_prompt_for_pull_credentials,
    handle_errors,
)
from nemo_helix_ext.quickstart import QuickstartConfig
from nemo_helix_ext.quickstart.preflight import CheckStatus, PreflightResult
from nemo_helix_ext.quickstart.validators import ValidationResult
from typer.testing import CliRunner

runner = CliRunner()


def _normalize_rich(text: str) -> str:
    """Strip ANSI and collapse all whitespace for comparison.

    Collapses every run of whitespace (including newlines from Rich wrapping)
    to a single space. Removes spaces inside the config file path (paths have
    no spaces; Rich wrap can insert them).
    """
    no_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    normalized = re.sub(r"\s+", " ", no_ansi).strip()
    # Paths contain no spaces; any space in "• Configuration file: ..." is from wrapping
    normalized = re.sub(
        r"(• Configuration file: )(.+)$",
        lambda m: m.group(1) + m.group(2).replace(" ", ""),
        normalized,
    )
    return normalized


def _assert_exit_code(result, expected_code: int) -> None:
    assert result.exit_code == expected_code, (
        f"Expected exit code {expected_code}, got {result.exit_code}. "
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )


def test_preflight_needs_registry_credentials_prompt_only_for_registry_failures() -> None:
    """Only registry credential preflight failures should trigger the credential prompt."""
    assert _preflight_needs_registry_credentials_prompt(
        [
            PreflightResult(
                name="Registry Credentials",
                status=CheckStatus.FAIL,
                message="Registry credentials invalid or image is inaccessible",
            )
        ]
    )
    assert not _preflight_needs_registry_credentials_prompt(
        [
            PreflightResult(
                name="Container Image",
                status=CheckStatus.FAIL,
                message="Cannot verify container image",
            )
        ]
    )
    assert not _preflight_needs_registry_credentials_prompt(
        [
            PreflightResult(
                name="Registry Credentials",
                status=CheckStatus.PASS,
                message="Registry credentials validated",
            )
        ]
    )
    assert not _preflight_needs_registry_credentials_prompt(
        [
            PreflightResult(
                name="Registry Credentials",
                status=CheckStatus.FAIL,
                message="NGC API key required for nvcr.io images",
            )
        ]
    )


def test_docker_desktop_sign_in_required_error_detection() -> None:
    """Docker Desktop org sign-in errors should not trigger registry credential prompts."""
    assert _is_docker_desktop_sign_in_required_error(
        Exception(
            '407 Client Error: Proxy Authentication Required ("Sign in to continue using Docker Desktop. '
            'Membership in the organization is required. Sign in enforced by your administrators (via registry.json).")'
        )
    )
    assert not _is_docker_desktop_sign_in_required_error(Exception("403 Client Error: Forbidden"))


def test_docker_registry_auth_error_detection() -> None:
    """Registry auth errors are separated from generic pull failures."""
    assert _is_docker_registry_auth_error(Exception("401 Client Error: Unauthorized"))
    assert _is_docker_registry_auth_error(Exception("403 Client Error: Forbidden"))
    assert _is_docker_registry_auth_error(Exception("Error response from daemon: status code 401"))
    assert _is_docker_registry_auth_error(Exception("pull access denied for ghcr.io/nvidia-nemo/image"))
    assert _is_docker_registry_auth_error(Exception("unauthorized: authentication required"))
    assert not _is_docker_registry_auth_error(Exception("connection reset by peer"))
    assert not _is_docker_registry_auth_error(
        Exception("failed to pull ghcr.io/nvidia-nemo/image-401:latest: connection reset by peer")
    )
    assert not _is_docker_registry_auth_error(
        Exception("failed to pull ghcr.io/nvidia-nemo/403-image:latest: context deadline exceeded")
    )


def test_pull_credentials_prompt_requires_auth_error_and_userpass_registry() -> None:
    """Pull retry credential prompts only apply to username/password registries."""
    userpass_config = QuickstartConfig(image="ghcr.io/nvidia-nemo/image:tag")
    ngc_config = QuickstartConfig(image="nvcr.io/nvidia/nemo-microservices/nhx-api:latest")

    assert _should_prompt_for_pull_credentials(Exception("403 Client Error: Forbidden"), userpass_config)
    assert not _should_prompt_for_pull_credentials(Exception("connection reset by peer"), userpass_config)
    assert not _should_prompt_for_pull_credentials(Exception("403 Client Error: Forbidden"), ngc_config)


def test_quickstart_error_wrapper_preserves_typer_exit(capsys: pytest.CaptureFixture[str]) -> None:
    """Clean command exits should not be reprinted as generic Error: 1 messages."""

    @handle_errors
    def exits_cleanly() -> None:
        raise typer.Exit(code=1)

    with pytest.raises(typer.Exit):
        exits_cleanly()

    captured = capsys.readouterr()
    assert "Error: 1" not in captured.err
    assert "Error: 1" not in captured.out


class TestQuickstartConfigureAutoTokenValidation:
    """Auto configure validates the NGC token before persisting it."""

    def test_auto_validates_ngc_api_key_from_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv("NGC_API_KEY", "test-key")

        with patch(
            "nemo_helix_ext.quickstart.validate_ngc_credentials",
            return_value=ValidationResult(True, "NGC credentials are valid"),
        ) as validate_ngc:
            result = runner.invoke(app, ["quickstart", "configure", "--auto"])

        _assert_exit_code(result, 0)
        validate_ngc.assert_called_once_with("test-key")
        assert "NGC API Key validated" in result.stderr

    def test_auto_rejects_invalid_ngc_api_key_from_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        monkeypatch.setenv("NGC_API_KEY", "bad-key")

        with patch(
            "nemo_helix_ext.quickstart.validate_ngc_credentials",
            return_value=ValidationResult(False, "NGC login failed: unauthorized"),
        ):
            result = runner.invoke(app, ["quickstart", "configure", "--auto"])

        _assert_exit_code(result, 1)
        assert "NGC login failed: unauthorized" in result.stderr


class TestQuickstartRegistryAuthCommand:
    """Dedicated registry auth command validates entered GitLab tokens immediately."""

    def test_auth_validates_registry_token_from_options(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        with (
            patch(
                "nemo_helix_ext.quickstart.validators.validate_registry_credentials",
                return_value=ValidationResult(True, "Registry credentials are valid"),
            ) as validate_registry,
            patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="test-token"),
        ):
            result = runner.invoke(
                app,
                [
                    "quickstart",
                    "auth",
                    "--registry",
                    "ghcr.io",
                    "--username",
                    "test-user",
                    "--token",
                    "-",
                ],
            )

        _assert_exit_code(result, 0)
        validate_registry.assert_called_once_with("ghcr.io", "test-user", "test-token")
        assert "quickstart registry auth updated" in result.stderr
        config = QuickstartConfig.load()
        assert config.registry_host == "ghcr.io"
        assert config.registry_username == "test-user"
        assert config.registry_password is not None
        assert config.registry_password.get_secret_value() == "test-token"

    def test_auth_rejects_inline_registry_token(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        result = runner.invoke(
            app,
            [
                "quickstart",
                "auth",
                "--registry",
                "ghcr.io",
                "--username",
                "test-user",
                "--token",
                "test-token",
            ],
        )

        _assert_exit_code(result, 1)
        assert "For safety, omit --token or use --token - to prompt securely." in result.stderr

    def test_auth_rejects_invalid_registry_token(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        with (
            patch(
                "nemo_helix_ext.quickstart.validators.validate_registry_credentials",
                return_value=ValidationResult(False, "Registry login failed: unauthorized"),
            ),
            patch("nemo_helix_ext.quickstart.prompts.prompt_password", return_value="bad-token"),
        ):
            result = runner.invoke(
                app,
                [
                    "quickstart",
                    "auth",
                    "--registry",
                    "ghcr.io",
                    "--username",
                    "test-user",
                    "--token",
                    "-",
                ],
            )

        _assert_exit_code(result, 1)
        assert "Registry login failed: unauthorized" in result.stderr


class TestQuickstartUpCredentialPromptOnPullFailure:
    """Pull failure should prompt for credentials regardless of whether --image was passed."""

    @pytest.mark.skip(
        reason="Crashes under pytest-xdist (worker 'node down: Not properly terminated'); Rich/terminal usage in quickstart CLI."
    )
    def test_up_prompts_for_credentials_when_pull_fails_without_image_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """nemo quickstart up should prompt for registry credentials on pull failure even without --image."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_path = QuickstartConfig.get_default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("image: nvcr.io/nvidia/platform-api:nightly-20260310\n")

        pull_call_count = 0

        def fake_pull_with_progress(self, auth_override=None):
            nonlocal pull_call_count
            pull_call_count += 1
            if pull_call_count == 1:
                raise Exception("403 Client Error: Forbidden")
            return iter([])

        from pydantic import SecretStr

        with (
            patch(
                "nemo_helix_ext.quickstart.container.ContainerManager.pull_image_with_progress",
                fake_pull_with_progress,
            ),
            patch(
                "nemo_helix_ext.quickstart.container.ContainerManager.is_image_available",
                return_value=False,
            ),
            patch(
                "nemo_helix_ext.quickstart.prompts.prompt_for_registry_credentials",
                return_value=("nvcr.io", "$oauthtoken", SecretStr("test-key")),
            ) as mock_prompt,
            patch("nemo_helix_ext.quickstart.is_interactive", return_value=True),
            patch("nemo_helix_ext.quickstart.QuickstartCluster.preflight", return_value=[]),
            patch(
                "nemo_helix_ext.quickstart.container.ContainerManager.start",
                return_value=MagicMock(),
            ),
        ):
            runner.invoke(app, ["quickstart", "up", "--skip-preflight"])

        mock_prompt.assert_called_once()


class TestQuickstartUpRequiresReservedGpuDeviceIdsWhenHostGpu:
    """When use_gpu is True, reserved_gpu_device_ids must be set (no backwards compatibility)."""

    def test_up_fails_when_use_gpu_and_no_reserved_gpu_device_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """nemo quickstart up exits with 1 and tells user to re-run configure when host-gpu but no GPU list."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_path = QuickstartConfig.get_default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # Old/incompatible config: host-gpu enabled but reserved_gpu_device_ids never set
        config_path.write_text("image: nvcr.io/nvidia/nhx-api:latest\ninference_provider: host-gpu\nuse_gpu: true\n")

        result = runner.invoke(app, ["quickstart", "up", "--skip-preflight"])

        _assert_exit_code(result, 1)
        stderr = _normalize_rich(result.stderr)
        assert "GPU device IDs are not set" in stderr
        assert "nemo quickstart configure" in stderr

    @pytest.mark.skip(
        reason="Crashes under pytest-xdist (worker 'node down: Not properly terminated'); Rich/terminal usage in quickstart CLI."
    )
    def test_up_does_not_fail_with_configure_message_when_reserved_gpu_device_ids_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """nemo quickstart up does not fail with 're-run configure' when reserved_gpu_device_ids is set."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        config_path = QuickstartConfig.get_default_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "image: nvcr.io/nvidia/nhx-api:latest\n"
            "inference_provider: host-gpu\n"
            "use_gpu: true\n"
            'reserved_gpu_device_ids: "0,1"\n'
        )

        with patch("nemo_helix_ext.quickstart.QuickstartCluster") as mock_cluster_class:
            mock_cluster = MagicMock()
            mock_cluster_class.return_value = mock_cluster
            mock_cluster.preflight.return_value = []
            mock_cluster._preflight_checker.has_failures.return_value = False
            mock_cluster._preflight_checker.is_already_running.return_value = False
            mock_cluster._preflight_checker.has_warnings.return_value = False
            result = runner.invoke(app, ["quickstart", "up", "--skip-preflight", "--no-pull"])

        stderr = _normalize_rich(result.stderr)
        # Should not fail with the "re-run configure" message (may fail later for Docker etc.)
        assert "GPU device IDs are not set" not in stderr
