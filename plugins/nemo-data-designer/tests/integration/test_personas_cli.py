# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from unittest.mock import Mock, patch

import nemo_data_designer_plugin.testing.utils as u
import pytest
from data_designer_nemo.nemotron_personas import WORKSPACE, get_resource_name_for_locale
from nemo_data_designer_plugin.cli import personas as personas_module
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import NemoClient
from nemo_platform_plugin.files.client import FilesClient
from nemo_platform_plugin.files.storage_config import NGCStorageConfig
from nemo_platform_plugin.secrets.client import SecretsClient
from nemo_platform_plugin.secrets.types import PlatformSecretCreateRequest
from pydantic import SecretStr

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_ngc_client() -> Generator[dict[str, Mock]]:
    with (
        patch("nmp.core.files.app.backends.ngc.Client") as mock_client_cls,
        patch("nmp.core.files.app.backends.ngc.ResourceAPI") as mock_resource_api_cls,
    ):
        mock_client = Mock()
        mock_resource_api = Mock()

        # The NGC backend resolves "latest" by calling `list_versions(...)`
        # and reading `.versionId` off the first entry, so the mock must yield
        # a fresh iterator of version-shaped objects on every call (each backend
        # instance re-resolves and consumes the iterator once).
        mock_version = Mock()
        mock_version.versionId = "0.0.2"
        mock_resource_api.list_versions.side_effect = lambda *_a, **_kw: iter([mock_version])

        mock_client_cls.return_value = mock_client
        mock_resource_api_cls.return_value = mock_resource_api

        yield {
            "client": mock_client,
            "resource_api": mock_resource_api,
        }


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch, mock_ngc_client: dict[str, Mock]) -> Generator[NeMoPlatform]:
    with u.make_mock_client_context() as client_context:
        monkeypatch.setenv("NGC_API_KEY", "nvapi-abc123")
        yield client_context.sdk
        monkeypatch.delenv("NGC_API_KEY")


@pytest.fixture
def cli_sdk(monkeypatch: pytest.MonkeyPatch, sdk: NeMoPlatform) -> NeMoPlatform:
    client = client_from_platform(sdk, NemoClient)
    monkeypatch.setattr(personas_module.NemoClient, "from_config", staticmethod(lambda: client))
    return sdk


def test_make_fileset_creates_requested_locale_with_existing_secret(cli_sdk: NeMoPlatform) -> None:
    result = u.invoke_cli(
        [
            "personas",
            "make-fileset",
            "--locale",
            "en_US",
            "--api-key-secret",
            "system/ngc-api-key",
        ]
    )

    assert result.exit_code == 0, result.output
    files = client_from_platform(cli_sdk, FilesClient)
    filesets_page = files.list_filesets(workspace=WORKSPACE)
    assert [fileset.name for fileset in filesets_page.items()] == [get_resource_name_for_locale("en_US")]

    fileset = files.get_fileset(name=get_resource_name_for_locale("en_US"), workspace=WORKSPACE).data()
    assert isinstance(fileset.storage, NGCStorageConfig)
    assert fileset.storage.api_key_secret.root == "system/ngc-api-key"


def test_make_fileset_creates_secret_from_env_then_fileset(
    monkeypatch: pytest.MonkeyPatch, cli_sdk: NeMoPlatform
) -> None:
    monkeypatch.setenv("MY_NGC_API_KEY", "nvapi-from-env")

    result = u.invoke_cli(
        [
            "personas",
            "make-fileset",
            "--locale",
            "en_US",
            "--api-key-secret",
            "system/my-ngc-key",
            "--api-key-env-var",
            "MY_NGC_API_KEY",
        ]
    )

    assert result.exit_code == 0, result.output
    secret = client_from_platform(cli_sdk, SecretsClient).access_secret(name="my-ngc-key", workspace="system").data()
    assert secret.value == "nvapi-from-env"

    files = client_from_platform(cli_sdk, FilesClient)
    fileset = files.get_fileset(name=get_resource_name_for_locale("en_US"), workspace=WORKSPACE).data()
    assert isinstance(fileset.storage, NGCStorageConfig)
    assert fileset.storage.api_key_secret.root == "system/my-ngc-key"


def test_make_fileset_missing_env_var() -> None:
    result = u.invoke_cli(
        [
            "personas",
            "make-fileset",
            "--locale",
            "en_US",
            "--api-key-secret",
            "system/my-ngc-key",
            "--api-key-env-var",
            "MISSING_NGC_API_KEY",
        ]
    )

    assert result.exit_code != 0
    assert "MISSING_NGC_API_KEY" in result.output
    assert "not set or is empty" in result.output


def test_make_fileset_unknown_locale() -> None:
    result = u.invoke_cli(
        [
            "personas",
            "make-fileset",
            "--locale",
            "de_DE",
            "--api-key-secret",
            "system/ngc-api-key",
        ]
    )

    assert result.exit_code != 0
    assert "Invalid value for '--locale'" in result.output
    assert "de_DE" in result.output


def test_make_fileset_bare_secret_name() -> None:
    result = u.invoke_cli(
        [
            "personas",
            "make-fileset",
            "--locale",
            "en_US",
            "--api-key-secret",
            "ngc-api-key",
        ]
    )

    assert result.exit_code != 0
    assert "WORKSPACE/NAME" in result.output


def test_make_fileset_create_secret_conflict_does_not_create_fileset(
    monkeypatch: pytest.MonkeyPatch, cli_sdk: NeMoPlatform
) -> None:
    client_from_platform(cli_sdk, SecretsClient).create_secret(
        workspace="system",
        body=PlatformSecretCreateRequest(name="my-ngc-key", value=SecretStr("nvapi-existing")),
    )
    monkeypatch.setenv("MY_NGC_API_KEY", "nvapi-from-env")

    result = u.invoke_cli(
        [
            "personas",
            "make-fileset",
            "--locale",
            "en_US",
            "--api-key-secret",
            "system/my-ngc-key",
            "--api-key-env-var",
            "MY_NGC_API_KEY",
        ]
    )

    assert result.exit_code == 1
    assert "already exists" in result.output
    files = client_from_platform(cli_sdk, FilesClient)
    assert list(files.list_filesets(workspace=WORKSPACE).items()) == []


def test_make_fileset_create_secret_internal_error_surfaces_clearly(
    monkeypatch: pytest.MonkeyPatch, cli_sdk: NeMoPlatform
) -> None:
    monkeypatch.setenv("MY_NGC_API_KEY", "nvapi-from-env")

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secrets backend exploded")

    # ``create_secret`` is a descriptor whose ``__get__`` rejects class-level access,
    # so patch the typed-client constructor boundary.
    mock_secrets = Mock()
    mock_secrets.create_secret.side_effect = _boom
    with patch.object(personas_module.SecretsClient, "from_client", return_value=mock_secrets):
        result = u.invoke_cli(
            [
                "personas",
                "make-fileset",
                "--locale",
                "en_US",
                "--api-key-secret",
                "system/my-ngc-key",
                "--api-key-env-var",
                "MY_NGC_API_KEY",
            ]
        )

    assert result.exit_code == 1
    assert "Failed to create secret" in result.output
    assert "secrets backend exploded" in result.output
    files = client_from_platform(cli_sdk, FilesClient)
    assert list(files.list_filesets(workspace=WORKSPACE).items()) == []


def test_make_fileset_is_idempotent_when_fileset_already_exists(cli_sdk: NeMoPlatform) -> None:
    # First invocation creates the fileset.
    first = u.invoke_cli(
        [
            "personas",
            "make-fileset",
            "--locale",
            "en_US",
            "--api-key-secret",
            "system/ngc-api-key",
        ]
    )
    assert first.exit_code == 0, first.output
    assert "Created fileset" in first.output

    # Second invocation against the same locale should succeed and report the
    # already-exists path without creating a duplicate fileset.
    second = u.invoke_cli(
        [
            "personas",
            "make-fileset",
            "--locale",
            "en_US",
            "--api-key-secret",
            "system/ngc-api-key",
        ]
    )

    assert second.exit_code == 0, second.output
    assert "already exists" in second.output

    files = client_from_platform(cli_sdk, FilesClient)
    assert [fileset.name for fileset in files.list_filesets(workspace=WORKSPACE).items()] == [
        get_resource_name_for_locale("en_US")
    ]


def test_make_fileset_create_fileset_internal_error_surfaces_clearly(cli_sdk: NeMoPlatform) -> None:
    error_message = "kaboom-fileset-error"

    mock_files = Mock()
    mock_files.create_fileset.side_effect = RuntimeError(error_message)

    with patch.object(personas_module.FilesClient, "from_client", return_value=mock_files):
        result = u.invoke_cli(
            [
                "personas",
                "make-fileset",
                "--locale",
                "en_US",
                "--api-key-secret",
                "system/ngc-api-key",
            ]
        )

    assert result.exit_code == 1
    assert "Failed to create fileset" in result.output
    assert error_message in result.output
    files = client_from_platform(cli_sdk, FilesClient)
    assert list(files.list_filesets(workspace=WORKSPACE).items()) == []
