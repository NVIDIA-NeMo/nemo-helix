# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for fileset create + secret resolution with auth enabled."""

from contextlib import contextmanager
from typing import Generator
from unittest.mock import patch

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import NemoHTTPError
from nemo_helix_plugin.files.client import FilesClient
from nemo_helix_plugin.files.types import CreateFilesetRequest
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.core.auth.app.bundle import (
    build_authorization_data as _real_build_authorization_data,
)
from nhx.core.files.app.backends.huggingface import HuggingfaceStorageImpl
from nhx.core.files.service import FilesService
from nhx.core.secrets.service import SecretsService
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    as_user,
    create_test_client,
    grant_workspace_role,
    short_unique_name,
    unique_email,
)
from pydantic import SecretStr


async def _build_authorization_data_without_secrets_read(entities_client=None):
    """Create a test role with Editor writes but without secrets.read/list."""
    data = await _real_build_authorization_data(entities_client)

    data["authz"]["roles"]["ViewerNoSecrets"] = {
        "description": "Viewer without secrets permissions (test-only)",
        "permissions": [
            p for p in data["authz"]["roles"]["Viewer"]["permissions"] if p not in ("secrets.read", "secrets.list")
        ],
    }

    data["authz"]["roles"]["EditorNoSecrets"] = {
        "description": "Editor without secrets permissions (test-only)",
        "includes": ["ViewerNoSecrets"],
        "permissions": list(data["authz"]["roles"]["Editor"]["permissions"]),
    }
    return data


@contextmanager
def patched_authz_data(build_fn):
    """Patch authz data builders used by embedded PDP."""
    with (
        patch("nhx.core.auth.app.bundle.build_authorization_data", side_effect=build_fn),
        patch(
            "nhx.core.auth.app.embedded_pdp.data.build_authorization_data",
            side_effect=build_fn,
        ),
    ):
        yield


@pytest.fixture(scope="module")
def client() -> Generator[NemoClient, None, None]:
    """Auth-enabled test stack with Files + Secrets services."""
    with create_test_client(
        FilesService,
        SecretsService,
        auth_enabled=True,
        client_type=NemoClient,
    ) as client:
        yield client


@pytest.fixture
def no_hf_network(monkeypatch):
    """Fixture to disable live HuggingFace network calls: make validation/config-resolution purely local."""

    async def _validate_noop(self):
        return None

    async def _resolve_passthrough(self):
        # Keep config unchanged; enough for create path testing.
        return self.config

    monkeypatch.setattr(HuggingfaceStorageImpl, "validate_storage", _validate_noop)
    monkeypatch.setattr(HuggingfaceStorageImpl, "resolve_config", _resolve_passthrough)


@pytest.mark.integration
class TestFilesetCreateWithSecretAuth:
    def test_editor_can_create_hf_fileset_with_token_secret(
        self,
        client: NemoClient,
        no_hf_network,
    ):
        workspace = short_unique_name("hf-ok")
        editor_email = unique_email("editor")
        secret_name = short_unique_name("hf-token")
        fileset_name = short_unique_name("fileset")

        admin_client = as_user(client, TEST_ADMIN_EMAIL)
        WorkspacesClient.from_client(admin_client).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        SecretsClient.from_client(admin_client).create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("hf_dummy_token")),
            workspace=workspace,
        )
        grant_workspace_role(
            admin_client,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_client = as_user(client, editor_email)
        files = FilesClient.from_client(editor_client)
        created = files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=fileset_name,
                description="hf fileset",
                storage={
                    "type": "huggingface",
                    "repo_id": "Qwen/Qwen3-0.6B",
                    "repo_type": "model",
                    "token_secret": secret_name,
                },
            ),
        ).data()

        assert created.name == fileset_name
        assert created.storage.type == "huggingface"

    def test_custom_role_without_secrets_read_denied_with_token_secret(
        self,
        client: NemoClient,
        no_hf_network,
    ):
        with patched_authz_data(_build_authorization_data_without_secrets_read):
            workspace = short_unique_name("hf-deny")
            user_email = unique_email("nosecrets")
            secret_name = short_unique_name("hf-token")

            admin_client = as_user(client, TEST_ADMIN_EMAIL)
            WorkspacesClient.from_client(admin_client).create_workspace(
                body=CreateWorkspaceRequest(name=workspace)
            ).data()
            SecretsClient.from_client(admin_client).create_secret(
                body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("hf_dummy_token")),
                workspace=workspace,
            )
            grant_workspace_role(
                admin_client,
                workspace=workspace,
                principal=user_email,
                roles=["EditorNoSecrets"],
            )

            user_client = as_user(client, user_email)
            user_files = FilesClient.from_client(user_client)
            with pytest.raises(NemoHTTPError) as exc_info:
                user_files.create_fileset(
                    workspace=workspace,
                    body=CreateFilesetRequest(
                        name=short_unique_name("fileset"),
                        description="should fail",
                        storage={
                            "type": "huggingface",
                            "repo_id": "Qwen/Qwen3-0.6B",
                            "repo_type": "model",
                            "token_secret": secret_name,
                        },
                    ),
                )

            assert exc_info.value.status_code == 400
            assert "access denied to secret" in str(exc_info.value).lower()

    def test_missing_secret_returns_secret_not_found_error(
        self,
        client: NemoClient,
        no_hf_network,
    ):
        workspace = short_unique_name("hf-miss")
        editor_email = unique_email("editor")

        admin_client = as_user(client, TEST_ADMIN_EMAIL)
        WorkspacesClient.from_client(admin_client).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        grant_workspace_role(
            admin_client,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_client = as_user(client, editor_email)
        editor_files = FilesClient.from_client(editor_client)
        with pytest.raises(NemoHTTPError) as exc_info:
            editor_files.create_fileset(
                workspace=workspace,
                body=CreateFilesetRequest(
                    name=short_unique_name("fileset"),
                    description="missing secret",
                    storage={
                        "type": "huggingface",
                        "repo_id": "Qwen/Qwen3-0.6B",
                        "repo_type": "model",
                        "token_secret": "does-not-exist",
                    },
                ),
            )

        assert exc_info.value.status_code == 400
        assert "secret not found" in str(exc_info.value).lower()

    def test_public_hf_without_token_secret_succeeds(
        self,
        client: NemoClient,
        no_hf_network,
    ):
        workspace = short_unique_name("hf-public")
        editor_email = unique_email("editor")
        fileset_name = short_unique_name("fileset")

        admin_client = as_user(client, TEST_ADMIN_EMAIL)
        WorkspacesClient.from_client(admin_client).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        grant_workspace_role(
            admin_client,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_client = as_user(client, editor_email)
        editor_files = FilesClient.from_client(editor_client)
        created = editor_files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=fileset_name,
                description="no token",
                storage={
                    "type": "huggingface",
                    "repo_id": "Qwen/Qwen3-0.6B",
                    "repo_type": "model",
                    # no token_secret
                },
            ),
        ).data()

        assert created.name == fileset_name
        assert created.storage.type == "huggingface"

    def test_editor_can_list_files_from_hf_fileset_with_token_secret(
        self,
        client: NemoClient,
        no_hf_network,
        monkeypatch,
    ):
        workspace = short_unique_name("hf-read")
        editor_email = unique_email("editor")
        secret_name = short_unique_name("hf-token")
        fileset_name = short_unique_name("fileset")

        async def _list_files_noop(self, path=None):
            return []

        monkeypatch.setattr(HuggingfaceStorageImpl, "list_files", _list_files_noop)

        admin_client = as_user(client, TEST_ADMIN_EMAIL)
        WorkspacesClient.from_client(admin_client).create_workspace(body=CreateWorkspaceRequest(name=workspace)).data()
        SecretsClient.from_client(admin_client).create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("hf_dummy_token")),
            workspace=workspace,
        )
        grant_workspace_role(
            admin_client,
            workspace=workspace,
            principal=editor_email,
            roles=["Editor"],
        )

        editor_client = as_user(client, editor_email)
        editor_files = FilesClient.from_client(editor_client)
        editor_files.create_fileset(
            workspace=workspace,
            body=CreateFilesetRequest(
                name=fileset_name,
                description="hf fileset read test",
                storage={
                    "type": "huggingface",
                    "repo_id": "Qwen/Qwen3-0.6B",
                    "repo_type": "model",
                    "token_secret": secret_name,
                },
            ),
        )

        files = editor_files.list_files(workspace=workspace, name=fileset_name).data()
        assert files.data == []
