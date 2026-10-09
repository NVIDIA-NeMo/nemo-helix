# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E auth tests for the secrets service.

Non-auth secret tests (CRUD, duplicate detection, value non-exposure, etc.)
have been migrated to nemo-helix (e2e/test_secrets.py). Only auth-enabled
tests remain here.
"""

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from nhx.testing import as_user, grant_workspace_role, short_unique_name, unique_email
from pydantic import SecretStr

pytestmark = [pytest.mark.timeout(1800)]

ON_BEHALF_OF_HEADER = "X-NHX-Principal-On-Behalf-Of"


def _create_secret(secrets: SecretsClient, workspace: str, name: str, value: str) -> None:
    secrets.create_secret(
        workspace=workspace,
        body=HelixSecretCreateRequest(name=name, value=SecretStr(value)),
    )


def _secret_names(secrets: SecretsClient, workspace: str) -> list[str]:
    return [s.name for s in secrets.list_secrets(workspace=workspace).items()]


# ============================================================================
# Auth-enabled tests for secrets access control
# These tests require the docker_auth_enabled configuration
# ============================================================================


@pytest.mark.feature("auth")
class TestSecretsAccessControl:
    """Tests for secrets access control with authentication enabled.

    These tests verify that:
    - Viewers can read secret metadata but cannot access values
    - Editors can create/update/delete secrets but cannot access values
    - Only service principals can access secret values
    """

    def test_viewer_can_list_secrets(self, client: NemoClient):
        """Test that a Viewer can list secrets in a workspace."""
        admin_email = unique_email("admin")
        viewer_email = unique_email("viewer")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        admin_secrets = SecretsClient.from_client(admin)
        viewer_secrets = SecretsClient.from_client(as_user(client, viewer_email))
        workspace_name = short_unique_name("viewer-list")
        secret_name = short_unique_name("secret")

        # Admin creates workspace and secret
        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        _create_secret(admin_secrets, workspace_name, secret_name, "secret-value")

        # Admin adds viewer
        grant_workspace_role(admin, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        # Viewer can list secrets
        assert secret_name in _secret_names(viewer_secrets, workspace_name)

        # Clean up
        admin_secrets.delete_secret(workspace=workspace_name, name=secret_name)
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_viewer_can_retrieve_secret_metadata(self, client: NemoClient):
        """Test that a Viewer can retrieve secret metadata."""
        admin_email = unique_email("admin")
        viewer_email = unique_email("viewer")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        admin_secrets = SecretsClient.from_client(admin)
        viewer_secrets = SecretsClient.from_client(as_user(client, viewer_email))
        workspace_name = short_unique_name("viewer-get")
        secret_name = short_unique_name("secret")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        _create_secret(admin_secrets, workspace_name, secret_name, "secret-value")
        grant_workspace_role(admin, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        # Viewer can retrieve secret metadata
        secret = viewer_secrets.get_secret(name=secret_name, workspace=workspace_name).data()
        assert secret.name == secret_name

        # Clean up
        admin_secrets.delete_secret(workspace=workspace_name, name=secret_name)
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_viewer_cannot_access_secret_value(self, client: NemoClient):
        """Test that a Viewer cannot access the secret value via /access endpoint."""
        admin_email = unique_email("admin")
        viewer_email = unique_email("viewer")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        admin_secrets = SecretsClient.from_client(admin)
        viewer_secrets = SecretsClient.from_client(as_user(client, viewer_email))
        workspace_name = short_unique_name("viewer-acc")
        secret_name = short_unique_name("secret")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        _create_secret(admin_secrets, workspace_name, secret_name, "secret-value")
        grant_workspace_role(admin, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        # Viewer should NOT be able to access the secret value
        with pytest.raises(PermissionDeniedError):
            viewer_secrets.access_secret(name=secret_name, workspace=workspace_name)

        # Clean up
        admin_secrets.delete_secret(workspace=workspace_name, name=secret_name)
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_viewer_cannot_create_secret(self, client: NemoClient):
        """Test that a Viewer cannot create secrets."""
        admin_email = unique_email("admin")
        viewer_email = unique_email("viewer")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        viewer_secrets = SecretsClient.from_client(as_user(client, viewer_email))
        workspace_name = short_unique_name("viewer-crt")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        with pytest.raises(PermissionDeniedError):
            _create_secret(viewer_secrets, workspace_name, short_unique_name("new-secret"), "should-fail")

        # Clean up
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_editor_can_create_and_manage_secrets(self, client: NemoClient):
        """Test that an Editor can create, update, and delete secrets."""
        admin_email = unique_email("admin")
        editor_email = unique_email("editor")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        editor_secrets = SecretsClient.from_client(as_user(client, editor_email))
        workspace_name = short_unique_name("editor-crud")
        secret_name = short_unique_name("secret")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        # Editor can create secrets
        secret = editor_secrets.create_secret(
            workspace=workspace_name,
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("editor-created")),
        ).data()
        assert secret.name == secret_name

        # Editor can update secrets
        editor_secrets.update_secret(
            workspace=workspace_name,
            name=secret_name,
            body=HelixSecretUpdateRequest(value=SecretStr("editor-updated")),
        )

        # Editor can delete secrets
        editor_secrets.delete_secret(workspace=workspace_name, name=secret_name)

        # Verify deleted
        assert secret_name not in _secret_names(editor_secrets, workspace_name)

        # Clean up
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_editor_cannot_access_secret_value(self, client: NemoClient):
        """Test that an Editor cannot access the secret value via /access endpoint."""
        admin_email = unique_email("admin")
        editor_email = unique_email("editor")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        editor_secrets = SecretsClient.from_client(as_user(client, editor_email))
        workspace_name = short_unique_name("editor-acc")
        secret_name = short_unique_name("secret")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        grant_workspace_role(admin, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        # Editor creates a secret
        _create_secret(editor_secrets, workspace_name, secret_name, "editor-secret")

        # Editor should NOT be able to access the secret value
        with pytest.raises(PermissionDeniedError):
            editor_secrets.access_secret(name=secret_name, workspace=workspace_name)

        # Clean up
        editor_secrets.delete_secret(workspace=workspace_name, name=secret_name)
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_admin_cannot_access_secret_value(self, client: NemoClient):
        """Test that a workspace Admin cannot access the secret value.

        Only Platform Admin and service credentials can access secret values,
        not workspace-level Admin roles.
        """
        admin_email = unique_email("admin")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        admin_secrets = SecretsClient.from_client(admin)
        workspace_name = short_unique_name("admin-acc")
        secret_name = short_unique_name("secret")

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        _create_secret(admin_secrets, workspace_name, secret_name, "admin-secret")

        # Workspace Admin should NOT be able to access the secret value
        with pytest.raises(PermissionDeniedError):
            admin_secrets.access_secret(name=secret_name, workspace=workspace_name)

        # Clean up
        admin_secrets.delete_secret(workspace=workspace_name, name=secret_name)
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_service_principal_can_access_secret_value(self, client: NemoClient):
        """Test that a service principal can access the secret value.

        Service principals (principals starting with 'service:') have elevated
        permissions including secrets.access.
        """
        admin_email = unique_email("admin")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        admin_secrets = SecretsClient.from_client(admin)
        service_secrets = SecretsClient.from_client(as_user(client, "service:e2e-test"))
        workspace_name = short_unique_name("svc-acc")
        secret_name = short_unique_name("secret")
        secret_value = "service-accessible-secret"

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        _create_secret(admin_secrets, workspace_name, secret_name, secret_value)

        # Service principal can access the secret value
        result = service_secrets.access_secret(name=secret_name, workspace=workspace_name).data()
        assert result.name == secret_name
        assert result.value == secret_value

        # Clean up
        admin_secrets.delete_secret(workspace=workspace_name, name=secret_name)
        admin_workspaces.delete_workspace(name=workspace_name)

    def test_service_can_access_secret_if_viewer_can_get_secret_metadata(self, client: NemoClient):
        """Test that a service principal can access a secret if the user can.

        This test verifies that service principals calling on behalf of a user
        can access secrets that the user has access to.
        """

        admin_email = unique_email("admin")
        viewer_email = unique_email("viewer")
        other_email = unique_email("other")
        admin = as_user(client, admin_email)
        admin_workspaces = WorkspacesClient.from_client(admin)
        admin_secrets = SecretsClient.from_client(admin)
        service_secrets = SecretsClient.from_client(as_user(client, "service:e2e-test-on-behalf"))
        workspace_name = short_unique_name("svc-user-acc")
        secret_name = short_unique_name("secret")
        secret_value = "service-on-behalf-secret"

        admin_workspaces.create_workspace(body=CreateWorkspaceRequest(name=workspace_name))
        _create_secret(admin_secrets, workspace_name, secret_name, secret_value)
        grant_workspace_role(admin, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        # Service principal acting on behalf of viewer can access the secret
        delegated_viewer_secrets = service_secrets.with_headers({ON_BEHALF_OF_HEADER: viewer_email})
        result = delegated_viewer_secrets.access_secret(name=secret_name, workspace=workspace_name).data()
        assert result.name == secret_name
        assert result.value == secret_value

        # Verify that service principal acting on behalf of a user without access is denied
        delegated_other_secrets = service_secrets.with_headers({ON_BEHALF_OF_HEADER: other_email})
        with pytest.raises(PermissionDeniedError):
            delegated_other_secrets.access_secret(name=secret_name, workspace=workspace_name)

        # Clean up
        admin_secrets.delete_secret(workspace=workspace_name, name=secret_name)
        admin_workspaces.delete_workspace(name=workspace_name)
