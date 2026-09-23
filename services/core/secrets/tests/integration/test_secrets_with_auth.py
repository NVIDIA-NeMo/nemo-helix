# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for secrets CRUD operations with authorization enabled.

These tests verify:
- Role-based access control for secrets (Viewer, Editor, Admin, HelixAdmin)
- Viewer can read secret metadata (list, get) but not access value or create/update/delete
- Editor/Admin can create, update, delete secrets but cannot access the secret value
- HelixAdmin can manage secrets but cannot directly access the secret value
- Only service principals can retrieve secret values via /access (delegation pattern)

Uses the create_test_client pattern for fast in-memory testing.
"""

from typing import Generator

import pytest
from nemo_helix_plugin.client.client import NemoClient
from nemo_helix_plugin.client.errors import PermissionDeniedError as ClientPermissionDeniedError
from nemo_helix_plugin.client.errors import UnprocessableEntityError as ClientUnprocessableEntityError
from nemo_helix_plugin.secrets.client import SecretsClient
from nemo_helix_plugin.secrets.types import HelixSecretCreateRequest, HelixSecretUpdateRequest
from nemo_helix_plugin.workspaces.client import WorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceMemberRequest, CreateWorkspaceRequest
from nhx.common.auth.models import Principal
from nhx.core.secrets.config import SecretsServiceConfig
from nhx.core.secrets.service import SecretsService
from nhx.testing import (
    TEST_ADMIN_EMAIL,
    ClientContext,
    create_test_client,
    short_unique_name,
    unique_email,
)
from pydantic import SecretStr, ValidationError

# Service principals have elevated access (like platform admin)
SERVICE_PRINCIPAL = "service:integration-test"


@pytest.fixture
def client(service_config: SecretsServiceConfig) -> Generator[NemoClient, None, None]:
    """Typed client for the in-process SecretsService (auth enabled), without principal headers."""
    with create_test_client(
        SecretsService,
        auth_enabled=True,
        service_configs={SecretsService: service_config},
        client_type=ClientContext,
    ) as ctx:
        yield NemoClient(base_url="http://testserver", http_client=ctx.test_client)


def _as_user(client: NemoClient, principal_id: str, groups: list[str] | None = None) -> NemoClient:
    headers = Principal(
        id=principal_id,
        email=principal_id if "@" in principal_id else None,
        groups=groups or [],
        authz_aliases=[principal_id],
    ).get_headers()
    return client.with_headers(headers)


def _on_behalf_of(client: NemoClient, on_behalf_of: str | Principal) -> NemoClient:
    """Clone *client* with on-behalf-of headers so the delegated principal's permissions are checked."""
    if isinstance(on_behalf_of, str):
        return client.with_headers({"X-NHX-Principal-On-Behalf-Of": on_behalf_of})
    headers = {"X-NHX-Principal-On-Behalf-Of": on_behalf_of.id}
    if on_behalf_of.email:
        headers["X-NHX-Principal-On-Behalf-Of-Email"] = on_behalf_of.email
    if on_behalf_of.groups:
        headers["X-NHX-Principal-On-Behalf-Of-Groups"] = ",".join(on_behalf_of.groups)
    if on_behalf_of.authz_aliases:
        headers["X-NHX-Subject-Aliases"] = ",".join(on_behalf_of.authz_aliases)
    return client.with_headers(headers)


def _secrets(client: NemoClient) -> SecretsClient:
    return SecretsClient.from_client(client)


def _create_workspace(client: NemoClient, name: str) -> None:
    WorkspacesClient.from_client(client).create_workspace(body=CreateWorkspaceRequest(name=name)).data()


def _grant_workspace_role(client: NemoClient, *, workspace: str, principal: str, roles: list[str]) -> None:
    WorkspacesClient.from_client(client).create_workspace_member(
        workspace=workspace,
        body=CreateWorkspaceMemberRequest(principal=principal, roles=roles),
        query_params={"wait_role_propagation": True},
    ).data()


@pytest.mark.integration
class TestSecretsAuthBasics:
    """Basic authorization tests for secrets endpoints."""

    def test_create_secret_without_auth_fails(self, client: NemoClient):
        """Test that creating a secret without auth headers returns 401."""
        secret_name = short_unique_name("noauth")

        # Use raw client to test without auth headers
        response = client._client.post(
            "/apis/secrets/v2/workspaces/default/secrets",
            json={"name": secret_name, "value": "test-value"},
        )

        assert response.status_code == 401

    def test_list_secrets_without_auth_fails(self, client: NemoClient):
        """Test that listing secrets without auth headers returns 401."""
        response = client._client.get("/apis/secrets/v2/workspaces/default/secrets")
        assert response.status_code == 401


@pytest.mark.integration
class TestViewerSecretsAccess:
    """Test that Viewer role can read secret metadata but not modify or access values."""

    def test_viewer_can_list_secrets(self, client: NemoClient):
        """Test that a Viewer can list secrets in the workspace."""
        # Setup: platform admin creates workspace and secret
        workspace_name = short_unique_name("vw-list")
        secret_name = short_unique_name("secret")
        viewer_email = unique_email("viewer")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        platform_admin_secrets = secrets
        platform_admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("secret-value")),
            workspace=workspace_name,
        ).data()
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        # Test: viewer can list secrets
        viewer_client = _as_user(client, viewer_email)
        viewer_secrets = _secrets(viewer_client)
        resp = viewer_secrets.list_secrets(workspace=workspace_name)
        secret_names = [s.name for s in resp.items()]

        assert secret_name in secret_names

    def test_viewer_can_get_secret_metadata(self, client: NemoClient):
        """Test that a Viewer can get secret metadata."""
        workspace_name = short_unique_name("vw-get")
        secret_name = short_unique_name("secret")
        viewer_email = unique_email("viewer")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        platform_admin_secrets = secrets
        platform_admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("secret-value")),
            workspace=workspace_name,
        ).data()
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        viewer_client = _as_user(client, viewer_email)
        viewer_secrets = _secrets(viewer_client)
        secret = viewer_secrets.get_secret(name=secret_name, workspace=workspace_name).data()

        assert secret.name == secret_name
        assert secret.workspace == workspace_name

    def test_viewer_cannot_access_secret_value(self, client: NemoClient):
        """Test that a Viewer cannot access the secret value via /access endpoint."""
        workspace_name = short_unique_name("vw-acc")
        secret_name = short_unique_name("secret")
        viewer_email = unique_email("viewer")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        platform_admin_secrets = secrets
        platform_admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("secret-value")),
            workspace=workspace_name,
        ).data()
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        viewer_client = _as_user(client, viewer_email)
        viewer_secrets = _secrets(viewer_client)
        with pytest.raises(ClientPermissionDeniedError):
            viewer_secrets.access_secret(name=secret_name, workspace=workspace_name)

    def test_viewer_cannot_create_secret(self, client: NemoClient):
        """Test that a Viewer cannot create secrets."""
        workspace_name = short_unique_name("vw-crt")
        viewer_email = unique_email("viewer")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        viewer_client = _as_user(client, viewer_email)
        viewer_secrets = _secrets(viewer_client)
        with pytest.raises(ClientPermissionDeniedError):
            viewer_secrets.create_secret(
                body=HelixSecretCreateRequest(name=short_unique_name("new-sec"), value=SecretStr("should-fail")),
                workspace=workspace_name,
            )

    def test_viewer_cannot_update_secret(self, client: NemoClient):
        """Test that a Viewer cannot update secrets."""
        workspace_name = short_unique_name("vw-upd")
        secret_name = short_unique_name("secret")
        viewer_email = unique_email("viewer")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        platform_admin_secrets = secrets
        platform_admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("original-value")),
            workspace=workspace_name,
        ).data()
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        viewer_client = _as_user(client, viewer_email)
        viewer_secrets = _secrets(viewer_client)
        with pytest.raises(ClientPermissionDeniedError):
            viewer_secrets.update_secret(
                name=secret_name,
                body=HelixSecretUpdateRequest(value=SecretStr("should-fail")),
                workspace=workspace_name,
            )

    def test_viewer_cannot_delete_secret(self, client: NemoClient):
        """Test that a Viewer cannot delete secrets."""
        workspace_name = short_unique_name("vw-del")
        secret_name = short_unique_name("secret")
        viewer_email = unique_email("viewer")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        platform_admin_secrets = secrets
        platform_admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("secret-value")),
            workspace=workspace_name,
        ).data()
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])

        viewer_client = _as_user(client, viewer_email)
        viewer_secrets = _secrets(viewer_client)
        with pytest.raises(ClientPermissionDeniedError):
            viewer_secrets.delete_secret(name=secret_name, workspace=workspace_name)


@pytest.mark.integration
class TestEditorSecretsAccess:
    """Test that Editor role can create, update, delete secrets but not access values."""

    def test_editor_can_create_secret(self, client: NemoClient):
        """Test that an Editor can create secrets."""
        workspace_name = short_unique_name("ed-crt")
        editor_email = unique_email("editor")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        editor_client = _as_user(client, editor_email)
        editor_secrets = _secrets(editor_client)
        secret_name = short_unique_name("ed-sec")
        secret = editor_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("editor-created-secret")),
            workspace=workspace_name,
        ).data()

        assert secret.name == secret_name
        assert secret.workspace == workspace_name

    def test_editor_can_list_secrets(self, client: NemoClient):
        """Test that an Editor can list secrets."""
        workspace_name = short_unique_name("ed-list")
        editor_email = unique_email("editor")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        editor_client = _as_user(client, editor_email)
        editor_secrets = _secrets(editor_client)
        secret_name = short_unique_name("list-sec")
        editor_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("test-data")),
            workspace=workspace_name,
        ).data()

        resp = editor_secrets.list_secrets(workspace=workspace_name)
        secret_names = [s.name for s in resp.items()]

        assert secret_name in secret_names

    def test_editor_can_get_secret_metadata(self, client: NemoClient):
        """Test that an Editor can get secret metadata."""
        workspace_name = short_unique_name("ed-get")
        editor_email = unique_email("editor")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        editor_client = _as_user(client, editor_email)
        editor_secrets = _secrets(editor_client)
        secret_name = short_unique_name("get-sec")
        editor_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("test-data")),
            workspace=workspace_name,
        ).data()

        secret = editor_secrets.get_secret(name=secret_name, workspace=workspace_name).data()
        assert secret.name == secret_name

    def test_editor_can_update_secret(self, client: NemoClient):
        """Test that an Editor can update secrets."""
        workspace_name = short_unique_name("ed-upd")
        editor_email = unique_email("editor")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        editor_client = _as_user(client, editor_email)
        editor_secrets = _secrets(editor_client)
        secret_name = short_unique_name("upd-sec")
        editor_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("original-data")),
            workspace=workspace_name,
        ).data()

        updated = editor_secrets.update_secret(
            name=secret_name,
            body=HelixSecretUpdateRequest(value=SecretStr("updated-data")),
            workspace=workspace_name,
        ).data()
        assert updated.name == secret_name

    def test_editor_can_delete_secret(self, client: NemoClient):
        """Test that an Editor can delete secrets."""
        workspace_name = short_unique_name("ed-del")
        editor_email = unique_email("editor")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        editor_client = _as_user(client, editor_email)
        editor_secrets = _secrets(editor_client)
        secret_name = short_unique_name("del-sec")
        editor_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("to-be-deleted")),
            workspace=workspace_name,
        ).data()

        editor_secrets.delete_secret(name=secret_name, workspace=workspace_name)

        # Verify it's gone
        resp = editor_secrets.list_secrets(workspace=workspace_name)
        secret_names = [s.name for s in resp.items()]
        assert secret_name not in secret_names

    def test_editor_cannot_access_secret_value(self, client: NemoClient):
        """Test that an Editor cannot access the secret value via /access endpoint."""
        workspace_name = short_unique_name("ed-acc")
        editor_email = unique_email("editor")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        editor_client = _as_user(client, editor_email)
        editor_secrets = _secrets(editor_client)
        secret_name = short_unique_name("no-acc")
        editor_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("secret-data")),
            workspace=workspace_name,
        ).data()

        with pytest.raises(ClientPermissionDeniedError):
            editor_secrets.access_secret(name=secret_name, workspace=workspace_name)

    @pytest.mark.skip("Need to add the ability to authenticate as admin for non-workspaced route")
    def test_editor_cannot_rotate_encryption_keys(self, client: NemoClient):
        """Test that an Editor cannot call the rotate encryption keys endpoint."""
        workspace_name = short_unique_name("ed-rot")
        editor_email = unique_email("editor")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])

        editor_client = _as_user(client, editor_email)
        editor_secrets = _secrets(editor_client)

        with pytest.raises(ClientPermissionDeniedError):
            editor_secrets.rotate_encryption_keys().data()


@pytest.mark.integration
class TestAdminSecretsAccess:
    """Test that Admin role has same access as Editor for secrets (no value access)."""

    def test_admin_can_create_secret(self, client: NemoClient):
        """Test that an Admin can create secrets."""
        admin_email = unique_email("admin")
        workspace_name = short_unique_name("adm-crt")

        # Platform admin creates workspace and adds our test admin
        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=admin_email, roles=["Admin"])

        admin_client = _as_user(client, admin_email)
        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("adm-sec")
        secret = admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("admin-created-secret")),
            workspace=workspace_name,
        ).data()

        assert secret.name == secret_name

    def test_admin_can_update_secret(self, client: NemoClient):
        """Test that an Admin can update secrets."""
        admin_email = unique_email("admin")
        workspace_name = short_unique_name("adm-upd")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=admin_email, roles=["Admin"])

        admin_client = _as_user(client, admin_email)
        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("upd-sec")
        admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("original-data")),
            workspace=workspace_name,
        ).data()

        admin_secrets.update_secret(
            name=secret_name,
            body=HelixSecretUpdateRequest(value=SecretStr("updated-by-admin")),
            workspace=workspace_name,
        ).data()

    def test_admin_can_delete_secret(self, client: NemoClient):
        """Test that an Admin can delete secrets."""
        admin_email = unique_email("admin")
        workspace_name = short_unique_name("adm-del")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=admin_email, roles=["Admin"])

        admin_client = _as_user(client, admin_email)
        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("del-sec")
        admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("to-be-deleted")),
            workspace=workspace_name,
        ).data()

        admin_secrets.delete_secret(name=secret_name, workspace=workspace_name)

    def test_admin_cannot_access_secret_value(self, client: NemoClient):
        """Test that an Admin cannot access the secret value via /access endpoint."""
        admin_email = unique_email("admin")
        workspace_name = short_unique_name("adm-acc")

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=admin_email, roles=["Admin"])

        admin_client = _as_user(client, admin_email)
        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("no-acc")
        admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("admin-secret")),
            workspace=workspace_name,
        ).data()

        with pytest.raises(ClientPermissionDeniedError):
            admin_secrets.access_secret(name=secret_name, workspace=workspace_name)


@pytest.mark.integration
class TestHelixAdminSecretsAccess:
    """Test that Platform Admin cannot directly access secret values.

    HelixAdmin is denied direct access to the /access endpoint by OPA policy.
    Secret values must be accessed through the service delegation pattern only.
    """

    def test_platform_admin_cannot_access_secret_value(self, client: NemoClient):
        """Test that a Platform Admin cannot directly access the secret value."""
        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("pa-sec")
        secret_value = "platform-admin-secret-value"

        admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace="default",
        ).data()

        with pytest.raises(ClientPermissionDeniedError):
            admin_secrets.access_secret(name=secret_name, workspace="default")

    def test_platform_admin_can_create_secret_in_any_workspace(self, client: NemoClient):
        """Test that a Platform Admin can create secrets in any workspace."""
        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        workspace_name = short_unique_name("pa-ws")

        _create_workspace(admin_client, workspace_name)

        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("pa-sec")
        secret = admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("secret-in-new-workspace")),
            workspace=workspace_name,
        ).data()

        assert secret.name == secret_name
        assert secret.workspace == workspace_name

    @pytest.mark.skip("Need to add the ability to authenticate as admin for non-workspaced route")
    def test_platform_admin_can_rotate_encryption_keys(self, client: NemoClient):
        """Test that a Platform Admin can call the rotate encryption keys endpoint.

        The rotate-encryption-keys endpoint re-encrypts all secrets with the current
        encryption provider. This is an admin-only operation for key rotation scenarios.
        """
        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(admin_client)
        admin_secrets = secrets

        # Create some secrets to ensure there's data to potentially rotate
        secret_name = short_unique_name("rot-sec")
        admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("secret-for-rotation-test")),
            workspace="default",
        ).data()

        # Call the rotate encryption keys endpoint
        response = _secrets(client).rotate_encryption_keys().data()

        assert response.success is True
        assert response.rotated_secrets >= 1

        # Verify the secret is still accessible after rotation
        result = admin_secrets.access_secret(name=secret_name, workspace="default").data()
        assert result.value == "secret-for-rotation-test"


@pytest.mark.integration
class TestServiceCredentialsSecretsAccess:
    """Test that service credentials can access secret values."""

    def test_service_credentials_can_access_secret_value(self, client: NemoClient):
        """Test that service credentials can access the secret value via /access endpoint."""
        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(admin_client)
        admin_secrets = secrets
        secret_name = short_unique_name("svc-sec")
        secret_value = "service-accessible-secret"

        admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace="default",
        ).data()

        service_client = _as_user(client, SERVICE_PRINCIPAL)
        service_secrets = _secrets(service_client)
        result = service_secrets.access_secret(name=secret_name, workspace="default").data()

        assert result.name == secret_name
        assert result.value == secret_value

    def test_service_credentials_can_list_secrets(self, client: NemoClient):
        """Test that service credentials can list secrets."""
        service_client = _as_user(client, SERVICE_PRINCIPAL)
        service_secrets = _secrets(service_client)

        result = list(service_secrets.list_secrets(workspace="default").items())
        assert result is not None

    def test_service_credentials_can_create_secrets(self, client: NemoClient):
        """Test that service credentials can create secrets."""
        service_client = _as_user(client, SERVICE_PRINCIPAL)
        service_secrets = _secrets(service_client)
        secret_name = short_unique_name("svc-crt")

        secret = service_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("service-created-secret")),
            workspace="default",
        ).data()

        assert secret.name == secret_name


@pytest.mark.integration
class TestSecretDataNotExposed:
    """Test that secret data is never exposed in metadata responses."""

    def test_secret_data_not_in_create_response(self, client: NemoClient):
        """Test that secret data is not returned in create response."""
        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("no-data")

        secret = admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("should-not-be-visible")),
            workspace="default",
        ).data()

        assert secret.name == secret_name

        # Verify via raw HTTP response
        raw_response = client._client.post(
            "/apis/secrets/v2/workspaces/default/secrets",
            json={"name": short_unique_name("raw"), "value": "hidden"},
            headers={"X-NHX-Principal-Id": TEST_ADMIN_EMAIL},
        )
        response_json = raw_response.json()
        assert "data" not in response_json
        assert "_data" not in response_json

    def test_secret_data_not_in_list_response(self, client: NemoClient):
        """Test that secret data is not returned when listing secrets."""
        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("list-no")

        admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("hidden-in-list")),
            workspace="default",
        ).data()

        response = client._client.get(
            "/apis/secrets/v2/workspaces/default/secrets",
            headers={"X-NHX-Principal-Id": TEST_ADMIN_EMAIL},
        )
        response_json = response.json()

        for secret in response_json.get("data", []):
            assert "data" not in secret
            assert "_data" not in secret

    def test_secret_data_not_in_get_response(self, client: NemoClient):
        """Test that secret data is not returned when getting a single secret."""
        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        admin_secrets = _secrets(admin_client)
        secret_name = short_unique_name("get-no")

        admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("hidden-in-get")),
            workspace="default",
        ).data()

        response = client._client.get(
            f"/apis/secrets/v2/workspaces/default/secrets/{secret_name}",
            headers={"X-NHX-Principal-Id": TEST_ADMIN_EMAIL},
        )
        response_json = response.json()

        assert "data" not in response_json
        assert "_data" not in response_json


@pytest.mark.integration
class TestDelegatedSecretAccess:
    """Test delegated secret access using on-behalf-of header.

    These tests verify that when a principal acts on behalf of another user,
    the delegated user's permissions are checked for secret access.

    Note: Viewer role includes secrets.read permission (see static-authz.yaml).
    """

    def test_service_principal_can_access_on_behalf_of_viewer(self, client: NemoClient):
        """Test service principal accessing secret on behalf of a Viewer who has secrets.read.

        Only service principals can call the /access endpoint. HelixAdmin is denied
        direct access by OPA policy, so delegation must use a service principal as caller.
        """
        workspace_name = short_unique_name("del-vw")
        secret_name = short_unique_name("secret")
        secret_value = "delegated-secret-value"
        viewer_email = unique_email("viewer")

        # Setup: create workspace, add viewer, create secret
        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=viewer_email, roles=["Viewer"])
        secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace=workspace_name,
        ).data()

        # Service principal accesses secret on behalf of viewer
        # Viewer has secrets.read permission, so this should succeed
        delegated_client = _on_behalf_of(_as_user(client, SERVICE_PRINCIPAL), viewer_email)
        result = _secrets(delegated_client).access_secret(name=secret_name, workspace=workspace_name).data()

        assert result.name == secret_name
        assert result.value == secret_value

    def test_service_principal_can_access_on_behalf_of_group_bound_viewer(self, client: NemoClient):
        """Test delegated access succeeds when the delegated user's group has Viewer."""
        workspace_name = short_unique_name("del-grp")
        secret_name = short_unique_name("secret")
        secret_value = "delegated-group-secret"
        delegated_email = unique_email("viewer")
        delegated_group = f"group-{short_unique_name('vw')}"

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(
            platform_admin_client, workspace=workspace_name, principal=delegated_group, roles=["Viewer"]
        )
        secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace=workspace_name,
        ).data()

        delegated_client = _on_behalf_of(
            _as_user(client, SERVICE_PRINCIPAL),
            Principal(
                id=delegated_email,
                email=delegated_email,
                groups=[delegated_group],
            ),
        )

        result = _secrets(delegated_client).access_secret(name=secret_name, workspace=workspace_name).data()

        assert result.name == secret_name
        assert result.value == secret_value

    def test_platform_admin_cannot_access_on_behalf_of_non_member(self, client: NemoClient):
        """Test platform admin accessing secret on behalf of a non-member user."""
        workspace_name = short_unique_name("del-nm")
        secret_name = short_unique_name("secret")
        secret_value = "delegated-secret-value"
        non_member_email = unique_email("nonmember")

        # Setup: create workspace and secret, but don't add the user as a member
        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace=workspace_name,
        ).data()

        # Platform admin accesses secret on behalf of non-member
        # Non-member doesn't have any role in workspace, so this should fail
        delegated_client = _on_behalf_of(_as_user(client, TEST_ADMIN_EMAIL), non_member_email)
        with pytest.raises(ClientPermissionDeniedError):
            _secrets(delegated_client).access_secret(name=secret_name, workspace=workspace_name)

    def test_service_principal_denies_on_behalf_of_user_missing_group_bound_role(self, client: NemoClient):
        """Test delegated access fails when the delegated user lacks the bound group."""
        workspace_name = short_unique_name("del-grp-no")
        secret_name = short_unique_name("secret")
        delegated_email = unique_email("viewer")
        bound_group = f"group-{short_unique_name('bound')}"
        other_group = f"group-{short_unique_name('other')}"

        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=bound_group, roles=["Viewer"])
        secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr("delegated-group-secret")),
            workspace=workspace_name,
        ).data()

        delegated_client = _on_behalf_of(
            _as_user(client, SERVICE_PRINCIPAL),
            Principal(
                id=delegated_email,
                email=delegated_email,
                groups=[other_group],
            ),
        )

        with pytest.raises(ClientPermissionDeniedError):
            _secrets(delegated_client).access_secret(name=secret_name, workspace=workspace_name)

    def test_service_principal_can_access_on_behalf_of_editor(self, client: NemoClient):
        """Test service principal accessing secret on behalf of an Editor."""
        workspace_name = short_unique_name("del-ed")
        secret_name = short_unique_name("secret")
        secret_value = "service-delegated-secret"
        editor_email = unique_email("editor")

        # Setup: create workspace, add editor, create secret
        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])
        secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace=workspace_name,
        ).data()

        # Service principal accesses secret on behalf of editor
        # Editor has secrets.read permission (inherits from Viewer), so this should succeed
        delegated_client = _on_behalf_of(_as_user(client, SERVICE_PRINCIPAL), editor_email)
        result = _secrets(delegated_client).access_secret(name=secret_name, workspace=workspace_name).data()

        assert result.name == secret_name
        assert result.value == secret_value

    def test_delegated_access_without_on_behalf_of_uses_caller_permissions(self, client: NemoClient):
        """Test that without on-behalf-of header, caller's own permissions are used.

        HelixAdmin cannot directly access secret values (denied by OPA policy).
        Service principals can access directly without delegation.
        """
        workspace_name = short_unique_name("no-del")
        secret_name = short_unique_name("secret")
        secret_value = "non-delegated-secret"

        # Setup: create workspace and secret
        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        platform_admin_secrets = secrets
        platform_admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace=workspace_name,
        ).data()

        # Platform admin cannot access secret directly (denied by OPA deny rule)
        with pytest.raises(ClientPermissionDeniedError):
            platform_admin_secrets.access_secret(name=secret_name, workspace=workspace_name)

        # Service principal can access directly without delegation
        service_client = _as_user(client, SERVICE_PRINCIPAL)
        result = _secrets(service_client).access_secret(name=secret_name, workspace=workspace_name).data()
        assert result.name == secret_name
        assert result.value == secret_value

    def test_service_can_access_on_behalf_of_service(self, client: NemoClient):
        """Test service principal accessing secret on behalf of another service."""
        workspace_name = short_unique_name("svc-del")
        secret_name = short_unique_name("secret")
        secret_value = "service-to-service-secret"
        other_service = "service:auth"

        # Setup: create workspace and secret
        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace=workspace_name,
        ).data()

        # Service accesses secret on behalf of another service
        # Both services have elevated permissions, so this should succeed
        delegated_client = _on_behalf_of(_as_user(client, SERVICE_PRINCIPAL), other_service)
        result = _secrets(delegated_client).access_secret(name=secret_name, workspace=workspace_name).data()

        assert result.name == secret_name
        assert result.value == secret_value

    def test_editor_cannot_access_on_behalf_of_non_member(self, client: NemoClient):
        """Test that an Editor cannot access secrets on behalf of a non-member."""
        workspace_name = short_unique_name("ed-del")
        secret_name = short_unique_name("secret")
        secret_value = "editor-delegated-secret"
        editor_email = unique_email("editor")
        non_member_email = unique_email("nonmember")

        # Setup: create workspace, add editor, create secret
        platform_admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        secrets = _secrets(platform_admin_client)
        _create_workspace(platform_admin_client, workspace_name)
        _grant_workspace_role(platform_admin_client, workspace=workspace_name, principal=editor_email, roles=["Editor"])
        secrets.create_secret(
            body=HelixSecretCreateRequest(name=secret_name, value=SecretStr(secret_value)),
            workspace=workspace_name,
        ).data()

        # Editor tries to access secret on behalf of non-member
        delegated_client = _on_behalf_of(_as_user(client, editor_email), non_member_email)
        with pytest.raises(ClientPermissionDeniedError):
            _secrets(delegated_client).access_secret(name=secret_name, workspace=workspace_name)


@pytest.mark.integration
class TestSecretNameValidation:
    """Test that secret name validation errors are surfaced correctly.

    Regression tests for: validation errors from entity store should return
    422 (Unprocessable Content) with clear error messages, not 500 (Internal Server Error).
    """

    def test_create_secret_with_uppercase_returns_422(self, client: NemoClient):
        """Test that creating a secret with uppercase letters returns 422.

        Entity names must be DNS-compliant (lowercase letters, digits, hyphens only).
        When a name contains uppercase letters, the entity store validation fails.
        This should return a 422 Unprocessable Content with a clear error message.

        Regression test: Previously returned 500 "Internal authorization error" because
        the validation error from the entity store was caught as an unexpected exception
        in the authorization middleware.
        """

        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        admin_secrets = _secrets(admin_client)

        # Name with uppercase letter - violates DNS-compliant naming rules
        invalid_name = "test-secret-123-Test"

        with pytest.raises(ValidationError) as local_exc:
            HelixSecretCreateRequest(name=invalid_name, value=SecretStr("test-value"))
        assert "should match pattern" in str(local_exc.value).lower()

        # model_construct skips validation so the name still reaches the server.
        with pytest.raises(ClientUnprocessableEntityError) as exc_info:
            admin_secrets.create_secret(
                body=HelixSecretCreateRequest.model_construct(
                    name=invalid_name, value=SecretStr("test-value"), description=None
                ),
                workspace="default",
            )

        # Verify the error message mentions the name validation issue
        error_message = str(exc_info.value)
        assert "should match pattern" in error_message.lower(), (
            f"Expected error message about pattern requirement, got: {error_message}"
        )

    def test_create_secret_with_all_uppercase_returns_422(self, client: NemoClient):
        """Test that creating a secret with all uppercase letters returns 422."""

        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        admin_secrets = _secrets(admin_client)
        invalid_name = "TEST-SECRET"

        with pytest.raises(ValidationError) as local_exc:
            HelixSecretCreateRequest(name=invalid_name, value=SecretStr("test-value"))
        assert "should match pattern" in str(local_exc.value).lower()

        with pytest.raises(ClientUnprocessableEntityError) as exc_info:
            admin_secrets.create_secret(
                body=HelixSecretCreateRequest.model_construct(
                    name=invalid_name, value=SecretStr("test-value"), description=None
                ),
                workspace="default",
            )

        error_message = str(exc_info.value)
        assert "should match pattern" in error_message.lower()

    def test_create_secret_with_valid_name_succeeds(self, client: NemoClient):
        """Test that creating a secret with a valid DNS-compliant name works."""
        admin_client = _as_user(client, TEST_ADMIN_EMAIL)
        admin_secrets = _secrets(admin_client)

        # Valid DNS-compliant name (lowercase, digits, hyphens only)
        valid_name = short_unique_name("valid-secret")

        secret = admin_secrets.create_secret(
            body=HelixSecretCreateRequest(name=valid_name, value=SecretStr("test-value")),
            workspace="default",
        ).data()

        assert secret.name == valid_name
