# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the private Helix workspace for gateway-owned credential values."""

import io
import tarfile

from nemo_helix_plugin.auth import is_service_principal_id
from nemo_helix_plugin.auth.client import AsyncAuthenticationClient
from nemo_helix_plugin.client.errors import ConflictError, NemoClientError, NotFoundError
from nemo_helix_plugin.iam.client import AsyncIAMClient
from nemo_helix_plugin.iam.types import AuthzRequest, RoleBindingInput
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nemo_helix_plugin.workspaces.client import AsyncWorkspacesClient
from nemo_helix_plugin.workspaces.types import CreateWorkspaceRequest
from pydantic import BaseModel, ValidationError

OWNED_SECRETS_WORKSPACE = "openshell-internal"


class OwnedWorkspaceError(ValueError):
    """The owned-credential workspace cannot be used privately by this driver."""


class _PrincipalRoles(BaseModel):
    workspaces: dict[str, list[str]]


class _AuthorizationPolicy(BaseModel):
    principals: dict[str, _PrincipalRoles]


class _PolicyBundle(BaseModel):
    authz: _AuthorizationPolicy


async def _private_policy(iam_client: AsyncIAMClient) -> set[str]:
    # Workspace members omit static grants. The bundle includes both sources.
    response = await iam_client.get_opa_bundle()
    with tarfile.open(fileobj=io.BytesIO(await response.read()), mode="r:gz") as archive:
        data = archive.extractfile("data.json")
        if data is None:
            raise OwnedWorkspaceError("Authorization bundle omitted its policy data.")
        policy = _PolicyBundle.model_validate_json(data.read()).authz
    administrators = {
        principal
        for principal, roles in policy.principals.items()
        if principal != "*" and "HelixAdmin" in roles.workspaces.get("system", [])
    }
    if any(
        roles.workspaces.get(OWNED_SECRETS_WORKSPACE)
        and not is_service_principal_id(principal)
        and principal not in administrators
        for principal, roles in policy.principals.items()
    ):
        raise OwnedWorkspaceError("Owned-credential workspace grants access to ordinary principals.")
    return administrators


async def ensure_owned_workspace(secrets_client: AsyncSecretsClient, *, auth_enabled: bool) -> str:
    """Create the storage scope or validate it without changing existing access grants."""
    try:
        return await _ensure_owned_workspace(secrets_client, auth_enabled=auth_enabled)
    except (NemoClientError, ValidationError, tarfile.TarError, KeyError):
        raise OwnedWorkspaceError("Could not validate the owned-credential workspace.") from None


async def _ensure_owned_workspace(secrets_client: AsyncSecretsClient, *, auth_enabled: bool) -> str:
    workspaces_client = AsyncWorkspacesClient.from_client(secrets_client)
    iam_client = AsyncIAMClient.from_client(secrets_client)
    principal: str | None = None
    administrators: set[str] = set()
    if auth_enabled:
        identity = (await AsyncAuthenticationClient.from_client(secrets_client).authenticate_bearer_token_get()).data()
        if (
            not is_service_principal_id(identity.principal)
            or identity.token_kind != "workload_access_token"
            or identity.on_behalf_of is not None
            or (identity.email is not None and identity.email != identity.principal)
        ):
            raise OwnedWorkspaceError("Owned credentials require a direct internal service workload identity.")
        principal = identity.principal
        administrators = await _private_policy(iam_client)

    created = False
    try:
        await workspaces_client.get_workspace(name=OWNED_SECRETS_WORKSPACE)
    except NotFoundError:
        response = await workspaces_client.create_workspace(
            body=CreateWorkspaceRequest(
                name=OWNED_SECRETS_WORKSPACE, description="OpenShell gateway-owned credentials."
            ),
            exist_ok=True,
        )
        created = response.http_response.status_code == 201
    if not auth_enabled:
        return OWNED_SECRETS_WORKSPACE

    members = (await workspaces_client.list_workspace_members(workspace=OWNED_SECRETS_WORKSPACE)).data()
    if any(
        member.roles and not is_service_principal_id(member.principal) and member.principal not in administrators
        for member in members.data
    ):
        raise OwnedWorkspaceError("Owned-credential workspace grants access to ordinary principals.")
    if created:
        # Workspace creation grants Admin, which replaces a service's default
        # ServiceSystem role and does not include Secrets value access.
        assert principal is not None
        try:
            await iam_client.create_role_binding(
                body=RoleBindingInput(principal=principal, workspace=OWNED_SECRETS_WORKSPACE, role="ServiceSystem"),
                query_params={"wait_role_propagation": True},
            )
        except ConflictError:
            pass
    permissions = (
        await iam_client.evaluate_authorization(
            entrypoint="has_permissions",
            body=AuthzRequest(
                input={
                    "principal_id": principal,
                    "workspace": OWNED_SECRETS_WORKSPACE,
                    "permissions": ["secrets.create", "secrets.access", "secrets.update", "secrets.delete"],
                }
            ),
        )
    ).data()
    if permissions.result.get("allowed") is not True:
        raise OwnedWorkspaceError("The internal service lacks owned-credential storage permissions.")
    return OWNED_SECRETS_WORKSPACE
