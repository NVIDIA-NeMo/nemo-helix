# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate provider references using the caller's typed client, without values."""

from nemo_helix_plugin.entities.base import parse_qualified_name
from nemo_helix_plugin.openshell.types import GatewayProviderBinding
from nemo_helix_plugin.secrets.client import AsyncSecretsClient


async def _require_creator_permission(
    *, principal_id: str, workspace: str, secrets_client: AsyncSecretsClient, service_name: str
) -> None:
    # Platform-only dependencies stay lazy so standalone driver imports do not
    # require a configured Helix runtime. Only trusted controllers issue tickets.
    from nhx.common.auth import AuthClient, Principal
    from nhx.common.config import get_auth_config

    config = get_auth_config()
    if not config.enabled or principal_id.startswith("service:"):
        raise PermissionError("Secret binding requires a configured creator permission check.")
    auth = AuthClient(
        principal=Principal(id=principal_id),
        config=config,
        http_client=secrets_client.http_client,
        service_name=service_name,
    )
    # Check the creator directly. A delegated service's metadata lookup would
    # otherwise succeed using the service wildcard, even after creator revocation.
    if not await auth.has_permissions(workspace, ["secrets.read"]):
        raise PermissionError("The workload creator cannot use this secret.")


async def authorize_secret_binding(
    *,
    workspace: str,
    name: str,
    principal_id: str | None,
    secrets_client: AsyncSecretsClient,
    service_name: str,
) -> None:
    """OpenShell preflight using existing permission and metadata APIs; no values.

    The trusted controller supplies the signed creator identity, or None only
    when platform auth is disabled. Normal Secrets access still enforces the
    driver's and delegated creator's permissions when resolving the credential.
    """
    if principal_id is not None:
        await _require_creator_permission(
            principal_id=principal_id, workspace=workspace, secrets_client=secrets_client, service_name=service_name
        )
    secret = (await secrets_client.get_secret(workspace=workspace, name=name)).data()
    if (secret.workspace, secret.name) != (workspace, name):
        raise PermissionError("Secret metadata does not match the requested binding.")


async def authorize_provider_references(
    bindings: dict[str, GatewayProviderBinding], *, workspace: str, secrets_client: AsyncSecretsClient
) -> None:
    for ref in sorted({ref for binding in bindings.values() for ref in binding.credentials.values()}):
        secret_workspace, name = parse_qualified_name(ref, default_workspace=workspace)
        await secrets_client.get_secret(workspace=secret_workspace, name=name)
