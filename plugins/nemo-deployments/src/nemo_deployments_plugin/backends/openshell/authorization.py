# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorize deployment provider bindings before signing registration tickets."""

from nemo_helix_plugin.auth import platform_auth_enabled
from nemo_helix_plugin.iam.client import AsyncIAMClient
from nemo_helix_plugin.iam.types import AuthzRequest
from nemo_helix_plugin.secrets.client import AsyncSecretsClient


async def authorize_secret_binding(
    *, workspace: str, name: str, principal_id: str | None, secrets_client: AsyncSecretsClient
) -> None:
    """Check creator permission and metadata without reading the secret value."""
    if principal_id is not None:
        if not platform_auth_enabled() or principal_id.startswith("service:"):
            raise PermissionError("Secret binding requires a configured creator permission check.")
        response = await AsyncIAMClient.from_client(secrets_client).evaluate_authorization(
            entrypoint="has_permissions",
            body=AuthzRequest(
                input={"principal_id": principal_id, "workspace": workspace, "permissions": ["secrets.read"]}
            ),
        )
        if response.data().result.get("allowed") is not True:
            raise PermissionError("The workload creator cannot use this secret.")
    secret = (await secrets_client.get_secret(workspace=workspace, name=name)).data()
    if (secret.workspace, secret.name) != (workspace, name):
        raise PermissionError("Secret metadata does not match the requested binding.")
