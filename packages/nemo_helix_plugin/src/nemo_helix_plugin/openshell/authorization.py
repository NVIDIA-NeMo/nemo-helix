# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate provider references using the caller's typed client, without values."""

from nemo_helix_plugin.entities.base import parse_qualified_name
from nemo_helix_plugin.openshell.types import GatewayProviderBinding
from nemo_helix_plugin.secrets.client import AsyncSecretsClient


async def authorize_provider_references(
    bindings: dict[str, GatewayProviderBinding], *, workspace: str, secrets_client: AsyncSecretsClient
) -> None:
    for ref in sorted({ref for binding in bindings.values() for ref in binding.credentials.values()}):
        secret_workspace, name = parse_qualified_name(ref, default_workspace=workspace)
        await secrets_client.get_secret(workspace=secret_workspace, name=name)
