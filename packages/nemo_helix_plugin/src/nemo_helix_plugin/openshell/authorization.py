# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that configured provider secret references are readable as metadata."""

from nemo_helix_plugin.entities.base import parse_qualified_name
from nemo_helix_plugin.openshell.types import GatewayProviderBinding
from nemo_helix_plugin.secrets.client import AsyncSecretsClient


async def authorize_provider_references(
    bindings: dict[str, GatewayProviderBinding], *, workspace: str, secrets_client: AsyncSecretsClient
) -> None:
    """Read each secret's metadata with the supplied client's identity, without fetching values.

    This does not separately check the workload creator's permission to read values;
    the owning backend performs that check before signing a registration request.
    """

    for ref in sorted({ref for binding in bindings.values() for ref in binding.credentials.values()}):
        secret_workspace, name = parse_qualified_name(ref, default_workspace=workspace)
        await secrets_client.get_secret(workspace=secret_workspace, name=name)
