# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorize provider bindings through the shared typed policy adapter."""

from nemo_helix_plugin.auth import AuthContext
from nemo_helix_plugin.openshell.authorization import authorize_secret_binding as authorize_binding
from nemo_helix_plugin.secrets.client import AsyncSecretsClient


async def authorize_secret_binding(
    *, workspace: str, name: str, creator: AuthContext | None, secrets_client: AsyncSecretsClient
) -> None:
    await authorize_binding(
        workspace=workspace,
        name=name,
        creator=creator,
        actor_service="deployments",
        secrets_client=secrets_client,
    )
