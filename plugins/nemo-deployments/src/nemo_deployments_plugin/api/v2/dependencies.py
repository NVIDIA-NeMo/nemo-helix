# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the deployments plugin API."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client

__all__ = ["get_entity_client"]


async def get_service_entity_client(
    request_client: AsyncNemoClient = Depends(get_nemo_client),
) -> AsyncIterator[NemoEntitiesClient]:
    """Use deployments credentials for records containing the verified creator snapshot."""
    async with get_async_nemo_client(
        as_service="deployments", internal=True, http_client=request_client._http
    ) as client:
        yield NemoEntitiesClient(AsyncEntitiesClient.from_client(client))
