# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependency injection for the Agents plugin API.

Re-exports the standard NeMo Helix dependencies so route handlers can import from
a single location.
"""

from collections.abc import AsyncIterator

import httpx
from fastapi import Depends
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import get_nemo_client
from nemo_helix_plugin.entity_client import get_entity_client
from nemo_helix_plugin.files.client import AsyncFilesClient


def get_files_client(client: AsyncNemoClient = Depends(get_nemo_client)) -> AsyncFilesClient:
    """Provide a Files service client sharing the request's platform client transport."""
    return AsyncFilesClient.from_client(client)


async def get_registry_http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Provide an HTTP client for reading public container registries."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30.0)) as http:
        yield http


__all__ = ["get_entity_client", "get_files_client", "get_registry_http_client"]
