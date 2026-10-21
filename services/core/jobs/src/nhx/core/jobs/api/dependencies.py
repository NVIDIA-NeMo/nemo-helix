# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the Jobs API."""

from fastapi import Depends, Request
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client, get_nemo_client
from nhx.core.jobs.app.dispatcher import JobDispatcher


def dep_files_client(client: AsyncNemoClient = Depends(get_nemo_client)) -> AsyncFilesClient:
    """Dependency to get a request-scoped Files service client."""
    return AsyncFilesClient.from_client(client)


def dep_system_entity_client(request: Request) -> EntityClient | None:
    """Entity client acting as the jobs service, for records kept in the system workspace.

    The request-scoped client acts for the caller, and callers only read the system workspace.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        return None
    return service.dependency_provider.get_service_entity_client(service.name)


async def dep_dispatcher(
    entity_client: EntityClient = Depends(get_entity_client),
    client: AsyncNemoClient = Depends(get_nemo_client),
    system_entity_client: EntityClient | None = Depends(dep_system_entity_client),
) -> JobDispatcher:
    """Dependency to get the job dispatcher with EntityClient and request-scoped typed clients."""
    return JobDispatcher(
        store=entity_client,
        files=AsyncFilesClient.from_client(client),
        secrets=AsyncSecretsClient.from_client(client),
        system_store=system_entity_client,
    )
