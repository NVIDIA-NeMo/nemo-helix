# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the Jobs API."""

from fastapi import Depends, Request
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.secrets.client import AsyncSecretsClient
from nhx.common.entities.client import EntityClient
from nhx.common.service.dependencies import get_entity_client, get_nemo_client, get_service
from nhx.core.jobs.app.dispatcher import JobDispatcher


def dep_files_client(client: AsyncNemoClient = Depends(get_nemo_client)) -> AsyncFilesClient:
    """Dependency to get a request-scoped Files service client."""
    return AsyncFilesClient.from_client(client)


def dep_system_entity_client(request: Request) -> EntityClient:
    """Entity client acting as the jobs service, for records kept in the system workspace.

    The request-scoped client acts for the caller, and callers only read the system workspace.
    Raises rather than falling back to the caller's client, which would check the system
    workspace as the caller.
    """
    service = get_service(request, "jobs")
    return service.dependency_provider.get_service_entity_client(service.name)


async def dep_dispatcher(
    entity_client: EntityClient = Depends(get_entity_client),
    client: AsyncNemoClient = Depends(get_nemo_client),
    system_entity_client: EntityClient = Depends(dep_system_entity_client),
) -> JobDispatcher:
    """Dependency to get the job dispatcher with EntityClient and request-scoped typed clients."""
    return JobDispatcher(
        store=entity_client,
        files=AsyncFilesClient.from_client(client),
        secrets=AsyncSecretsClient.from_client(client),
        system_store=system_entity_client,
    )
