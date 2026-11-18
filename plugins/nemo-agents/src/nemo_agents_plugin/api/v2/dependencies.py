# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependency injection for the Agents plugin API.

Re-exports the standard NeMo Helix dependencies so route handlers can import from
a single location.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends
from nemo_agents_plugin.skill_service import SkillService
from nemo_agents_plugin.skill_storage import SkillArchiveStore
from nemo_helix_plugin.auth import current_auth_context
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.dependencies import RequestAuthorizer, get_nemo_client, get_request_authorizer
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client
from nemo_helix_plugin.evals.client import AsyncEvaluatorClient
from nemo_helix_plugin.files.client import AsyncFilesClient

_AGENTS_SERVICE = "agents"


def get_files_client(client: AsyncNemoClient = Depends(get_nemo_client)) -> AsyncFilesClient:
    """Provide a Files service client sharing the request's platform client transport."""
    return AsyncFilesClient.from_client(client)


@asynccontextmanager
async def _delegated_service_client(request_client: AsyncNemoClient) -> AsyncGenerator[AsyncNemoClient, None]:
    """Yield a client that calls the platform as the agents service, delegating the request's caller.

    Without a request principal (auth off) there is nobody to delegate, so the request-scoped
    client is used rather than an undelegated service client.
    """
    caller = current_auth_context()
    if caller is None:
        yield request_client
        return
    async with get_async_nemo_client(
        as_service=_AGENTS_SERVICE, internal=True, on_behalf_of=caller.to_principal()
    ) as service_client:
        yield service_client


async def get_skill_service(
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    request_client: AsyncNemoClient = Depends(get_nemo_client),
    authorize: RequestAuthorizer = Depends(get_request_authorizer),
) -> AsyncGenerator[SkillService, None]:
    """Provide a :class:`SkillService` for one request.

    Files and evals calls go through the agents service delegating the caller, because the service
    writes its own ``nemo-agents-skills`` fileset, which the caller may not be able to write. That
    client can read more than the caller can, so :class:`SkillService` checks every caller-supplied
    source (the source fileset and any linked taskset) with ``authorize``, as the caller, before
    reading it.
    """
    async with _delegated_service_client(request_client) as client:
        yield SkillService(
            entity_client,
            SkillArchiveStore(AsyncFilesClient.from_client(client)),
            authorize=authorize,
            evals_client=AsyncEvaluatorClient.from_client(client),
        )


__all__ = ["get_entity_client", "get_files_client", "get_skill_service"]
