# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the Jobs API."""

from fastapi import Depends
from nemo_platform import AsyncNeMoPlatform
from nmp.common.entities.client import EntityClient
from nmp.common.service.dependencies import get_entity_client, get_sdk_client
from nmp.core.jobs.app.dispatcher import JobDispatcher


async def dep_dispatcher(
    entity_client: EntityClient = Depends(get_entity_client),
    sdk: AsyncNeMoPlatform = Depends(get_sdk_client),
) -> JobDispatcher:
    """Dependency to get the job dispatcher with EntityClient and SDK client."""
    return JobDispatcher(
        store=entity_client,
        sdk=sdk,
    )
