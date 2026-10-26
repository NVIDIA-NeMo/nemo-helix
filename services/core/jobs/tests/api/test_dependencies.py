# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import httpx
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nhx.common.entities.client import EntityClient
from nhx.core.jobs.api.dependencies import dep_system_entity_client


def test_system_entity_client_uses_jobs_identity_and_reuses_transport() -> None:
    http_client = AsyncMock(spec=httpx.AsyncClient)
    http_client.headers = {}
    entity_client = EntityClient(
        AsyncEntitiesClient(
            base_url="http://localhost:8080",
            workspace="default",
            default_headers={"X-NHX-Principal-Id": "alice@example.com"},
            http_client=http_client,
        )
    )

    system_client = dep_system_entity_client(entity_client)

    assert system_client is not entity_client
    assert system_client._client._default_headers["X-NHX-Principal-Id"] == "service:jobs"
    assert system_client._client._default_headers["X-NHX-Internal"] == "true"
    assert entity_client._client._default_headers["X-NHX-Principal-Id"] == "alice@example.com"
    assert system_client._client._http is entity_client._client._http
