# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merged routing keeps each caller's owning-service credentials."""

from fastapi import APIRouter, Depends
from nemo_helix_plugin.client.client import NemoClient
from nhx.common.entities.client import EntityClient
from nhx.common.service import RouterConfig, Service
from nhx.common.service.dependencies import get_entity_client
from nhx.testing import as_user, create_test_client


class CredentialProbe(Service):
    def __init__(self, name: str):
        super().__init__(name=name, module_name=__name__)

    def get_routers(self) -> list[RouterConfig]:
        router = APIRouter()

        @router.get("/v2/probe")
        async def probe(client: EntityClient = Depends(get_entity_client)) -> dict[str, str]:
            return client._client.default_headers

        return [RouterConfig(router, tag="Probe", description="Credential binding probe")]


def test_merged_routes_select_their_own_service_credential() -> None:
    with create_test_client(
        lambda: CredentialProbe("jobs"),
        lambda: CredentialProbe("models"),
        auth_enabled=False,
        client_type=NemoClient,
    ) as base:
        client = as_user(base, "creator@example.com")
        for service in ("jobs", "models"):
            response = client._http.get(f"/apis/{service}/v2/probe", headers=client.default_headers)
            assert response.status_code == 200, response.text
            headers = response.json()
            assert headers["X-NHX-Principal-Id"] == f"service:{service}"
            assert headers["X-NHX-Principal-On-Behalf-Of"] == "creator@example.com"
