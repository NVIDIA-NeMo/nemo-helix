# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse span client tests."""

import asyncio
from typing import Annotated, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, HTTPException
from fastapi.testclient import TestClient
from nhx.intake.dependencies import get_clickhouse_client, get_optional_clickhouse_client
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE
from nhx.intake.service import IntakeService
from nhx.intake.spans.clickhouse_client import (
    ClickHouseSettings,
    ClickHouseSpanClient,
)


class CountingClickHouseSpanClient(ClickHouseSpanClient):
    def __init__(self) -> None:
        super().__init__(
            ClickHouseSettings(
                url="http://localhost:8123",
                user="default",
                password="",
                database="default",
            )
        )
        self.created = 0

    async def _create_raw_client(self, *, database: str) -> Any:
        self.created += 1
        await asyncio.sleep(0)
        return object()


@pytest.mark.asyncio
async def test_get_raw_client_initializes_once_under_concurrency():
    client = CountingClickHouseSpanClient()

    raw_clients = await asyncio.gather(*(client._get_raw_client() for _ in range(10)))

    assert client.created == 1
    assert len({id(raw_client) for raw_client in raw_clients}) == 1


@pytest.mark.asyncio
async def test_clickhouse_dependency_returns_actionable_503(monkeypatch: pytest.MonkeyPatch) -> None:
    client = CountingClickHouseSpanClient()
    monkeypatch.setattr(client, "bootstrap_schema", AsyncMock(side_effect=PermissionError("read-only volume")))
    intake = IntakeService()
    intake.clickhouse_client = client
    dependency = get_clickhouse_client(intake)
    with pytest.raises(HTTPException) as error:
        await anext(dependency)

    assert error.value.status_code == 503
    assert error.value.detail == CLICKHOUSE_UNAVAILABLE_MESSAGE


def test_intake_routes_see_late_resource_adoption_and_disabled_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    intake = IntakeService()
    app = intake.create_app()

    @app.get("/optional-storage")
    def optional_storage(
        client: Annotated[ClickHouseSpanClient | None, Depends(get_optional_clickhouse_client)],
    ) -> bool:
        return client is not None

    @app.get("/required-storage")
    async def required_storage(client: Annotated[ClickHouseSpanClient, Depends(get_clickhouse_client)]) -> bool:
        return client is not None

    test_client = TestClient(app)
    assert test_client.get("/optional-storage").json() is False
    assert test_client.get("/required-storage").status_code == 503

    client = CountingClickHouseSpanClient()
    monkeypatch.setattr(client, "bootstrap_schema", AsyncMock(return_value=None))
    intake.clickhouse_client = client
    assert test_client.get("/optional-storage").json() is True
    assert test_client.get("/required-storage").json() is True

    intake.clickhouse_client = None
    assert test_client.get("/optional-storage").json() is False
    assert test_client.get("/required-storage").status_code == 503
