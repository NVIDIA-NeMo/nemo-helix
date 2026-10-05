# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClickHouse span client tests."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from nhx.intake.readiness import CLICKHOUSE_UNAVAILABLE_MESSAGE
from nhx.intake.spans.clickhouse_client import (
    ClickHouseSettings,
    ClickHouseSpanClient,
    get_clickhouse_client,
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
    app = FastAPI()
    app.state.intake_service = SimpleNamespace(clickhouse_client=client)
    request = Request(
        {
            "type": "http",
            "app": app,
            "headers": [],
            "method": "GET",
            "path": "/",
            "query_string": b"",
        }
    )

    dependency = get_clickhouse_client(request)
    with pytest.raises(HTTPException) as error:
        await anext(dependency)

    assert error.value.status_code == 503
    assert error.value.detail == CLICKHOUSE_UNAVAILABLE_MESSAGE
