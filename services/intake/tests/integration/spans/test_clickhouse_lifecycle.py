# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ClickHouse client ownership through the platform test harness."""

import asyncio
from typing import Any

import pytest
from aiohttp import ClientSession
from nemo_helix_plugin.intake.client import AsyncIntakeClient
from nhx.intake.config import IntakeConfig
from nhx.intake.service import IntakeService
from nhx.intake.spans.clickhouse_client import ClickHouseSpanClient
from nhx.testing import ClientContext, create_test_client


def test_concurrent_async_callers_share_and_close_one_driver(
    intake_config: IntakeConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[ClientSession] = []
    create = ClickHouseSpanClient._create_raw_client

    async def tracked_create(self: ClickHouseSpanClient, *, database: str) -> Any:
        raw_client = await create(self, database=database)
        sessions.append(raw_client._session)
        return raw_client

    monkeypatch.setattr(ClickHouseSpanClient, "_create_raw_client", tracked_create)
    service = IntakeService()

    with asyncio.Runner() as caller:
        with create_test_client(
            lambda: service,
            client_type=ClientContext,
            service_configs={IntakeService: intake_config},
        ) as ctx:
            client = AsyncIntakeClient.from_client(ctx.async_client)
            assert ctx.test_client.portal is not None
            service_loop = ctx.test_client.portal.call(asyncio.get_running_loop)
            assert service.clickhouse_client is not None

            async def read_spans() -> None:
                spans = await client.list_spans(workspace="default")
                assert [span async for span in spans.items()] == []

            async def read_concurrently() -> None:
                await asyncio.gather(*(read_spans() for _ in range(4)))

            asyncio.run(read_concurrently())
            caller.run(read_concurrently())
            assert len(sessions) == 1
            assert sessions[0]._loop is service_loop
            assert not sessions[0].closed

        # The caller's loop is still open when the service closes its connection.
        assert sessions[0].closed
