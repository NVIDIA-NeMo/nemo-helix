# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from nemo_agents_plugin.client_disconnect import (
    CLIENT_CLOSED_REQUEST_STATUS,
    ClientDisconnectedError,
    cancel_on_disconnect,
)
from starlette.requests import Request


def _request(disconnected: asyncio.Event) -> Request:
    async def receive() -> dict[str, Any]:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    return Request({"type": "http", "method": "POST", "headers": []}, receive)


async def _value_after(started: asyncio.Event, cancelled: asyncio.Event, value: str = "done") -> str:
    started.set()
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        cancelled.set()
        raise
    return value


async def test_returns_work_result_while_client_stays_connected() -> None:
    async def work() -> str:
        return "answer"

    assert await cancel_on_disconnect(_request(asyncio.Event()), work()) == "answer"


async def test_propagates_work_failure() -> None:
    async def work() -> str:
        raise ValueError("agent failed")

    with pytest.raises(ValueError, match="agent failed"):
        await cancel_on_disconnect(_request(asyncio.Event()), work())


async def test_disconnect_cancels_work_and_reports_client_closed_request() -> None:
    disconnected, started, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    waiter = asyncio.create_task(cancel_on_disconnect(_request(disconnected), _value_after(started, cancelled)))
    await started.wait()

    disconnected.set()

    with pytest.raises(ClientDisconnectedError) as raised:
        await waiter
    assert raised.value.status_code == CLIENT_CLOSED_REQUEST_STATUS
    assert cancelled.is_set()


async def test_cancelling_the_caller_cancels_work() -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()
    waiter = asyncio.create_task(cancel_on_disconnect(_request(asyncio.Event()), _value_after(started, cancelled)))
    await started.wait()

    waiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert cancelled.is_set()
