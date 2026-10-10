# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cancel request-scoped work when the HTTP client goes away."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, TypeVar

from fastapi import HTTPException
from starlette.requests import Request

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

CLIENT_CLOSED_REQUEST_STATUS = 499


class ClientDisconnectedError(HTTPException):
    """The client disconnected before its response was ready, so the work behind it was cancelled."""

    def __init__(self) -> None:
        super().__init__(status_code=CLIENT_CLOSED_REQUEST_STATUS, detail="Client closed the request.")


async def _wait_for_disconnect(request: Request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def cancel_on_disconnect(request: Request, work: Awaitable[_T]) -> _T:
    """Await *work*, cancelling it and raising :class:`ClientDisconnectedError` if the client leaves first.

    Call only after the request body has been read: the disconnect watcher consumes ``receive()``.
    """
    work_task = asyncio.ensure_future(work)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    try:
        await asyncio.wait({work_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        disconnect_task.cancel()
        if not work_task.done():
            work_task.cancel()
            await _drain_cancelled(work_task)
    if work_task.cancelled() and disconnect_task.done() and not disconnect_task.cancelled():
        raise ClientDisconnectedError
    return work_task.result()


async def _drain_cancelled(task: asyncio.Future[Any]) -> None:
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        if not task.done():
            raise
    except Exception:
        logger.warning("Work cancelled after a client disconnect failed while unwinding.", exc_info=True)
