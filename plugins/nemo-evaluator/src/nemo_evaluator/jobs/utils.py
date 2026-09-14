# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for evaluator plugin jobs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar, overload

import httpx
from nemo_evaluator_sdk.execution.metric_execution import run_sync
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_platform_plugin.sdk import AsyncNeMoPlatform, NeMoPlatform

T = TypeVar("T")
AsyncClientT = TypeVar("AsyncClientT", bound=AsyncNemoClient)


@overload
def as_nemo_client(sdk: NemoClient | NeMoPlatform) -> NemoClient: ...
@overload
def as_nemo_client(sdk: None) -> None: ...


def as_nemo_client(sdk: NemoClient | NeMoPlatform | None) -> NemoClient | None:
    """Return *sdk* as a typed :class:`NemoClient`, adapting a generated SDK handle if needed.

    Task containers inject typed clients and the local CLI injects generated ``NeMoPlatform``
    handles, so a job ``run`` receives either.
    """
    if isinstance(sdk, NeMoPlatform):
        return client_from_platform(sdk, NemoClient)
    return sdk


@overload
def as_async_nemo_client(async_sdk: AsyncNemoClient | AsyncNeMoPlatform) -> AsyncNemoClient: ...
@overload
def as_async_nemo_client(async_sdk: None) -> None: ...


def as_async_nemo_client(async_sdk: AsyncNemoClient | AsyncNeMoPlatform | None) -> AsyncNemoClient | None:
    """Async counterpart of :func:`as_nemo_client`."""
    if isinstance(async_sdk, AsyncNeMoPlatform):
        return client_from_platform(async_sdk, AsyncNemoClient)
    return async_sdk


def run_with_isolated_async_client(
    async_client: AsyncClientT,
    fn: Callable[[AsyncClientT], Awaitable[T]],
) -> T:
    """Run ``fn(cloned_client)`` via ``run_sync`` without binding ``async_client``'s httpx client.

    Clones ``async_client`` onto a throwaway httpx client for the duration of ``fn`` so the injected
    client's transport is not bound to this temporary event loop (and later ``run_sync`` calls can
    still reuse ``async_client``).
    """

    async def _run() -> T:
        async with httpx.AsyncClient() as http_client:
            return await fn(async_client.with_http_client(http_client))

    return run_sync(_run)
