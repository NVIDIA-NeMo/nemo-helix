# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for evaluator plugin jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TypeVar, cast

import httpx
from nemo_evaluator_sdk.execution.metric_execution import run_sync
from nemo_platform_plugin.client.client import AsyncNemoClient, NemoClient

T = TypeVar("T")
AsyncClientT = TypeVar("AsyncClientT", bound=AsyncNemoClient)

type _SyncRequestHook = Callable[[httpx.Request], None]
type _SyncResponseHook = Callable[[httpx.Response], None]
type _AsyncRequestHook = Callable[[httpx.Request], Awaitable[None]]
type _AsyncResponseHook = Callable[[httpx.Response], Awaitable[None]]


def _async_request_hook(sync_hook: _SyncRequestHook) -> _AsyncRequestHook:
    async def _run(request: httpx.Request) -> None:
        await asyncio.to_thread(sync_hook, request)

    return _run


def _async_response_hook(sync_hook: _SyncResponseHook) -> _AsyncResponseHook:
    async def _run(response: httpx.Response) -> None:
        await asyncio.to_thread(sync_hook, response)

    return _run


def _sync_request_hooks(raw_hooks: object) -> list[_AsyncRequestHook]:
    if not isinstance(raw_hooks, list):
        return []
    return [_async_request_hook(cast(_SyncRequestHook, hook)) for hook in raw_hooks if callable(hook)]


def _sync_response_hooks(raw_hooks: object) -> list[_AsyncResponseHook]:
    if not isinstance(raw_hooks, list):
        return []
    return [_async_response_hook(cast(_SyncResponseHook, hook)) for hook in raw_hooks if callable(hook)]


def _sync_event_hooks_as_async(client: NemoClient) -> dict[str, list[_AsyncRequestHook] | list[_AsyncResponseHook]]:
    """Convert a sync typed client's httpx event hooks for use on an async transport."""
    event_hooks: dict[str, list[_AsyncRequestHook] | list[_AsyncResponseHook]] = {}
    raw_event_hooks = getattr(client._client, "event_hooks", None)
    if not isinstance(raw_event_hooks, Mapping):
        return event_hooks

    request_hooks = _sync_request_hooks(raw_event_hooks.get("request"))
    response_hooks = _sync_response_hooks(raw_event_hooks.get("response"))
    if request_hooks:
        event_hooks["request"] = request_hooks
    if response_hooks:
        event_hooks["response"] = response_hooks
    return event_hooks


@contextmanager
def async_client_from_sync_client(client: NemoClient) -> Iterator[AsyncNemoClient]:
    """Yield an async typed client with the sync client's auth, headers, and request hooks.

    Sync evaluator jobs still use their sync client for downloads and inference identity checks.
    Result persistence and Intake publication are async side effects, so they need an async client
    carrying the same platform identity.
    """
    http_client = httpx.AsyncClient(event_hooks=_sync_event_hooks_as_async(client))
    async_client = AsyncNemoClient(
        base_url=client.base_url,
        workspace=client.workspace,
        auth=client._auth,
        default_headers=client.default_headers or None,
        timeout=client._timeout,
        retry=client._retry,
        http_client=http_client,
        owns_http_client=True,
        url_resolver=client._url_resolver,
    )
    try:
        yield async_client
    finally:
        run_sync(async_client.aclose)


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
        event_hooks = {event: list(hooks) for event, hooks in async_client._client.event_hooks.items()}
        # Generated SDK auth refresh is implemented as request hooks; preserve it on the
        # loop-local transport so isolation does not turn authenticated clients anonymous.
        async with httpx.AsyncClient(event_hooks=event_hooks) as http_client:
            return await fn(async_client.with_http_client(http_client))

    return run_sync(_run)
