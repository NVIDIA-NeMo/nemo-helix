# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async intake client backed by the NeMo Helix SDK."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import quote

from switchyard.lib.factories.intake_sink.intake_sink_config import (
    IntakeSinkConfig,
)

if TYPE_CHECKING:
    from nemo_helix import AsyncNeMoHelix

log = logging.getLogger(__name__)
JsonObject = dict[str, object]

#: Max seconds ``aclose()`` waits for the in-flight queue to drain before
#: bailing out and closing the HTTP client anyway.
_DRAIN_TIMEOUT_S = 30.0

_SDK_INSTALL_HINT = (
    "Install the NeMo Helix SDK with the `intake` extra, e.g. "
    "`uv run --extra intake switchyard launch claude --enable-intake`."
)

_SDK_LOGIN_HINT = (
    "Run `uv run nhx auth login --base-url https://platform.example.com`, "
    "or pass --intake-base-url and --intake-api-key."
)


class IntakeClient:
    """Fail-open queue + worker that POSTs completed turns through AsyncNeMoHelix.

    The SDK handles config bootstrap, auth, and retries. Intake still uses a
    generic POST path because the stable SDK does not expose a generated
    ``client.intake`` resource.
    """

    def __init__(self, config: IntakeSinkConfig) -> None:
        self._client: AsyncNeMoHelix = _build_sdk_client(config)
        # Payloads need a concrete workspace even when the SDK resolves it.
        self._config = config.model_copy(update={
            "workspace": config.workspace or _sdk_workspace(self._client) or "default",
        })
        self._queue: asyncio.Queue[JsonObject | None] = asyncio.Queue(
            maxsize=self._config.max_queue_size,
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._dropped = 0

    @property
    def effective_config(self) -> IntakeSinkConfig:
        """Config with workspace resolved against SDK / nhx config."""
        return self._config

    async def enqueue(self, payload: JsonObject) -> None:
        """Queue a payload for background POST."""
        await self._enqueue_payload(payload, allow_closed=False)

    def enqueue_background(self, payload_factory: Callable[[], JsonObject]) -> None:
        """Build and enqueue a payload in a client-retained background task."""
        if self._closed:
            log.warning("IntakeClient is closed; dropping payload")
            return

        task = asyncio.create_task(self._build_and_enqueue(payload_factory))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _enqueue_payload(self, payload: JsonObject, *, allow_closed: bool) -> None:
        if self._closed and not allow_closed:
            log.warning("IntakeClient is closed; dropping payload")
            return
        self._ensure_worker()
        if self._config.on_queue_full == "block":
            await self._queue.put(payload)
            return
        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            self._dropped += 1
            log.warning(
                "Intake queue full (max=%d); dropping payload. dropped=%d",
                self._config.max_queue_size,
                self._dropped,
            )

    async def _build_and_enqueue(self, payload_factory: Callable[[], JsonObject]) -> None:
        try:
            payload = payload_factory()
        except Exception:
            log.exception("Failed to build intake payload; dropping")
            return
        await self._enqueue_payload(payload, allow_closed=True)

    async def aclose(self) -> None:
        """Drain pending items, stop the worker, and close the HTTP client."""
        if self._closed:
            return
        self._closed = True
        await self._wait_for_background_tasks()
        self._ensure_worker()
        await self._queue.put(None)
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._worker_task, timeout=_DRAIN_TIMEOUT_S)
            except asyncio.TimeoutError:
                log.warning("Timed out draining intake queue during shutdown")
                self._worker_task.cancel()
                await asyncio.gather(self._worker_task, return_exceptions=True)
        await self._client.close()

    async def _wait_for_background_tasks(self) -> None:
        if not self._background_tasks:
            return
        pending = list(self._background_tasks)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=_DRAIN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning("Timed out waiting for intake background tasks during shutdown")
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            payload = await self._queue.get()
            if payload is None:
                self._queue.task_done()
                return
            try:
                await self._post_payload(payload)
            except Exception:
                log.exception("Intake worker failed while posting payload; dropping")
            finally:
                self._queue.task_done()

    async def _post_payload(self, payload: JsonObject) -> None:
        try:
            await self._client.post(
                _entries_path(self._config.workspace or "default"),
                cast_to=object,
                body=payload,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                raise
            log.warning(
                "Intake rejected payload with %d: %s",
                status_code,
                str(exc)[:200],
            )


def _build_sdk_client(config: IntakeSinkConfig) -> AsyncNeMoHelix:
    """Build an ``AsyncNeMoHelix`` for the intake POST path."""
    if importlib.util.find_spec("nemo_helix") is None:
        raise RuntimeError(_SDK_INSTALL_HINT)

    from nemo_helix import AsyncNeMoHelix

    client_kwargs: dict[str, object] = {
        "timeout": config.request_timeout_s,
        "max_retries": config.max_retries,
    }
    if config.workspace is not None:
        client_kwargs["workspace"] = config.workspace
    if config.intake_base_url is not None:
        client_kwargs["base_url"] = config.intake_base_url
    if config.api_key is not None:
        client_kwargs["access_token"] = config.api_key

    try:
        return AsyncNeMoHelix(**client_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to construct AsyncNeMoHelix for intake: {exc}. "
            + _SDK_LOGIN_HINT,
        ) from exc


def _sdk_workspace(client: AsyncNeMoHelix) -> str | None:
    workspace = getattr(client, "workspace", None)
    return workspace if isinstance(workspace, str) else None


def _entries_path(workspace: str) -> str:
    return f"/apis/intake/v2/workspaces/{quote(workspace, safe='')}/entries"
