# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test doubles for the agent-hardener plugin.

The job/SDK seams take concrete types (:class:`JobContext`, :class:`NemoClient`) that are impractical to
build in a unit test, so these factories return duck-typed stand-ins narrowed with :func:`typing.cast`.
The cast is deliberate and lives here only: keeping the fakes in one place means a stub that drifts from
the real shape is fixed once, rather than per file — the drift that let a bad fixture hide a real bug
(see docs/agent-hardener-review/findings.md, #66).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.job_context import JobContext


def make_job_context(
    tmp_path: Path,
    *,
    workspace: str = "default",
    job_id: str = "job-123",
    attempt_id: str = "attempt-123",
    on_save: Any = None,
) -> JobContext:
    """A :class:`JobContext` stand-in backed by *tmp_path* for persistent storage.

    Covers the surface the jobs actually touch: ``workspace``, ``job_id``, ``attempt_id``,
    ``storage.persistent`` and ``results.save``. Pass *on_save* to capture saved artifacts.
    """
    return cast(
        JobContext,
        SimpleNamespace(
            workspace=workspace,
            job_id=job_id,
            attempt_id=attempt_id,
            storage=SimpleNamespace(persistent=tmp_path),
            results=SimpleNamespace(save=on_save or (lambda *_a, **_k: None)),
        ),
    )


def make_sdk(entities: Any = None, **namespaces: Any) -> NemoClient:
    """A ``NemoClient`` stand-in exposing only the namespaces a test needs (usually ``entities``)."""
    return cast(NemoClient, SimpleNamespace(entities=entities, **namespaces))


def make_async_sdk(**namespaces: Any) -> AsyncNemoClient:
    """An ``AsyncNemoClient`` stand-in (the async SDK resources only read ``base_url``)."""
    return cast(AsyncNemoClient, SimpleNamespace(**namespaces))


def make_entity(**data: Any) -> Any:
    """An entity-store record shaped like the real one: domain fields live under ``.data``.

    Deliberately *not* a bare ``Mock`` — ``Mock().some_field`` is truthy, which is exactly how the
    events-fileset fallback bug reached production while its test passed.
    """
    return SimpleNamespace(data=dict(data))


class SecretsStore:
    """A real :class:`AsyncNemoClient` whose Secrets service holds *existing* (``workspace/name``) secrets.

    Records every request path, so a test can assert that an existence check never read a value
    (``.../access``). *status* answers every lookup instead, to model a store that is down.
    """

    def __init__(self, existing: set[str], *, status: int | None = None) -> None:
        self.paths: list[str] = []
        self._existing, self._status = existing, status

    def client(self, **_kwargs: Any) -> AsyncNemoClient:
        """A fresh client per call, since callers may close it (``async with``); takes a provider's kwargs."""
        return AsyncNemoClient(
            base_url="http://platform",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle), base_url="http://platform"),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if self._status is not None:
            return httpx.Response(self._status, json={"detail": "unavailable"})
        workspace, name = request.url.path.split("/workspaces/")[1].split("/secrets/")
        if f"{workspace}/{name}" in self._existing:
            return httpx.Response(200, json={"name": name, "workspace": workspace})
        return httpx.Response(404, json={"detail": "not found"})
