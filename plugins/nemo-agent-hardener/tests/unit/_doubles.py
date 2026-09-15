# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test doubles for the agent-hardener plugin.

The job/SDK seams take concrete types (:class:`JobContext`, ``NeMoPlatform``) that are impractical to
build in a unit test, so these factories return duck-typed stand-ins narrowed with :func:`typing.cast`.
The cast is deliberate and lives here only: keeping the fakes in one place means a stub that drifts from
the real shape is fixed once, rather than per file — the drift that let a bad fixture hide a real bug
(see docs/agent-hardener-review/findings.md, #66).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from nemo_platform import AsyncNeMoPlatform, NeMoPlatform
from nemo_platform_plugin.job_context import JobContext


def make_job_context(
    tmp_path: Path,
    *,
    workspace: str = "default",
    job_id: str = "job-123",
    on_save: Any = None,
) -> JobContext:
    """A :class:`JobContext` stand-in backed by *tmp_path* for persistent storage.

    Covers the surface the jobs actually touch: ``workspace``, ``job_id``, ``storage.persistent`` and
    ``results.save``. Pass *on_save* to capture saved artifacts.
    """
    return cast(
        JobContext,
        SimpleNamespace(
            workspace=workspace,
            job_id=job_id,
            storage=SimpleNamespace(persistent=tmp_path),
            results=SimpleNamespace(save=on_save or (lambda *_a, **_k: None)),
        ),
    )


def make_sdk(entities: Any = None, **namespaces: Any) -> NeMoPlatform:
    """A ``NeMoPlatform`` stand-in exposing only the namespaces a test needs (usually ``entities``)."""
    return cast(NeMoPlatform, SimpleNamespace(entities=entities, **namespaces))


def make_async_sdk(**namespaces: Any) -> AsyncNeMoPlatform:
    """An ``AsyncNeMoPlatform`` stand-in (the async SDK resources only read ``base_url``)."""
    return cast(AsyncNeMoPlatform, SimpleNamespace(**namespaces))


def make_entity(**data: Any) -> Any:
    """An entity-store record shaped like the real one: domain fields live under ``.data``.

    Deliberately *not* a bare ``Mock`` — ``Mock().some_field`` is truthy, which is exactly how the
    events-fileset fallback bug reached production while its test passed.
    """
    return SimpleNamespace(data=dict(data))
