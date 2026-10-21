# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task build job + status persistence — **approach-agnostic**.

Legacy synchronous build primitives. New finalize requests are persisted and
processed by :mod:`scaled_evals.api.build.queue_worker`; these functions remain
for compatibility with direct callers.

This layer is reused by EVERY build approach: it depends only on the contract
"produce `(image_ref, image_digest)` or raise". The actual build is delegated to
the current approach (`buildkit`); a future multi-backend setup would select the
backend at that one call site. Everything else here — the job wrapper and the
`ready`/`failed` status writes — is shared. See docs/internals/ARCHITECTURE.md § Container
Build.
"""

from __future__ import annotations

import asyncio

from scaled_evals.api.build.buildkit import build_revision_image
from scaled_evals.api.build.image_builder_service import resolve_uploaded_revision_image
from scaled_evals.api.settings import settings
from scaled_evals.api.task_store import TaskStore


def run_finalize_build(task_id: str, revision: int, tarball_object_key: str) -> None:
    """Run a revision build and persist its outcome. Background-job entrypoint.

    Runs after the HTTP response, so it writes the outcome itself. On success the
    revision flips to `ready` with image_ref/image_digest and becomes the
    task's current revision; on failure it flips to `failed` with the
    captured build log in build_error.

    New finalize requests are handled by the durable leased worker in
    `queue_worker` instead. It is synchronous so a FastAPI BackgroundTask can run it in the
    threadpool — the blocking download and Entity Store writes stay off the event
    loop, while the build itself drives an async subprocess via `asyncio.run`.
    """
    try:
        # Delegate the actual build to the current approach (BuildKit). A future
        # multi-backend setup would select the backend here.
        image_ref, image_digest = asyncio.run(
            build_revision_image(task_id, revision, tarball_object_key)
        )
    except Exception as exc:  # any failure is recorded as build_error, never lost
        _record_failure(task_id, revision, str(exc))
        return
    _record_success(task_id, revision, image_ref, image_digest)


def run_finalize_uploaded(
    task_id: str,
    revision: int,
    *,
    tarball_object_key: str,
    context_path: str = ".",
) -> None:
    """Resolve an uploaded task pack via the builder service and persist it."""
    try:
        image_ref, image_digest = resolve_uploaded_revision_image(
            tarball_object_key=tarball_object_key,
            context_path=context_path,
        )
    except Exception as exc:  # any failure is recorded as build_error, never lost
        _record_failure(task_id, revision, str(exc))
        return
    _record_success(task_id, revision, image_ref, image_digest)


def _record_success(task_id: str, revision: int, image_ref: str, image_digest: str) -> None:
    TaskStore().record_success(task_id, revision, image_ref=image_ref, image_digest=image_digest)


def _record_failure(task_id: str, revision: int, build_error: str) -> None:
    TaskStore().record_failure(task_id, revision, build_error)
