# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller-scoped read checks for sources that submission copies into a frozen spec.

Submission resolves tasks, metrics and model references with the evaluator service's credentials,
which can read more than the caller. A resolver that is given an authorizer checks each source as
the caller before reading it, so the frozen spec never holds a definition the caller cannot read.
Resolvers given no authorizer keep their existing behavior.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal
from urllib.parse import quote

#: Request-scoped check of ``(method, path)`` as the caller; raises when the caller is denied.
type SourceReadAuthorizer = Callable[[str, str], Awaitable[None]]

type SourceKind = Literal["task", "metric", "model", "virtual_model"]

_READ_PATHS: dict[SourceKind, str] = {
    "task": "/apis/evals/v2/workspaces/{workspace}/tasks/{name}",
    "metric": "/apis/evals/v2/workspaces/{workspace}/metrics/{name}",
    "model": "/apis/models/v2/workspaces/{workspace}/models/{name}",
    "virtual_model": "/apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}",
}


async def authorize_source_read(
    authorize: SourceReadAuthorizer | None, kind: SourceKind, workspace: str, name: str
) -> None:
    """Check that the caller may read one source, before the service reads it.

    Args:
        authorize: Caller-scoped authorizer, or ``None`` when the resolver runs without one.
        kind: Source collection to check.
        workspace: Workspace that owns the source.
        name: Source name.

    Raises:
        Exception: Whatever ``authorize`` raises for a denied read, typically a 403 ``HTTPException``.
    """
    if authorize is None:
        return
    path = _READ_PATHS[kind].format(workspace=quote(workspace, safe=""), name=quote(name, safe=""))
    await authorize("GET", path)


def remember_allowed(authorize: SourceReadAuthorizer) -> SourceReadAuthorizer:
    """Skip repeat checks of a path already allowed within one submission.

    A taskset can name hundreds of tasks that share one metric or judge model. Only successes are
    remembered, so a denial is never cached and concurrent first checks of one path may both run.
    """
    allowed: set[tuple[str, str]] = set()

    async def check(method: str, path: str) -> None:
        if (method, path) in allowed:
            return
        await authorize(method, path)
        allowed.add((method, path))

    return check
