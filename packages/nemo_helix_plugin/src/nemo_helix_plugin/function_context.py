# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime context handed to :meth:`NemoFunction.run`.

A function context carries the request-scope facts a function may need
without forcing every signature to take them: the workspace the
caller scoped the request to, and a request id for log correlation.

The context is intentionally narrower than :class:`JobContext`:

- No ``storage`` — functions run in the request path, not in a task
  container; if a function needs to land artifacts it goes through
  the SDK like any other request handler.
- No ``results`` sink — the function's response (or NDJSON stream)
  *is* the result channel.
- No ``principal`` — auth-on-behalf-of support is handled by the
  platform's auth middleware and cuts across functions and routes
  alike. Adding a per-call principal slot here without the cross-cutting fix
  would lock in a half-finished surface; the slot lands when the
  cross-cutting fix does.

Plugin authors opt in by declaring a keyword-only parameter named
``ctx`` on :meth:`NemoFunction.run`; the route adapter and the local
CLI both honour signature-based DI.

Example::

    class GreetFunction(NemoFunction):
        spec_schema = GreetSpec

        async def run(self, spec, *, ctx: FunctionContext) -> dict:
            logger.info("greet workspace=%s req=%s", ctx.workspace, ctx.request_id)
            return {"message": f"Hello, {spec.name}!"}
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class FunctionContext:
    """Per-request context for :meth:`NemoFunction.run`.

    Attributes:
        workspace: Workspace scope the request was made against.
            For local CLI runs this comes from ``--workspace`` and
            defaults to ``"default"``; for HTTP runs it's the path
            parameter the route adapter pulls from
            ``/v2/workspaces/{workspace}/...``.
        request_id: Optional correlation id for tracing. Set from the
            inbound ``X-Request-ID`` header by the route adapter; left
            ``None`` for local CLI runs unless the caller passes one.
    """

    workspace: str
    request_id: str | None = None


__all__ = ["FunctionContext"]
