# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference NemoFunction implementations for the example plugin.

Two functions cover the canonical wire shapes:

- :class:`GreetFunction` — non-streaming (single JSON response). The
  simplest possible NemoFunction: validate spec, await some work, return
  a Pydantic model.
- :class:`CountFunction` — streaming (NDJSON, one frame per line). Yields
  :class:`Tick` frames, then a :class:`~nemo_helix_plugin.functions.frames.Done`
  terminator. The route adapter wraps the stream with framework-managed
  heartbeat injection.

Both are exposed under the ``nemo.functions`` entry-point group as
``example.greet`` and ``example.count``. The example plugin's
:class:`~nemo_example_plugin.service.ExampleService` mounts the
auto-derived routers under ``/apis/example/v2/workspaces/{workspace}``
via :func:`~nemo_helix_plugin.functions.routes.add_function_routes`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import ClassVar, Literal

from nemo_example_plugin.core import say_hello
from nemo_helix_plugin.function import NemoFunction
from nemo_helix_plugin.function_context import FunctionContext
from nemo_helix_plugin.functions.frames import Done
from pydantic import BaseModel, Field


class GreetSpec(BaseModel):
    """Inputs for the non-streaming greet function."""

    name: str = Field(default="world", description="Name to greet.")


class GreetResponse(BaseModel):
    """Single-shot response for :class:`GreetFunction`."""

    message: str
    workspace: str


class GreetFunction(NemoFunction[GreetSpec]):
    """Return a greeting for *spec.name* alongside the active workspace.

    Demonstrates signature-based DI: declaring ``ctx: FunctionContext``
    on ``run`` opts the function in to the per-request context. The
    framework binds ``ctx.workspace`` from the URL path parameter
    (or ``--workspace`` flag) and ``ctx.request_id`` from the optional
    ``X-Request-ID`` header.
    """

    name: ClassVar[str] = "greet"
    description: ClassVar[str] = "Greet a name and echo back the active workspace."
    spec_schema: ClassVar[type[BaseModel]] = GreetSpec

    async def run(self, spec: GreetSpec, *, ctx: FunctionContext) -> GreetResponse:
        return GreetResponse(message=say_hello(spec.name), workspace=ctx.workspace)


class CountSpec(BaseModel):
    """Inputs for the streaming count function."""

    upto: int = Field(default=3, ge=0, le=100, description="How many tick frames to emit.")


class Tick(BaseModel):
    """Per-iteration frame emitted by :class:`CountFunction`.

    The ``kind`` discriminator follows the convention shared with
    :class:`~nemo_helix_plugin.functions.frames.Heartbeat` /
    :class:`~nemo_helix_plugin.functions.frames.Done` so client-side
    consumers can branch on a single field across all framework and
    plugin frames.
    """

    kind: Literal["tick"] = "tick"
    n: int


class CountFunction(NemoFunction[CountSpec]):
    """Stream ``spec.upto`` tick frames terminated by a ``Done`` frame.

    Demonstrates the streaming branch of ``add_function_routes``:
    declaring an async-generator ``run`` (``async def`` with ``yield``)
    flips the route to ``application/x-ndjson``. Each yielded
    Pydantic model becomes one NDJSON line on the wire; the route
    wraps the iterator with a heartbeat injector so proxies see
    keep-alive frames during long idle gaps.
    """

    name: ClassVar[str] = "count"
    description: ClassVar[str] = "Stream a sequence of NDJSON tick frames."
    spec_schema: ClassVar[type[BaseModel]] = CountSpec

    async def run(self, spec: CountSpec) -> AsyncIterator[BaseModel]:
        for n in range(spec.upto):
            yield Tick(n=n)
        yield Done()
