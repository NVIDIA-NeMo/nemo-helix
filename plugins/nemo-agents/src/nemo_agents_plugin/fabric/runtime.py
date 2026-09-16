# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime execution helpers for Fabric-backed agents.

This module is the internal Platform boundary around Fabric SDK runtime
execution. It accepts already-translated in-memory FabricConfig objects and
normalizes Fabric runtime results/errors for Platform callers.

It intentionally does not own Platform agent config loading, FabricConfig
translation, CLI/API wiring, deploy semantics, or durable session management.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# CI type-checks this plugin via ty extra-paths without installing nemo-agents deps.
from nemo_fabric import (
    Fabric,
    FabricConfig,
    FabricError,
    RunRequest,
    RunResult,
    Runtime,
)

# Defined in the plugin contract package so an extension author can read a result
# without depending on this package. Re-exported here for convenience.
from nemo_platform_plugin.agents.execute_extensions import (
    FabricRuntimeResult as FabricRuntimeResult,
)


@dataclass(frozen=True, slots=True)
class FabricInvocationRequest:
    """Platform-owned request for one invocation on an active Fabric runtime."""

    input: Any = ""
    request_id: str | None = None
    caller_context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class FabricOneShotRequest:
    """Platform-owned request for one ephemeral Fabric runtime invocation.

    This is an internal bridge type. The fields are intentionally close to
    Fabric's ``RunRequest`` while preserving Platform-owned lifecycle inputs
    such as ``base_dir`` and timeout policy.
    """

    fabric_config: FabricConfig
    base_dir: Path | str
    input: Any = ""
    request_id: str | None = None
    caller_context: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] | None = None
    timeout_seconds: float | None = None


class FabricRuntimeStream:
    """Platform-owned handle for one streaming Fabric runtime invocation."""

    def __init__(self, stream: Any, timeout_seconds: float | None = None) -> None:
        self._stream = stream
        self._timeout_seconds = timeout_seconds

    async def records(self) -> AsyncIterator[dict[str, Any]]:
        """Yield raw NeMo Relay ATOF records from Fabric."""
        try:
            async for record in self._stream:
                yield dict(record)
        except TimeoutError as error:
            raise FabricRuntimeTimeoutError(
                _timeout_error_message(self._timeout_seconds),
            ) from error
        except FabricError as error:
            raise FabricRuntimeExecutionError(
                f"Fabric runtime streaming failed: {error}",
            ) from error

    async def result(self) -> FabricRuntimeResult:
        """Return the authoritative terminal Fabric result for this stream."""
        try:
            result = await asyncio.wait_for(
                self._stream.result(),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as error:
            raise FabricRuntimeTimeoutError(
                _timeout_error_message(self._timeout_seconds),
            ) from error
        except FabricError as error:
            raise FabricRuntimeExecutionError(
                f"Fabric runtime streaming failed: {error}",
            ) from error
        return _normalize_fabric_run_result(result)

    async def aclose(self) -> None:
        """Finalize unread stream records without cancelling the harness turn."""
        try:
            await self._stream.aclose()
        except FabricError as error:
            raise FabricRuntimeExecutionError(
                f"Fabric runtime streaming cleanup failed: {error}",
            ) from error


class FabricRuntimeExecutionError(RuntimeError):
    """Raised when Fabric cannot return a normalized runtime result."""


class FabricRuntimeStartError(FabricRuntimeExecutionError):
    """Raised when Fabric cannot start a runtime."""


class FabricRuntimeTimeoutError(FabricRuntimeExecutionError):
    """Raised when a Fabric runtime invocation times out."""


def _timeout_error_message(timeout_seconds: float | None) -> str:
    if timeout_seconds is None:
        return "Fabric runtime invocation timed out."
    return f"Fabric runtime invocation timed out after {timeout_seconds:g}s."


async def invoke_fabric_runtime(
    runtime: Runtime,
    request: FabricInvocationRequest,
) -> FabricRuntimeResult:
    """Invoke an active Fabric runtime without changing its lifecycle."""
    try:
        result = await asyncio.wait_for(
            runtime.invoke(request=_with_platform_invocation_context(request)),
            timeout=request.timeout_seconds,
        )
    except TimeoutError as error:
        raise FabricRuntimeTimeoutError(
            _timeout_error_message(request.timeout_seconds),
        ) from error
    except FabricError as error:
        raise FabricRuntimeExecutionError(
            f"Fabric runtime invocation failed: {error}",
        ) from error

    return _normalize_fabric_run_result(result)


def stream_fabric_runtime(
    runtime: Runtime,
    request: FabricInvocationRequest,
) -> FabricRuntimeStream:
    """Start streaming one turn on an active Fabric runtime."""
    try:
        stream = runtime.invoke_stream(request=_with_platform_invocation_context(request))
    except FabricError as error:
        raise FabricRuntimeExecutionError(
            f"Fabric runtime streaming failed: {error}",
        ) from error
    return FabricRuntimeStream(stream, request.timeout_seconds)


async def run_fabric_agent_once(
    request: FabricOneShotRequest,
    *,
    fabric: Any | None = None,
) -> FabricRuntimeResult:
    """Start an ephemeral Fabric runtime, invoke it once, and stop it."""
    fabric_client = fabric or Fabric()
    async with _one_shot_runtime(request, fabric=fabric_client) as runtime:
        try:
            result = await asyncio.wait_for(
                runtime.invoke(request=_with_platform_invocation_context(request)),
                timeout=request.timeout_seconds,
            )
        except TimeoutError as error:
            raise FabricRuntimeTimeoutError(
                _timeout_error_message(request.timeout_seconds),
            ) from error
        except FabricError as error:
            raise FabricRuntimeExecutionError(
                f"Fabric runtime invocation failed: {error}",
            ) from error

    return _normalize_fabric_run_result(result)


@asynccontextmanager
async def stream_fabric_agent_once(
    request: FabricOneShotRequest,
    *,
    fabric: Any | None = None,
) -> AsyncIterator[FabricRuntimeStream]:
    """Start an ephemeral Fabric runtime and keep it alive for one stream."""
    fabric_client = fabric or Fabric()
    async with _one_shot_runtime(request, fabric=fabric_client, streaming=True) as runtime:
        yield stream_fabric_runtime(
            runtime,
            FabricInvocationRequest(
                input=request.input,
                request_id=request.request_id,
                caller_context=request.caller_context,
                timeout_seconds=request.timeout_seconds,
            ),
        )


@asynccontextmanager
async def _one_shot_runtime(
    request: FabricOneShotRequest,
    *,
    fabric: Any,
    streaming: bool = False,
) -> AsyncIterator[Runtime]:
    try:
        async with AsyncExitStack() as stack:
            try:
                runtime = await fabric.start_runtime(
                    request.fabric_config,
                    base_dir=request.base_dir,
                    overrides=request.overrides,
                    streaming=streaming,
                )
                runtime = await stack.enter_async_context(runtime)
            except FabricError as error:
                raise FabricRuntimeStartError(f"Fabric runtime startup failed: {error}") from error
            yield runtime
    except FabricError as error:
        raise FabricRuntimeExecutionError(f"Fabric runtime cleanup failed: {error}") from error


def _with_platform_invocation_context(request: FabricInvocationRequest | FabricOneShotRequest) -> RunRequest:
    """Preserve Platform invocation metadata when calling Fabric."""
    request_kwargs: dict[str, Any] = {
        "context": request.caller_context,
        "input": request.input,
    }
    if request.request_id is not None:
        request_kwargs["request_id"] = request.request_id

    return RunRequest(**request_kwargs)


def _normalize_fabric_run_result(result: RunResult) -> FabricRuntimeResult:
    """Convert Fabric's SDK result into the Platform runtime result shape."""
    output = _to_plain_value(result.output)
    return FabricRuntimeResult(
        status=result.status,
        output=output,
        response=output.get("response") if isinstance(output, Mapping) else None,
        error=_to_plain_value(result.error),
        artifacts=_to_plain_value(result.artifacts),
        telemetry=_to_plain_value(result.telemetry),
        events=_to_plain_value(result.events),
        metadata=_to_plain_value(result.metadata),
        runtime_id=result.runtime_id,
        invocation_id=result.invocation_id,
        request_id=result.request_id,
    )


def _to_plain_value(value: Any) -> Any:
    """Convert Fabric SDK mapping objects into plain Platform-owned values."""
    if hasattr(value, "to_mapping"):
        return _to_plain_value(value.to_mapping())
    if isinstance(value, Mapping):
        return {key: _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_plain_value(item) for item in value]
    return value
