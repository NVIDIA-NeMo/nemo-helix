# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle coordination for Platform-managed Fabric runtime sessions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.environment import ensure_local_workspace_dir
from nemo_agents_plugin.fabric.runtime import (
    FabricInvocationRequest,
    FabricOneShotRequest,
    FabricRuntimeResult,
    FabricRuntimeStartError,
    FabricRuntimeStream,
    fabric_streaming_options,
    invoke_fabric_runtime,
    run_fabric_agent_once,
    stream_fabric_agent_once,
    stream_fabric_runtime,
)
from nemo_agents_plugin.fabric.session_registry import (
    FabricRuntimeSession,
    FabricSessionNotFoundError,
    FabricSessionRegistry,
)
from nemo_agents_plugin.fabric.translator import FabricTranslationError, translate_agent_config
from nemo_fabric import Fabric, FabricConfig, FabricError

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT_INVOCATIONS = 8
# The gateway derives persisted ``AgentSession.expires_at`` from this same
# value. If the timeout becomes configurable, single-source it from the
# deployment so persisted expiry and process-local runtime eviction stay aligned.
DEFAULT_IDLE_SESSION_TIMEOUT_SECONDS = 30 * 60
DEFAULT_SESSION_CLEANUP_INTERVAL_SECONDS = 5 * 60


class FabricSessionStartError(FabricRuntimeStartError):
    """Raised when a Fabric runtime cannot be started for a Platform session."""


class FabricSessionStopError(RuntimeError):
    """Raised when a Fabric runtime cannot be stopped for a Platform session."""


@dataclass(slots=True)
class _SessionCreationGate:
    """Per-session runtime-start lock and its current holder/waiter count."""

    lock: asyncio.Lock
    users: int = 0


class FabricSessionManager:
    """Create Fabric runtimes lazily from one reusable Platform agent definition."""

    def __init__(
        self,
        agent_config: AgentConfig,
        *,
        base_dir: Path,
        session_registry: FabricSessionRegistry,
        fabric: Fabric | None = None,
        max_concurrent_invocations: int = DEFAULT_MAX_CONCURRENT_INVOCATIONS,
    ) -> None:
        if max_concurrent_invocations < 0:
            raise ValueError("max_concurrent_invocations must be greater than or equal to zero.")

        self._agent_config = agent_config
        self._base_dir = base_dir
        self._session_registry = session_registry
        self._fabric = fabric
        self._session_creation_gates: dict[str, _SessionCreationGate] = {}
        self._closed_session_ids: set[str] = set()
        self._invocation_semaphore = (
            asyncio.Semaphore(max_concurrent_invocations) if max_concurrent_invocations > 0 else None
        )

    async def open_session(self, *, session_id: str | None = None) -> FabricRuntimeSession:
        """Materialize a Fabric config, start its runtime, and register the session."""
        try:
            fabric_config = await self._materialize_fabric_config(streaming=True)
        except FabricRuntimeStartError as error:
            raise FabricSessionStartError(str(error)) from error
        fabric = self._fabric or Fabric()
        try:
            runtime = await fabric.start_runtime(
                fabric_config,
                base_dir=self._base_dir,
                streaming=True,
                **fabric_streaming_options(fabric_config),
            )
        except (FabricError, FabricRuntimeStartError) as error:
            raise FabricSessionStartError(f"Fabric runtime startup failed: {error}") from error

        try:
            return await self._session_registry.register(runtime, session_id=session_id)
        except BaseException:
            # A started runtime must not leak if registration fails or is cancelled.
            try:
                await runtime.stop()
            except FabricError:
                logger.exception("Failed to stop Fabric runtime after session registration failed.")
            raise

    async def resolve_session(self, session_id: str) -> FabricRuntimeSession:
        """Resolve a session, lazily starting a runtime under a supplied Platform ID."""
        if session_id in self._closed_session_ids:
            raise FabricSessionNotFoundError(f"Fabric session '{session_id}' was not found.")

        try:
            return await self._session_registry.get(session_id)
        except FabricSessionNotFoundError:
            pass

        # Serialize startup per Platform session ID. Without this lock, concurrent first
        # turns could each start a runtime before either one registers the shared ID.
        creation_gate = self._claim_session_creation_gate(session_id)
        try:
            async with creation_gate.lock:
                if session_id in self._closed_session_ids:
                    raise FabricSessionNotFoundError(f"Fabric session '{session_id}' was not found.")
                # Another request may have created the runtime while this request waited.
                try:
                    return await self._session_registry.get(session_id)
                except FabricSessionNotFoundError:
                    return await self.open_session(session_id=session_id)
        finally:
            self._release_session_creation_gate(session_id, creation_gate)

    async def invoke_once(self, request: FabricInvocationRequest) -> FabricRuntimeResult:
        """Run one request on an ephemeral runtime without registering a session."""
        async with self._invocation_slot():
            one_shot_request = await self._to_one_shot_request(request, streaming=False)
            return await run_fabric_agent_once(one_shot_request, fabric=self._fabric)

    @asynccontextmanager
    async def stream_once(self, request: FabricInvocationRequest) -> AsyncIterator[FabricRuntimeStream]:
        """Stream one request from an ephemeral, unregistered runtime."""
        async with self._invocation_slot():
            one_shot_request = await self._to_one_shot_request(request, streaming=True)
            async with stream_fabric_agent_once(one_shot_request, fabric=self._fabric) as stream:
                yield stream

    async def invoke_session(
        self,
        session: FabricRuntimeSession,
        request: FabricInvocationRequest,
    ) -> FabricRuntimeResult:
        """Serialize and invoke one turn on a session's active runtime."""
        async with session.invocation_lock:
            if session.closing:
                raise FabricSessionNotFoundError(f"Fabric session '{session.session_id}' was not found.")
            try:
                async with self._invocation_slot():
                    return await invoke_fabric_runtime(session.runtime, request)
            finally:
                await self._session_registry.refresh_activity(session)

    @asynccontextmanager
    async def stream_session(
        self,
        session: FabricRuntimeSession,
        request: FabricInvocationRequest,
    ) -> AsyncIterator[FabricRuntimeStream]:
        """Serialize and stream one turn on a session's active runtime."""
        async with session.invocation_lock:
            if session.closing:
                raise FabricSessionNotFoundError(f"Fabric session '{session.session_id}' was not found.")
            try:
                async with self._invocation_slot():
                    yield stream_fabric_runtime(session.runtime, request)
            finally:
                await self._session_registry.refresh_activity(session)

    async def close_session(self, session_id: str) -> None:
        """Remove a session and stop its runtime after any active turn finishes."""
        creation_gate = self._claim_session_creation_gate(session_id)
        try:
            async with creation_gate.lock:
                # Explicit cleanup is terminal for this Platform session in the current
                # deployment process. Idle expiration intentionally does not tombstone IDs.
                self._closed_session_ids.add(session_id)
                session = await self._session_registry.remove(session_id)
                if session is None:
                    raise FabricSessionNotFoundError(f"Fabric session '{session_id}' was not found.")

                await self._stop_session(session)
        finally:
            self._release_session_creation_gate(session_id, creation_gate)

    async def expire_idle_sessions(self, *, idle_timeout_seconds: float) -> int:
        """Stop and remove sessions that have exceeded the idle timeout."""
        expired = await self._session_registry.remove_expired(idle_timeout_seconds=idle_timeout_seconds)
        for session in expired:
            try:
                await self._stop_session(session)
            except FabricSessionStopError:
                logger.exception("Failed to stop expired Fabric session %s.", session.session_id)
        return len(expired)

    async def close_all_sessions(self) -> int:
        """Drain the registry and stop every remaining runtime."""
        sessions = await self._session_registry.drain()

        async def stop_session(session: FabricRuntimeSession) -> None:
            try:
                await self._stop_session(session)
            except FabricSessionStopError:
                logger.exception("Failed to stop Fabric session %s during shutdown.", session.session_id)

        results = await asyncio.gather(
            *(stop_session(session) for session in sessions),
            return_exceptions=True,
        )
        for session, result in zip(sessions, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "Unexpected error stopping Fabric session %s during shutdown.",
                    session.session_id,
                    exc_info=(type(result), result, result.__traceback__),
                )
        return len(sessions)

    def _claim_session_creation_gate(self, session_id: str) -> _SessionCreationGate:
        """Claim the shared startup gate for one session ID."""
        gate = self._session_creation_gates.get(session_id)
        if gate is None:
            gate = _SessionCreationGate(lock=asyncio.Lock())
            self._session_creation_gates[session_id] = gate
        gate.users += 1
        return gate

    def _release_session_creation_gate(self, session_id: str, gate: _SessionCreationGate) -> None:
        """Release a startup-gate claim and discard it after its final waiter."""
        gate.users -= 1
        if gate.users == 0 and self._session_creation_gates.get(session_id) is gate:
            self._session_creation_gates.pop(session_id)

    async def _stop_session(self, session: FabricRuntimeSession) -> None:
        """Stop one session after any active invocation releases its lock."""
        async with session.invocation_lock:
            try:
                await session.runtime.stop()
            except FabricError as error:
                raise FabricSessionStopError(f"Fabric runtime shutdown failed: {error}") from error

    async def _materialize_fabric_config(self, *, streaming: bool) -> FabricConfig:
        try:
            fabric_config = translate_agent_config(self._agent_config)
        except FabricTranslationError as error:
            raise FabricRuntimeStartError(f"Fabric config translation failed: {error}") from error

        await asyncio.to_thread(ensure_local_workspace_dir, self._agent_config, self._base_dir)
        return _prepare_serving_fabric_config(fabric_config) if streaming else fabric_config

    async def _to_one_shot_request(
        self,
        request: FabricInvocationRequest,
        *,
        streaming: bool,
    ) -> FabricOneShotRequest:
        return FabricOneShotRequest(
            fabric_config=await self._materialize_fabric_config(streaming=streaming),
            base_dir=self._base_dir,
            input=request.input,
            request_id=request.request_id,
            caller_context=request.caller_context,
            relay_session_root=request.relay_session_root,
            timeout_seconds=request.timeout_seconds,
        )

    @asynccontextmanager
    async def _invocation_slot(self) -> AsyncIterator[None]:
        if self._invocation_semaphore is None:
            yield
            return
        async with self._invocation_semaphore:
            yield


def _prepare_serving_fabric_config(fabric_config: FabricConfig) -> FabricConfig:
    """Enable serving-owned Relay support without mutating the translated config."""
    return fabric_config.model_copy(deep=True).enable_relay()
