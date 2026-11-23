# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle coordination for Platform-managed Fabric runtime sessions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.environment import (
    create_deepagents_workspace,
    deepagents_workspace_config,
    ensure_local_workspace_dir,
    preserve_deepagents_mcp_cwd,
)
from nemo_agents_plugin.fabric.runtime import (
    FabricInvocationRequest,
    FabricOneShotRequest,
    FabricRuntimeResult,
    FabricRuntimeStartError,
    FabricRuntimeStream,
    StreamingMethod,
    await_resource_task,
    fabric_streaming_options,
    invoke_fabric_runtime,
    release_fabric_workspace,
    run_fabric_agent_once,
    stream_fabric_agent_once,
    stream_fabric_runtime,
)
from nemo_agents_plugin.fabric.session_registry import (
    FabricRuntimeSession,
    FabricSessionNotFoundError,
    FabricSessionRegistry,
)
from nemo_agents_plugin.fabric.streaming import resolve_streaming_method
from nemo_agents_plugin.fabric.translator import FabricTranslationError, translate_agent_config
from nemo_agents_plugin.fabric.validation import FabricValidationError, plan_fabric_config
from nemo_fabric import Fabric, FabricConfig, FabricError, Runtime

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
        workspace_baseline: Path | None = None,
        fabric: Fabric | None = None,
        max_concurrent_invocations: int = DEFAULT_MAX_CONCURRENT_INVOCATIONS,
    ) -> None:
        if max_concurrent_invocations < 0:
            raise ValueError("max_concurrent_invocations must be greater than or equal to zero.")

        self._agent_config = agent_config
        self._base_dir = base_dir
        self._session_registry = session_registry
        self._workspace_baseline = workspace_baseline
        self._fabric = fabric
        self._session_creation_gates: dict[str, _SessionCreationGate] = {}
        self._closed_session_ids: set[str] = set()
        self._invocation_semaphore = (
            asyncio.Semaphore(max_concurrent_invocations) if max_concurrent_invocations > 0 else None
        )

    async def open_session(self, *, session_id: str | None = None) -> FabricRuntimeSession:
        """Materialize a Fabric config, start its runtime, and register the session."""
        workspace_dir: Path | None = None
        runtime: Runtime | None = None
        registered = False
        try:
            try:
                workspace_dir = await self._create_workspace()
                fabric_config, method = await self._materialize_fabric_config(
                    streaming=True, workspace_dir=workspace_dir
                )
            except FabricRuntimeStartError as error:
                raise FabricSessionStartError(str(error)) from error
            fabric = self._fabric or Fabric()
            relay_streaming = method is StreamingMethod.RELAY

            async def start() -> None:
                nonlocal runtime
                runtime = await fabric.start_runtime(
                    fabric_config,
                    base_dir=self._base_dir,
                    streaming=relay_streaming,
                    **(fabric_streaming_options(fabric_config) if relay_streaming else {}),
                )

            try:
                await await_resource_task(asyncio.create_task(start()))
            except (FabricError, FabricRuntimeStartError) as error:
                raise FabricSessionStartError(f"Fabric runtime startup failed: {error}") from error

            assert runtime is not None
            session = await self._session_registry.register(runtime, session_id=session_id, streaming_method=method)
            session.workspace_dir = workspace_dir
            registered = True
            return session
        except BaseException:
            if runtime is not None:
                try:
                    await await_resource_task(asyncio.create_task(runtime.stop()))
                except Exception:
                    logger.exception("Failed to stop Fabric runtime after session startup was interrupted.")
            raise
        finally:
            if not registered:
                await release_fabric_workspace(workspace_dir, runtime=runtime)

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
            one_shot_request, _ = await self._to_one_shot_request(request, streaming=False)
            return await run_fabric_agent_once(one_shot_request, fabric=self._fabric)

    @asynccontextmanager
    async def stream_once(self, request: FabricInvocationRequest) -> AsyncGenerator[FabricRuntimeStream, None]:
        """Stream one request from an ephemeral, unregistered runtime."""
        async with self._invocation_slot():
            one_shot_request, method = await self._to_one_shot_request(request, streaming=True)
            async with stream_fabric_agent_once(
                one_shot_request, fabric=self._fabric, streaming_method=method
            ) as stream:
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
    ) -> AsyncGenerator[FabricRuntimeStream, None]:
        """Serialize and stream one turn on a session's active runtime."""
        async with session.invocation_lock:
            if session.closing:
                raise FabricSessionNotFoundError(f"Fabric session '{session.session_id}' was not found.")
            try:
                async with self._invocation_slot():
                    yield stream_fabric_runtime(session.runtime, request, streaming_method=session.streaming_method)
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
        await self._stop_sessions(expired)
        return len(expired)

    async def close_all_sessions(self) -> int:
        """Drain the registry and stop every remaining runtime."""
        sessions = await self._session_registry.drain()
        await self._stop_sessions(sessions)
        return len(sessions)

    async def _stop_sessions(self, sessions: list[FabricRuntimeSession]) -> None:
        results = await await_resource_task(
            asyncio.gather(*(self._stop_session(session) for session in sessions), return_exceptions=True)
        )
        for session, result in zip(sessions, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "Failed to stop Fabric session %s.",
                    session.session_id,
                    exc_info=(type(result), result, result.__traceback__),
                )

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

        async def stop() -> None:
            async with session.invocation_lock:
                try:
                    await session.runtime.stop()
                except FabricError as error:
                    raise FabricSessionStopError(f"Fabric runtime shutdown failed: {error}") from error
                finally:
                    await release_fabric_workspace(session.workspace_dir, runtime=session.runtime)

        await await_resource_task(asyncio.create_task(stop()))

    async def _create_workspace(self) -> Path | None:
        if self._workspace_baseline is None:
            return None
        copy_task = asyncio.create_task(asyncio.to_thread(create_deepagents_workspace, self._workspace_baseline))
        try:
            return await await_resource_task(copy_task)
        except asyncio.CancelledError:
            if not copy_task.cancelled() and copy_task.exception() is None:
                await release_fabric_workspace(copy_task.result())
            raise
        except (OSError, ValueError) as error:
            raise FabricRuntimeStartError(f"DeepAgents workspace creation failed: {error}") from error

    async def _materialize_fabric_config(
        self, *, streaming: bool, workspace_dir: Path | None = None
    ) -> tuple[FabricConfig, StreamingMethod]:
        agent_config = self._agent_config
        if workspace_dir is not None:
            agent_config = deepagents_workspace_config(
                agent_config, base_dir=self._base_dir, workspace_dir=workspace_dir
            )
        try:
            fabric_config = translate_agent_config(agent_config)
        except FabricTranslationError as error:
            raise FabricRuntimeStartError(f"Fabric config translation failed: {error}") from error

        if workspace_dir is not None:
            preserve_deepagents_mcp_cwd(
                fabric_config, workspace=(self._base_dir / self._agent_config.environment.workspace).resolve()
            )
        await asyncio.to_thread(ensure_local_workspace_dir, agent_config, self._base_dir)
        if not streaming:
            return fabric_config, StreamingMethod.BUFFERED
        try:
            plan = await plan_fabric_config(fabric_config, base_dir=self._base_dir, fabric=self._fabric)
        except (FabricValidationError, FabricError) as error:
            raise FabricRuntimeStartError(str(error)) from error
        method = resolve_streaming_method(plan, fabric_config)
        if method is StreamingMethod.RELAY:
            fabric_config = fabric_config.model_copy(deep=True).enable_relay()
        return fabric_config, method

    async def _to_one_shot_request(
        self,
        request: FabricInvocationRequest,
        *,
        streaming: bool,
    ) -> tuple[FabricOneShotRequest, StreamingMethod]:
        workspace_dir = await self._create_workspace()
        try:
            fabric_config, method = await self._materialize_fabric_config(
                streaming=streaming, workspace_dir=workspace_dir
            )
        except BaseException:
            await release_fabric_workspace(workspace_dir)
            raise
        one_shot_request = FabricOneShotRequest(
            fabric_config=fabric_config,
            base_dir=self._base_dir,
            input=request.input,
            request_id=request.request_id,
            caller_context=request.caller_context,
            relay_session_root=request.relay_session_root,
            timeout_seconds=request.timeout_seconds,
            workspace_dir=workspace_dir,
        )
        return one_shot_request, method

    @asynccontextmanager
    async def _invocation_slot(self) -> AsyncGenerator[None, None]:
        if self._invocation_semaphore is None:
            yield
            return
        async with self._invocation_semaphore:
            yield
