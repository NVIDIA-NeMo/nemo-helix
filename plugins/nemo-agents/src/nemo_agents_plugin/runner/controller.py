# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AgentDeploymentController — reconciles AgentDeployment entities against RunnerBackends.

Registered under the ``nemo.controllers`` entry-point group so the platform
runner manages its lifecycle (startup, reconcile loop, graceful shutdown)
without any wiring in :class:`~nemo_agents_plugin.service.AgentsService`.

Every ``interval_seconds`` (driven by :class:`~nemo_platform_plugin.controller.NemoController`)
it queries the Entities Service for ``agent_deployment`` entities and drives
state transitions:

State machine::

    pending   → starting  (backend.create_deployment succeeds)
    starting  → running   (subprocess: health check; container: plugin READY projected)
    starting  → failed    (health check times out / process exits / plugin FAILED)
    running   → failed    (process exits unexpectedly / plugin FAILED)
    running   → pending   (process not found in backend, attempting to restart)
    deleting  → (removed) (backend.delete_deployment + entity deleted)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, cast

from nemo_agents_plugin.config import ControllerConfig
from nemo_agents_plugin.entities import (
    NEMO_AGENTS_SPEC_CONFIG_FORMAT,
    AgentDeployment,
    AgentSession,
    SessionStatus,
    is_container_deployment_mode,
)
from nemo_agents_plugin.runner.backend import RunnerBackend
from nemo_agents_plugin.runner.registry import RunnerBackendRegistry
from nemo_platform_plugin.controller import NemoController
from nemo_platform_plugin.entity_client import (
    NemoEntitiesClient,
    NemoEntityConflictError,
    NemoEntityNotFoundError,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

_SESSION_LIST_PAGE_SIZE = 100
_RUNTIME_HEALTH_TIMEOUT_SECONDS = 5.0


def _is_fabric_deployment(dep: AgentDeployment) -> bool:
    return dep.config.get("config_format") == NEMO_AGENTS_SPEC_CONFIG_FORMAT


class AgentDeploymentController(NemoController):
    """Reconciles ``agent_deployment`` entities against a :class:`RunnerBackend`.

    Extends :class:`~nemo_platform_plugin.controller.NemoController` so the platform
    runner manages its loop, startup, and graceful shutdown automatically.
    Register this class under ``nemo.controllers`` in ``pyproject.toml``; the
    platform will instantiate it and wire it into the thread-based
    ``Loop`` / ``Controller`` framework via ``NemoControllerAdapter``.

    All dependencies (entity client, backend) are initialised in
    :meth:`on_startup` so there is nothing platform-specific in ``__init__``.
    """

    name = "agents-deployment"
    dependencies: ClassVar[list[str]] = ["entities"]

    def __init__(self) -> None:
        self._registry: RunnerBackendRegistry | None = None
        self._entities: NemoEntitiesClient | None = None
        self._controller_config: ControllerConfig | None = None
        self._starting_since: dict[tuple[str, str], float] = {}
        self._runtime_instance_ids: dict[tuple[str, str], str] = {}
        self._pending_restart_reconciliations: dict[str, datetime] = {}
        self._runtime_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._runtime_health_client: httpx.AsyncClient | None = None
        self._startup_sessions_reconciled = False
        self._startup_session_reconciliation_at: datetime | None = None
        self._interval_seconds: float = 5.0  # default; overwritten in on_startup

    # ------------------------------------------------------------------
    # Narrowing properties — raise clearly if accessed before on_startup()
    # ------------------------------------------------------------------

    @property
    def registry(self) -> RunnerBackendRegistry:
        if self._registry is None:
            raise RuntimeError("AgentDeploymentController.registry accessed before on_startup()")
        return self._registry

    @property
    def backend(self) -> RunnerBackend:
        """Default (subprocess) backend — retained for callers/tests expecting ``.backend``."""
        return self.registry.backend

    def _backend_for(self, dep: AgentDeployment) -> RunnerBackend:
        return self.registry.backend_for(dep.deployment_mode)

    @property
    def entities(self) -> NemoEntitiesClient:
        if self._entities is None:
            raise RuntimeError("AgentDeploymentController.entities accessed before on_startup()")
        return self._entities

    @property
    def controller_config(self) -> ControllerConfig:
        if self._controller_config is None:
            raise RuntimeError("AgentDeploymentController.controller_config accessed before on_startup()")
        return self._controller_config

    # ------------------------------------------------------------------
    # NemoController interface
    # ------------------------------------------------------------------

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    async def on_startup(self) -> None:
        """Initialise the entity client and runner backends from config."""
        self._startup_session_reconciliation_at = datetime.now(UTC)
        # Imports deferred intentionally: these modules pull in the SDK,
        # entity-store client, and HTTP machinery.  Importing at module level
        # would add ~1s to every `nemo` CLI invocation during plugin discovery,
        # even when the agents controller is never started.  Do not hoist.
        from nemo_agents_plugin.config import AgentsConfig
        from nemo_agents_plugin.runner.registry import set_runner_registry
        from nemo_platform_plugin.client.adapter import client_from_platform
        from nemo_platform_plugin.entities import EntityClient as _EntityClient
        from nemo_platform_plugin.entities.client import AsyncEntitiesClient
        from nemo_platform_plugin.sdk_provider import get_async_platform_sdk

        config = AgentsConfig.get()
        self._interval_seconds = float(config.controller.interval_seconds)
        self._controller_config = config.controller

        # Build a service-principal entity client for the controller background task.
        #
        # We use get_async_platform_sdk() directly (not entity_client.as_service()) because
        # on_startup() runs outside request scope — there is no existing EntityClient to elevate.
        # get_async_platform_sdk(as_service=..., internal=True) applies the same headers that
        # as_service(internal=True) would: X-NMP-Principal-Id: service:agents plus
        # MARK_INTERNAL_REQUEST_HEADERS.  It also wires the shared HTTP client and URL router,
        # which as_service() would inherit from an existing client but we must set up from scratch.
        sdk = get_async_platform_sdk(as_service="agents", internal=True)
        entities_api = client_from_platform(sdk, AsyncEntitiesClient)
        self._entities = _EntityClient(entities_api)

        registry = RunnerBackendRegistry(config)
        self._registry = registry
        set_runner_registry(registry)

        await self._reconcile_sessions_after_controller_start()
        logger.info("AgentDeploymentController started.")

    async def on_shutdown(self) -> None:
        """Shut down the runner backends."""
        cleanup_tasks = list(self._runtime_cleanup_tasks)
        for task in cleanup_tasks:
            task.cancel()
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        if self._runtime_health_client is not None:
            await self._runtime_health_client.aclose()
            self._runtime_health_client = None
        if self._registry is not None:
            await self._registry.shutdown()
        logger.info("AgentDeploymentController shut down.")

    async def list_objects(self) -> list:
        """List all ``agent_deployment`` entities across workspaces."""
        try:
            result = await self.entities.list(AgentDeployment, workspace="-")
            return result.data
        except Exception:
            logger.exception("Failed to list deployments across all workspaces")
            return []

    async def reconcile(self) -> None:
        """Reconcile deployments, then independently expire due sessions."""
        if not self._startup_sessions_reconciled:
            await self._reconcile_sessions_after_controller_start()
            if self.stop_requested():
                return
        await self._reconcile_deployments()
        if not self.stop_requested():
            await self._retry_pending_restart_reconciliations()
        if not self.stop_requested():
            await self._reconcile_expired_sessions()

    async def _reconcile_deployments(self) -> None:
        """Reconcile each deployment with per-entity error isolation."""
        for deployment in await self.list_objects():
            if self.stop_requested():
                return
            try:
                await self.reconcile_one(deployment)
            except Exception:
                logger.exception("Failed to reconcile deployment %s", deployment)

    async def reconcile_one(self, obj: object) -> None:
        """Drive the state machine for a single deployment entity.

        :class:`~nmp.common.entities.client.EntityConflictError` is caught and
        logged as a debug message (optimistic lock; retry next cycle) so it does
        not propagate to the base class's generic error handler.
        """
        dep = cast(AgentDeployment, obj)
        try:
            await self._reconcile_one(dep)
        except NemoEntityConflictError:
            logger.debug("Optimistic lock conflict on '%s' — will retry next cycle.", dep.name)

    # ------------------------------------------------------------------
    # Internal state-machine helpers
    # ------------------------------------------------------------------

    async def _reconcile_one(self, dep: AgentDeployment) -> None:
        if dep.status == "pending":
            await self._start_deployment(dep)
        elif dep.status == "starting":
            await self._check_health(dep)
        elif dep.status == "running":
            await self._verify_running(dep)
        elif dep.status == "deleting":
            await self._delete_deployment(dep)

    async def _reconcile_expired_sessions(self) -> None:
        """Persist expiration and clean up runtimes for sessions past their deadline."""
        try:
            sessions = await self._list_active_sessions()
        except Exception:
            logger.exception("Failed to list active sessions across all workspaces")
            return
        reconciliation_time = datetime.now(UTC)
        for session in sessions:
            if self.stop_requested():
                return
            try:
                await self._expire_session_if_due(session, at=reconciliation_time)
            except NemoEntityNotFoundError:
                logger.debug(
                    "Session '%s' in workspace '%s' disappeared during expiration reconciliation.",
                    session.name,
                    session.workspace,
                )
            except NemoEntityConflictError:
                logger.debug(
                    "Optimistic lock conflict expiring session '%s' in workspace '%s' — will retry next cycle.",
                    session.name,
                    session.workspace,
                )
            except Exception:
                logger.exception(
                    "Failed to reconcile expiration for session '%s' in workspace '%s'.",
                    session.name,
                    session.workspace,
                )

    async def _list_active_sessions(self, *, deployment_id: str | None = None) -> list[AgentSession]:
        """List every active session across workspaces, following pagination."""
        sessions: list[AgentSession] = []
        page = 1
        filter_obj = {"status": SessionStatus.ACTIVE.value}
        if deployment_id is not None:
            filter_obj["deployment_id"] = deployment_id
        while True:
            result = await self.entities.list(
                AgentSession,
                workspace="-",
                filter_obj=filter_obj,
                page=page,
                page_size=_SESSION_LIST_PAGE_SIZE,
            )
            sessions.extend(result.data)
            if result.pagination is None or page >= result.pagination.total_pages:
                return sessions
            page += 1

    async def _reconcile_sessions_after_controller_start(self) -> None:
        """Conservatively invalidate invoked sessions after controller startup."""
        if self._startup_session_reconciliation_at is None:
            self._startup_session_reconciliation_at = datetime.now(UTC)
        self._startup_sessions_reconciled = await self._reconcile_sessions_after_restart(
            at=self._startup_session_reconciliation_at
        )
        if self._startup_sessions_reconciled:
            logger.info("Reconciled active sessions after agents controller startup.")

    async def _reconcile_sessions_after_restart(
        self,
        *,
        deployment_id: str | None = None,
        at: datetime | None = None,
    ) -> bool:
        """Expire or lose invoked active sessions, returning whether the pass completed."""
        try:
            sessions = await self._list_active_sessions(deployment_id=deployment_id)
        except Exception:
            logger.exception(
                "Failed to list active sessions while reconciling runtime restart%s.",
                f" for deployment ID '{deployment_id}'" if deployment_id is not None else "",
            )
            return False

        reconciliation_time = (at or datetime.now(UTC)).astimezone(UTC)
        reconciled = True
        for session in sessions:
            if self.stop_requested():
                return False
            try:
                await self._transition_session_after_restart(session, at=reconciliation_time)
            except NemoEntityNotFoundError:
                logger.debug(
                    "Session '%s' in workspace '%s' disappeared during restart reconciliation.",
                    session.name,
                    session.workspace,
                )
            except NemoEntityConflictError:
                reconciled = False
                logger.debug(
                    "Optimistic lock conflict reconciling restarted session '%s' in workspace '%s' — "
                    "will retry next cycle.",
                    session.name,
                    session.workspace,
                )
            except Exception:
                reconciled = False
                logger.exception(
                    "Failed to reconcile restarted session '%s' in workspace '%s'.",
                    session.name,
                    session.workspace,
                )
        return reconciled

    async def _transition_session_after_restart(
        self,
        session: AgentSession,
        *,
        at: datetime,
    ) -> AgentSession | None:
        """Move an invoked active session to expired or lost with one conflict retry."""
        from nemo_agents_plugin.session_lifecycle import session_expiration_is_due

        session_to_update = session
        for attempt in range(2):
            if session_to_update.status is not SessionStatus.ACTIVE:
                return session_to_update
            if (
                session_to_update.first_active_at is None
                and session_to_update.last_active_at is None
                and session_to_update.expires_at is None
            ):
                return None
            if session_to_update.first_active_at is not None:
                first_active_at = session_to_update.first_active_at
                if first_active_at.tzinfo is None or first_active_at.utcoffset() is None:
                    first_active_at = first_active_at.replace(tzinfo=UTC)
                else:
                    first_active_at = first_active_at.astimezone(UTC)
                if first_active_at >= at:
                    return None

            new_status = (
                SessionStatus.EXPIRED if session_expiration_is_due(session_to_update, at=at) else SessionStatus.LOST
            )
            session_to_update.status = new_status
            try:
                reconciled_session = await self.entities.update(session_to_update)
            except NemoEntityConflictError:
                if attempt == 1:
                    raise
                session_to_update = await self.entities.get_by_id(AgentSession, session.id)
                continue

            self._schedule_runtime_cleanup(reconciled_session)
            return reconciled_session

        raise RuntimeError("Session restart retry loop exited unexpectedly.")  # pragma: no cover

    async def _reconcile_deployment_sessions_after_restart(
        self,
        dep: AgentDeployment,
        *,
        at: datetime | None = None,
    ) -> bool:
        """Reconcile active sessions bound to one restarted Fabric deployment."""
        if not _is_fabric_deployment(dep):
            return True
        if dep.id is None:
            logger.warning("Cannot reconcile sessions for deployment '%s' without an entity ID.", dep.name)
            return False
        reconciliation_time = (
            at.astimezone(UTC) if at is not None else self._pending_restart_reconciliations.get(dep.id)
        )
        if reconciliation_time is None:
            reconciliation_time = datetime.now(UTC)
        reconciled = await self._reconcile_sessions_after_restart(
            deployment_id=dep.id,
            at=reconciliation_time,
        )
        if reconciled:
            self._pending_restart_reconciliations.pop(dep.id, None)
        else:
            self._pending_restart_reconciliations[dep.id] = reconciliation_time
        return reconciled

    async def _retry_pending_restart_reconciliations(self) -> None:
        """Retry deployment-scoped restart passes whose prior attempt was incomplete."""
        for deployment_id, reconciliation_time in list(self._pending_restart_reconciliations.items()):
            if self.stop_requested():
                return
            if await self._reconcile_sessions_after_restart(
                deployment_id=deployment_id,
                at=reconciliation_time,
            ):
                self._pending_restart_reconciliations.pop(deployment_id, None)

    async def _observe_runtime_instance(self, dep: AgentDeployment) -> None:
        """Detect a Fabric server process replacement from its health identity."""
        if not _is_fabric_deployment(dep):
            return
        current_runtime = await self._read_runtime_instance(dep)
        if current_runtime is None:
            return
        current_instance_id, runtime_started_at = current_runtime

        key = (dep.workspace, dep.name)
        previous_instance_id = self._runtime_instance_ids.get(key)
        if previous_instance_id is None:
            self._runtime_instance_ids[key] = current_instance_id
            return
        if previous_instance_id == current_instance_id:
            return

        self._runtime_instance_ids[key] = current_instance_id
        if await self._reconcile_deployment_sessions_after_restart(dep, at=runtime_started_at):
            logger.warning(
                "Fabric runtime instance changed for deployment '%s/%s'; active sessions were reconciled.",
                dep.workspace,
                dep.name,
            )

    async def _read_runtime_instance(self, dep: AgentDeployment) -> tuple[str, datetime] | None:
        """Read the current Fabric server process identity without persisting it."""
        from nemo_agents_plugin.deployment_routing import get_deployment_endpoint

        endpoint = get_deployment_endpoint(dep)
        if endpoint is None:
            return None
        if self._runtime_health_client is None:
            import httpx

            self._runtime_health_client = httpx.AsyncClient(
                timeout=_RUNTIME_HEALTH_TIMEOUT_SECONDS,
                follow_redirects=False,
            )
        try:
            response = await self._runtime_health_client.get(f"{endpoint.rstrip('/')}/health")
            response.raise_for_status()
            health = response.json()
            runtime_instance_id = health.get("runtime_instance_id")
            runtime_started_at = datetime.fromisoformat(health.get("runtime_started_at", ""))
        except Exception:
            logger.debug(
                "Could not read Fabric runtime instance for deployment '%s/%s'.",
                dep.workspace,
                dep.name,
                exc_info=True,
            )
            return None
        if not isinstance(runtime_instance_id, str) or not runtime_instance_id:
            return None
        if runtime_started_at.tzinfo is None or runtime_started_at.utcoffset() is None:
            return None
        return runtime_instance_id, runtime_started_at.astimezone(UTC)

    async def _expire_session_if_due(
        self,
        session: AgentSession,
        *,
        at: datetime,
    ) -> AgentSession | None:
        """Transition one due session to expired with one optimistic retry.

        The persisted deadline is authoritative even when an invocation is still
        running. That invocation may finish, but its completion cannot reactivate
        the terminal session.
        """
        from nemo_agents_plugin.session_lifecycle import session_expiration_is_due

        session_to_update = session
        for attempt in range(2):
            if session_to_update.status is not SessionStatus.ACTIVE:
                if session_to_update.status is SessionStatus.EXPIRED:
                    self._schedule_runtime_cleanup(session_to_update)
                    return session_to_update
                return None
            if not session_expiration_is_due(session_to_update, at=at):
                return None

            session_to_update.status = SessionStatus.EXPIRED
            try:
                expired_session = await self.entities.update(session_to_update)
            except NemoEntityConflictError:
                if attempt == 1:
                    raise
                session_to_update = await self.entities.get_by_id(AgentSession, session.id)
                continue

            self._schedule_runtime_cleanup(expired_session)
            return expired_session

        raise RuntimeError("Session expiration retry loop exited unexpectedly.")  # pragma: no cover

    def _schedule_runtime_cleanup(
        self,
        session: AgentSession,
    ) -> None:
        """Run best-effort runtime cleanup outside the reconciliation critical path."""
        from nemo_agents_plugin.session_lifecycle import cleanup_fabric_runtime

        task = asyncio.create_task(
            cleanup_fabric_runtime(self.entities, session),
            name=f"cleanup-agent-session-{session.id}",
        )
        self._runtime_cleanup_tasks.add(task)
        task.add_done_callback(self._runtime_cleanup_tasks.discard)

    async def _start_deployment(self, dep: AgentDeployment) -> None:
        """pending -> starting: allocate port (subprocess) and spawn via the mode backend."""
        await self._reconcile_deployment_sessions_after_restart(dep)
        t0 = time.perf_counter()
        backend = self._backend_for(dep)
        port = backend.allocate_port()
        try:
            info = await backend.create_deployment(
                workspace=dep.workspace,
                name=dep.name,
                config=dep.config,
                port=port,
                agent=dep.agent,
                image=dep.image or None,
                deployment_mode=dep.deployment_mode,
                created_by=dep.created_by,
                auth_context=dep.auth_context,
                resources=dep.compute.resources if dep.compute is not None else None,
                secrets=dep.secrets or None,
                use_image_entrypoint=dep.use_image_entrypoint,
            )
        except Exception as exc:
            logger.exception("Failed to start agent for deployment '%s'", dep.name)
            dep.status = "failed"
            dep.error = str(exc)
            await self._save(dep)
            return

        if info.status == "failed":
            dep.status = "failed"
            dep.error = info.error or "Backend failed to create deployment."
            await self._save(dep)
            return

        spawn_ms = (time.perf_counter() - t0) * 1000
        if info.staged_spec is not None:
            dep.spec_revision = info.staged_spec.revision
            dep.spec_tracked_revision = info.staged_spec.tracked_revision
        dep.status = info.status
        dep.port = info.port
        dep.pid = info.pid
        if is_container_deployment_mode(dep.deployment_mode):
            dep.endpoint = ""
            dep.endpoints = list(info.endpoints)
            dep.plugin_deployment = dep.plugin_deployment or dep.name
        else:
            dep.endpoint = info.endpoint
            dep.endpoints = []
        dep.error = ""
        if dep.status == "starting":
            self._starting_since[(dep.workspace, dep.name)] = time.monotonic()
        await self._save(dep)
        logger.info(
            "Deployment '%s' %s (mode=%s, pid=%d, port=%d, spawn=%.0fms, log=%s).",
            dep.name,
            dep.status,
            dep.deployment_mode,
            dep.pid,
            dep.port,
            spawn_ms,
            info.log_path or "<none>",
        )

    async def _check_health(self, dep: AgentDeployment) -> None:
        """starting -> running | failed: single-shot check per reconcile cycle.

        Subprocess mode: loopback ``GET /health``.
        Container modes: trust the deployments-plugin projected status (READY → running);
        no agents-side loopback health check.
        """
        # setdefault — without it, missing key returns now() forever, never times out.
        since = self._starting_since.setdefault((dep.workspace, dep.name), time.monotonic())
        timeout = self.controller_config.health_check_timeout_seconds
        elapsed = time.monotonic() - since
        remaining = timeout - elapsed

        if remaining <= 0:
            dep.status = "failed"
            dep.error = f"Health check timed out after {timeout}s."
            self._starting_since.pop((dep.workspace, dep.name), None)
            try:
                await self._backend_for(dep).delete_deployment(dep.workspace, dep.name)
            except Exception:
                logger.exception("Cleanup after health timeout failed for '%s'", dep.name)
            finally:
                await self._save(dep)
            logger.warning("Deployment '%s' health check timed out.", dep.name)
            return

        backend = self._backend_for(dep)
        info = await backend.get_deployment_status(dep.workspace, dep.name)
        if info is not None and info.status == "failed":
            dep.status = "failed"
            dep.error = info.error or "Process exited unexpectedly during startup."
            self._starting_since.pop((dep.workspace, dep.name), None)
            try:
                if is_container_deployment_mode(dep.deployment_mode):
                    await backend.delete_deployment(dep.workspace, dep.name)
            except Exception:
                logger.exception("Cleanup after failed startup failed for '%s'", dep.name)
            finally:
                await self._save(dep)
            logger.warning(
                "Deployment '%s' failed during startup: %s (log: %s)",
                dep.name,
                dep.error,
                info.log_path or "<none>",
            )
            return

        if is_container_deployment_mode(dep.deployment_mode):
            if info is None:
                logger.debug("Deployment '%s' not visible in deployments plugin yet.", dep.name)
                return
            # Project endpoints every cycle so the gateway can route once READY.
            dep.endpoints = list(info.endpoints)
            if info.status == "running":
                dep.status = "running"
                dep.endpoint = ""
                self._starting_since.pop((dep.workspace, dep.name), None)
                await self._observe_runtime_instance(dep)
                await self._save(dep)
                logger.info(
                    "Deployment '%s' is running (container mode, endpoints=%s, took %.1fs).",
                    dep.name,
                    [ep.url for ep in dep.endpoints],
                    time.monotonic() - since,
                )
            else:
                await self._save(dep)
                logger.debug(
                    "Deployment '%s' container not ready yet (status=%s, %.1fs elapsed).",
                    dep.name,
                    info.status,
                    elapsed,
                )
            return

        # Subprocess: loopback health check.
        if info is not None and info.endpoint:
            dep.endpoint = info.endpoint
        healthy = bool(dep.endpoint) and await backend.health_check(dep.endpoint)

        if healthy:
            if info is not None:
                info.status = "running"
            dep.status = "running"
            self._starting_since.pop((dep.workspace, dep.name), None)
            await self._observe_runtime_instance(dep)
            await self._save(dep)
            logger.info(
                "Deployment '%s' is running at %s (took %.1fs).",
                dep.name,
                dep.endpoint,
                time.monotonic() - since,
            )
        else:
            logger.debug("Deployment '%s' not healthy yet (%.1fs elapsed).", dep.name, elapsed)

    async def _verify_running(self, dep: AgentDeployment) -> None:
        """Mark failed if the runtime disappeared; subprocess may restart via pending."""
        info = await self._backend_for(dep).get_deployment_status(dep.workspace, dep.name)
        if info is None:
            await self._reconcile_deployment_sessions_after_restart(dep)
            if is_container_deployment_mode(dep.deployment_mode):
                # Do not bounce to pending — that would recreate plugin entities while a
                # container may still be running / mid-teardown.
                dep.status = "failed"
                dep.error = "Container deployment not found in deployments plugin."
            else:
                dep.status = "pending"
                dep.error = "Process not found in backend (attempting to restart)."
            await self._save(dep)
        elif info.status == "failed":
            await self._reconcile_deployment_sessions_after_restart(dep)
            dep.status = "failed"
            dep.error = info.error or "Process exited unexpectedly."
            await self._save(dep)
            logger.warning("Deployment '%s' failed: %s", dep.name, dep.error)
        else:
            if info.status == "starting":
                await self._reconcile_deployment_sessions_after_restart(dep)
            endpoints_changed = is_container_deployment_mode(dep.deployment_mode) and info.endpoints != dep.endpoints
            if endpoints_changed:
                dep.endpoints = list(info.endpoints)
                await self._save(dep)
            if info.status == "running":
                await self._observe_runtime_instance(dep)

    async def _delete_deployment(self, dep: AgentDeployment) -> None:
        """deleting → (removed): terminate runtime and delete entity when teardown completes."""
        try:
            cleaned = await self._backend_for(dep).delete_deployment(dep.workspace, dep.name)
        except Exception:
            logger.exception("Backend delete failed for '%s'; will retry while status=deleting", dep.name)
            dep.status = "deleting"
            dep.error = "Backend teardown failed; will retry."
            await self._save(dep)
            return

        if not cleaned:
            # Container teardown still in progress — keep AgentDeployment so the
            # next reconcile can finish DeploymentConfig cleanup.
            dep.status = "deleting"
            if not dep.error:
                dep.error = "Waiting for deployments plugin teardown to finish."
            await self._save(dep)
            logger.info("Deployment '%s' teardown still in progress; will retry.", dep.name)
            return

        self._starting_since.pop((dep.workspace, dep.name), None)
        self._runtime_instance_ids.pop((dep.workspace, dep.name), None)
        if dep.id is not None:
            self._pending_restart_reconciliations.pop(dep.id, None)
        try:
            await self.entities.delete(
                AgentDeployment,
                name=dep.name,
                workspace=dep.workspace,
                expected_db_version=dep.db_version,
            )
        except Exception:
            logger.exception("Failed to delete deployment entity '%s'", dep.name)
        else:
            logger.info("Deployment '%s' deleted.", dep.name)

    async def _save(self, dep: AgentDeployment) -> None:
        try:
            await self.entities.update(dep)
        except NemoEntityConflictError:
            logger.warning(
                "Optimistic lock conflict saving deployment '%s' — will retry on next reconcile cycle.",
                dep.name,
            )
            raise
        except Exception:
            logger.exception("Failed to update deployment entity '%s'", dep.name)
