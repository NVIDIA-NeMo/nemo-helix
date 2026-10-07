# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

"""Docker Compose sandbox provider orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..base import (
    SANDBOX_RUNTIME_RETURN_CODE,
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from . import _compose_lifecycle, _compose_transfer
from ._compose_cli import _ComposeCli, _run_shielded
from ._compose_contracts import (
    ComposeCleanupError,
    ComposeCommandResult,
    ComposeServiceTopology,
    ProgressCallback,
    PullPolicy,
    logger,
)
from ._compose_inspection import _service_is_running, _services_ready
from ._compose_state import _ComposeCommandScope, _ComposeProjectLock, _ComposeSession


class ComposeTeardownContext:
    """Constrained project operations available to a trusted teardown hook."""

    def __init__(self, provider: DockerComposeSandboxProvider, environment: Mapping[str, str]) -> None:
        """Bind teardown operations to one provider and its command environment.

        Args:
            provider: Provider that owns the Compose project being torn down.
            environment: Environment to forward to teardown commands.
        """
        self._provider = provider
        self._environment = environment

    async def service_is_running(self, service: str) -> bool:
        """Check whether a named service currently has a running container.

        Args:
            service: Compose service name to inspect.

        Returns:
            ``True`` when at least one matching service row is running.
        """
        return _service_is_running(await self._provider._compose_ps(self._environment), service)

    async def stop_service(self, service: str) -> ComposeCommandResult:
        """Gracefully stop a service, retrying transient command failures.

        Args:
            service: Compose service name to stop.

        Returns:
            Result of the final successful or exhausted stop attempt.
        """
        return await self._provider._cli.retry_compose(
            [
                "stop",
                "--timeout",
                str(max(1, int(self._provider.shutdown_timeout_seconds))),
                service,
            ],
            environment=self._environment,
            timeout=self._provider.shutdown_timeout_seconds + 10,
        )

    async def kill_service(self, service: str) -> ComposeCommandResult:
        """Force-stop a service, retrying transient command failures.

        Args:
            service: Compose service name to kill.

        Returns:
            Result of the final successful or exhausted kill attempt.
        """
        return await self._provider._cli.retry_compose(
            ["kill", service],
            environment=self._environment,
            timeout=self._provider.command_timeout_seconds,
        )

    async def exec_service(
        self,
        service: str,
        command: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> ComposeCommandResult:
        """Execute an argument-vector command in a running service.

        Args:
            service: Compose service name in which to execute the command.
            command: Command and arguments passed directly to ``docker compose exec``.
            timeout_seconds: Optional command deadline; defaults to the provider command timeout.

        Returns:
            Captured command result, including timeout state.

        Example:
            ``await context.exec_service("redis", ("redis-cli", "PING"))`` executes without shell parsing.
        """
        return await self._provider._cli.run_compose(
            ["exec", "--no-TTY", service, *command],
            environment=self._environment,
            timeout=timeout_seconds or self._provider.command_timeout_seconds,
        )

    def failure_message(self, prefix: str, result: ComposeCommandResult) -> str:
        """Format a redacted teardown failure message.

        Args:
            prefix: Human-readable description of the failed operation.
            result: Command result whose captured output should be summarized.

        Returns:
            Redacted message containing the prefix, timeout state, and command output.
        """
        return self._provider._cli.failure_message(prefix, result, self._environment)


TeardownHook = Callable[[ComposeTeardownContext], Awaitable[None]]


class DockerComposeSandboxProvider:
    """Own one exclusive Docker Compose project for a sandbox lifecycle."""

    name = "docker-compose"

    def __init__(
        self,
        *,
        compose_files: Sequence[str | Path],
        service_topology: ComposeServiceTopology,
        project_directory: str | Path | None = None,
        project_name: str | None = None,
        profiles: Sequence[str] = (),
        build: bool = False,
        pull_policy: PullPolicy = "missing",
        startup_timeout_seconds: float = 600,
        command_timeout_seconds: float = 60,
        shutdown_timeout_seconds: float = 30,
        lock_path: str | Path | None = None,
        diagnostics_dir: str | Path | None = None,
        environment_defaults: Mapping[str, str] | None = None,
        port_override_hints: Mapping[str, str] | None = None,
        teardown_hook: TeardownHook | None = None,
        remove_project_volumes: bool = False,
        progress_callback: ProgressCallback | None = None,
        docker_bin: str = "docker",
    ) -> None:
        """Configure one reusable owner for a caller-described Compose project.

        Args:
            compose_files: Ordered Compose files; later files override earlier files.
            service_topology: Exact active services and their expected lifecycle roles.
            project_directory: Base directory for relative Compose paths; defaults to the first file's parent.
            project_name: Compose project name; a unique evaluator name is generated when omitted.
            profiles: Compose profiles to enable, with duplicate names removed in first-seen order.
            build: Whether ``compose up`` may build source images.
            pull_policy: Image pull behavior passed to ``compose up``.
            startup_timeout_seconds: Maximum time allowed for project startup.
            command_timeout_seconds: Default deadline for Compose and Docker commands.
            shutdown_timeout_seconds: Grace period supplied to service stop and project shutdown.
            lock_path: Optional cross-process ownership lock file.
            diagnostics_dir: Optional directory for startup progress, service state, and logs.
            environment_defaults: Lowest-precedence environment values for Compose interpolation.
            port_override_hints: Service-specific configuration hints shown for occupied host ports.
            teardown_hook: Trusted project-specific cleanup run before ``compose down``.
            remove_project_volumes: Whether shutdown removes and verifies project volumes.
            progress_callback: Optional receiver for lifecycle progress messages.
            docker_bin: Docker CLI executable name or path.

        Raises:
            ValueError: If Compose files are empty, the project name is invalid, or the pull policy is unsupported.

        Example:
            ``DockerComposeSandboxProvider(compose_files=("compose.yaml",), service_topology=topology)``
            starts existing images by default; pass ``build=True`` to build provisioned source.
        """
        if isinstance(compose_files, (str, Path)) or not compose_files:
            raise ValueError("compose_files must contain at least one path")
        if pull_policy not in {"always", "missing", "never"}:
            raise ValueError("pull_policy must be one of: always, missing, never")
        self.compose_files = tuple(Path(path).expanduser().resolve() for path in compose_files)
        self.project_directory = (
            Path(project_directory).expanduser().resolve()
            if project_directory is not None
            else self.compose_files[0].parent
        )
        self.project_name = project_name or f"nemo-eval-{uuid.uuid4().hex[:12]}"
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.project_name) is None:
            raise ValueError(
                "project_name must start with a lowercase letter or digit and contain only "
                "lowercase letters, digits, hyphens, or underscores"
            )
        self.service_topology = service_topology
        self.target_service = service_topology.target_service
        self.profiles = tuple(dict.fromkeys(profiles))
        self.build = build
        self.pull_policy: PullPolicy = pull_policy
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.command_timeout_seconds = float(command_timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self.lock_path = (
            Path(lock_path)
            if lock_path is not None
            else Path(tempfile.gettempdir()) / (f"nemo-eval-compose-{self.project_name}.lock")
        )
        self.environment_defaults = dict(environment_defaults or {})
        self.port_override_hints = dict(port_override_hints or {})
        self.teardown_hook = teardown_hook
        self.remove_project_volumes = remove_project_volumes
        self.progress_callback = progress_callback
        self.docker_bin = docker_bin
        self.diagnostics_dir = Path(diagnostics_dir).expanduser().resolve() if diagnostics_dir is not None else None
        self._session: _ComposeSession | None = None
        self._closed = False
        self._cli = _ComposeCli(self._command_scope)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Validate, start, and claim exclusive ownership of the Compose project.

        Args:
            spec: Sandbox request. Its environment overrides host and provider defaults;
                provider options are not accepted by this provider.

        Returns:
            Handle targeting the configured long-running service.

        Raises:
            SandboxCreateError: If validation, locking, startup, readiness, or cleanup fails.
            asyncio.CancelledError: If the caller cancels creation after cleanup finishes.
        """
        if self._closed:
            raise SandboxCreateError("DockerComposeSandboxProvider is closed")
        if self._session is not None:
            raise SandboxCreateError("DockerComposeSandboxProvider already owns a stack")
        if spec.provider_options:
            raise SandboxCreateError(
                "DockerComposeSandboxProvider does not accept SandboxSpec.provider_options; "
                "configure the provider through its constructor"
            )
        command_scope = self._public_command_scope()
        target_service = self.target_service
        service_topology = self.service_topology
        missing_files = [path for path in command_scope.compose_files if not path.is_file()]
        if missing_files:
            raise SandboxCreateError(f"Compose files do not exist: {missing_files}")
        if not command_scope.project_directory.is_dir():
            raise SandboxCreateError(f"Compose project directory does not exist: {command_scope.project_directory}")

        environment = {**self.environment_defaults, **os.environ, **spec.env}
        for key, value in self.environment_defaults.items():
            if not environment.get(key):
                environment[key] = value
        try:
            project_lock = _ComposeProjectLock.acquire(self.lock_path)
        except SandboxCreateError:
            raise
        except OSError as exc:
            raise SandboxCreateError(f"Could not acquire Compose project lock {self.lock_path}: {exc}") from exc
        session = _ComposeSession(
            session_id=f"{command_scope.project_name}:{target_service}:{uuid.uuid4().hex}",
            environment=environment,
            lock=project_lock,
            command_scope=command_scope,
            target_service=target_service,
            service_topology=service_topology,
        )
        self._session = session
        try:
            await self._preflight(environment)
            session.owns_project = True
            up_args = [
                "up",
                "--detach",
                "--wait",
                "--wait-timeout",
                str(max(1, int(self.startup_timeout_seconds))),
                "--build" if self.build else "--no-build",
                "--pull",
                self.pull_policy,
            ]
            build_mode = "build enabled" if self.build else "reusing existing images"
            self._progress(f"Starting managed Compose project {command_scope.project_name!r} ({build_mode})...")
            progress_log_path: Path | None = None
            if self.diagnostics_dir is not None:
                self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
                progress_log_path = self.diagnostics_dir / "compose-up.log"
                self._progress(f"Compose startup log: {progress_log_path.as_uri()}")
            startup_started_at = time.monotonic()
            with contextlib.ExitStack() as stack:
                stream_output = (
                    stack.enter_context(progress_log_path.open("w", encoding="utf-8"))
                    if progress_log_path is not None
                    else None
                )
                result = await self._cli.run_compose(
                    up_args,
                    environment=environment,
                    timeout=self.startup_timeout_seconds,
                    stream_output=stream_output,
                )
            if not result.ok:
                raise SandboxCreateError(self._cli.failure_message("Compose startup failed", result, environment))
            await self._assert_ready(environment)
            self._progress(
                f"Managed Compose project {command_scope.project_name!r} ready in "
                f"{time.monotonic() - startup_started_at:.1f}s."
            )
        except BaseException as exc:
            cleanup_error = await self._shielded_cleanup(session, diagnostics_reason="startup-failure")
            if cleanup_error is not None:
                exc.add_note(f"Compose cleanup also failed: {cleanup_error}")
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, SandboxCreateError):
                raise
            raise SandboxCreateError(str(exc)) from exc

        handle = SandboxHandle(
            sandbox_id=session.session_id,
            provider_name=self.name,
            raw=session,
        )
        return handle

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        """Run a shell command in the configured target service.

        Args:
            handle: Active handle returned by this provider.
            command: Shell command evaluated by ``sh -lc`` in the target service.
            cwd: Optional working directory inside the service container.
            env: Optional command-specific environment variables.
            timeout_s: Optional command deadline; defaults to ``command_timeout_seconds``.
            stdin: Optional bytes forwarded to the command's standard input.

        Returns:
            Sandbox result containing captured output, return code, and timeout classification.
        """
        state = self._state(handle)
        args = ["exec", "--no-TTY"]
        if cwd is not None:
            args.extend(["--workdir", cwd])
        for key, value in (env or {}).items():
            args.extend(["--env", f"{key}={value}"])
        args.extend([state.target_service, "sh", "-lc", command])
        result = await self._cli.run_compose(
            args,
            environment=state.environment,
            timeout=float(timeout_s or self.command_timeout_seconds),
            stdin=stdin,
        )
        return SandboxExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=(SANDBOX_RUNTIME_RETURN_CODE if result.timed_out else result.return_code),
            error_type="timeout" if result.timed_out else None,
        )

    async def upload_file(
        self,
        handle: SandboxHandle,
        source_path: Path,
        target_path: str,
    ) -> None:
        """Upload one file under the target service user's parent-directory authority.

        The destination must name an exact, non-root file. Missing parents are created
        as the target service user; if that user cannot create them, the upload fails
        without changing parent or ancestor ownership or permissions. After copying,
        only the regular non-symlink file leaf is assigned to the runtime identity and
        made owner-writable.

        Upload is non-atomic: a copy followed by failed leaf repair may leave the file
        in place, and retrying the same source and destination is supported. The caller
        must keep the destination and its complete ancestor chain stable during the
        operation. Stable symlink ancestors follow normal POSIX path resolution; this
        API is not a security boundary against concurrent in-container path mutation.

        Args:
            handle: Active handle returned by this provider.
            source_path: Existing host file to copy.
            target_path: Exact non-root file path inside the target service.

        Raises:
            ValueError: If ``target_path`` is empty, root, or directory-shaped.
            RuntimeError: If target preparation, copying, or ownership repair fails.
        """
        await self._copy_to_service(handle, source_path, target_path, directory=False)

    async def upload_dir(
        self,
        handle: SandboxHandle,
        source_dir: Path,
        target_dir: str,
    ) -> None:
        """Upload directory contents into a dedicated service-owned target directory.

        The destination must be a non-root exact target that is not itself a symlink.
        The provider merges the source contents into that directory and recursively
        assigns the entire resulting tree to the target service user. Callers must not
        use a shared or externally owned tree for this operation.

        The caller must keep the destination and its complete ancestor chain stable
        during the upload. Stable symlink ancestors follow normal POSIX path resolution;
        this API is not a security boundary against concurrent in-container path
        mutation.

        Args:
            handle: Active handle returned by this provider.
            source_dir: Host directory whose contents should be copied.
            target_dir: Destination directory inside the target service. The provider
                recursively assigns this entire target tree to the service runtime user.

        Raises:
            ValueError: If ``target_dir`` is empty or resolves to the container root.
            RuntimeError: If target preparation, copying, or ownership repair fails.

        Example:
            Uploading ``/tmp/seed/.`` to ``/workspace`` produces ``/workspace/file`` rather
            than ``/workspace/seed/file``.
        """
        await self._copy_to_service(handle, source_dir, target_dir, directory=True)

    async def download_file(
        self,
        handle: SandboxHandle,
        source_path: str,
        target_path: Path,
    ) -> None:
        """Download one target-service file to the host.

        Args:
            handle: Active handle returned by this provider.
            source_path: File path inside the target service.
            target_path: Host destination file; missing parent directories are created.

        Raises:
            RuntimeError: If the Compose copy command fails.
        """
        await self._copy_from_service(handle, source_path, target_path, directory=False)

    async def download_dir(
        self,
        handle: SandboxHandle,
        source_dir: str,
        target_dir: Path,
    ) -> None:
        """Download directory contents into a host directory.

        Args:
            handle: Active handle returned by this provider.
            source_dir: Directory path inside the target service.
            target_dir: Host directory to create or merge copied contents into.

        Raises:
            RuntimeError: If the Compose copy command fails.
        """
        await self._copy_from_service(handle, source_dir, target_dir, directory=True)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Return the aggregate lifecycle state of the managed project.

        Args:
            handle: Handle whose project state should be inspected.

        Returns:
            ``RUNNING`` when all configured service roles are ready, ``STOPPED`` when
            the project is absent, ``ERROR`` for an unhealthy topology, or ``UNKNOWN``
            when inspection fails.
        """
        state = self._state(handle)
        if not state.owns_project:
            return SandboxStatus.STOPPED
        try:
            rows = await self._compose_ps(state.environment)
        except Exception:  # noqa: BLE001 - status must collapse provider failures
            return SandboxStatus.UNKNOWN
        if not rows:
            return SandboxStatus.STOPPED
        return SandboxStatus.RUNNING if _services_ready(rows, state.service_topology) is None else SandboxStatus.ERROR

    async def close(self, handle: SandboxHandle) -> None:
        """Tear down the active project and release its ownership lock.

        Args:
            handle: Active handle returned by this provider.

        Raises:
            ComposeCleanupError: If the teardown hook, Compose shutdown, or resource verification fails.
            asyncio.CancelledError: If cancellation arrives while shielded cleanup is running.
        """
        session = self._state(handle)
        error = await self._shielded_cleanup(session)
        if error is not None:
            raise error

    async def aclose(self) -> None:
        """Close provider-scoped resources and tear down any active project.

        The method is idempotent and permanently prevents subsequent ``create`` calls.

        Raises:
            ComposeCleanupError: If the active project cannot be fully removed.
            asyncio.CancelledError: If cancellation arrives while shielded cleanup is running.
        """
        if self._closed:
            return
        self._closed = True
        session = self._session
        if session is None:
            return
        error = await self._shielded_cleanup(session)
        if error is not None:
            raise error

    def _command_scope(self) -> _ComposeCommandScope:
        """Return the active lifecycle scope or current public configuration.

        Returns:
            Frozen settings for the active lifecycle, or a fresh public-configuration
            snapshot when the provider does not own a project.
        """
        if self._session is not None:
            return self._session.command_scope
        return self._public_command_scope()

    def _public_command_scope(self) -> _ComposeCommandScope:
        """Snapshot the provider's current public command configuration.

        Returns:
            Immutable settings suitable for the next lifecycle.
        """
        return _ComposeCommandScope(
            docker_bin=self.docker_bin,
            project_directory=self.project_directory,
            compose_files=self.compose_files,
            project_name=self.project_name,
            profiles=self.profiles,
        )

    async def _preflight(self, environment: dict[str, str]) -> None:
        """Validate rendered topology, project ownership, and published host ports.

        Args:
            environment: Fully merged environment used for Compose interpolation.

        Raises:
            SandboxCreateError: If the configuration is invalid, the project already has
                containers, service roles differ, or a published host port is unavailable.
        """
        session = self._session
        service_topology = session.service_topology if session is not None else self.service_topology
        await _compose_lifecycle._preflight(
            self._cli,
            self._command_scope(),
            service_topology,
            environment,
            command_timeout_seconds=self.command_timeout_seconds,
            port_override_hints=self.port_override_hints,
        )

    async def _assert_ready(self, environment: Mapping[str, str]) -> None:
        """Require every configured service to satisfy its lifecycle role.

        Args:
            environment: Environment used to query Compose state.

        Raises:
            SandboxCreateError: If a long-running or one-shot service is not ready.
        """
        session = self._session
        service_topology = session.service_topology if session is not None else self.service_topology
        await _compose_lifecycle._assert_ready(
            self._cli,
            service_topology,
            environment,
            command_timeout_seconds=self.command_timeout_seconds,
        )

    async def _compose_ps(self, environment: Mapping[str, str]) -> list[dict[str, Any]]:
        """Return parsed state rows for all project services.

        Args:
            environment: Environment forwarded to ``docker compose ps``.

        Returns:
            Parsed JSON objects emitted for service containers.

        Raises:
            RuntimeError: If Compose cannot inspect the project.
            json.JSONDecodeError: If Compose emits malformed JSON.
        """
        return await _compose_lifecycle._compose_ps(
            self._cli,
            environment,
            command_timeout_seconds=self.command_timeout_seconds,
        )

    async def _cleanup_owned_project(
        self,
        session: _ComposeSession,
    ) -> ComposeCleanupError | None:
        """Run diagnostics, caller cleanup, Compose shutdown, and resource verification.

        Args:
            session: Active lifecycle whose project and lock must be released.

        Returns:
            Aggregated cleanup error when any teardown phase fails; otherwise ``None``.

        Raises:
            asyncio.CancelledError: If the cleanup task itself is cancelled.
        """
        if not session.owns_project:
            self._retire_session(session)
            return None

        environment = session.environment
        errors: list[str] = []
        try:
            await self._capture_diagnostics(environment, reason="shutdown")
            if self.teardown_hook is not None:
                await self.teardown_hook(ComposeTeardownContext(self, environment))
        except BaseException as exc:  # noqa: BLE001 - Compose down must still run
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
            errors.append(f"Compose teardown hook failed: {type(exc).__name__}: {exc}")
        finally:
            try:
                down = await _compose_lifecycle._compose_down(
                    self._cli,
                    environment,
                    shutdown_timeout_seconds=self.shutdown_timeout_seconds,
                    command_timeout_seconds=self.command_timeout_seconds,
                    remove_project_volumes=self.remove_project_volumes,
                )
                if not down.ok:
                    errors.append(self._cli.failure_message("Compose down failed", down, environment))
                errors.extend(await self._verify_project_destroyed(environment))
            except BaseException as exc:  # noqa: BLE001 - release ownership even if Docker fails
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                errors.append(f"Compose teardown failed: {type(exc).__name__}: {exc}")
            finally:
                self._retire_session(session)

        if errors:
            return ComposeCleanupError("; ".join(errors))
        return None

    async def _shielded_cleanup(
        self,
        session: _ComposeSession,
        *,
        diagnostics_reason: str | None = None,
    ) -> ComposeCleanupError | None:
        """Finish project cleanup even when the calling task is cancelled.

        Args:
            session: Active lifecycle to clean up.
            diagnostics_reason: Optional diagnostic label to capture inside the shield
                before normal project cleanup starts.

        Returns:
            Cleanup error when teardown completes with failures; otherwise ``None``.

        Raises:
            asyncio.CancelledError: Re-raised after cleanup completes when the caller was cancelled.
        """

        process_exit: KeyboardInterrupt | SystemExit | None = None

        async def cleanup() -> ComposeCleanupError | None:
            """Capture optional diagnostics, then tear down the owned project."""
            nonlocal process_exit
            try:
                if diagnostics_reason is not None:
                    await self._capture_diagnostics(session.environment, reason=diagnostics_reason)
            except (KeyboardInterrupt, SystemExit) as exc:
                process_exit = exc

            try:
                return await self._cleanup_owned_project(session)
            except (KeyboardInterrupt, SystemExit) as exc:
                if process_exit is None:
                    process_exit = exc
                else:
                    process_exit.add_note(f"Compose cleanup also exited: {exc}")
                return None

        result, cancellation = await _run_shielded(cleanup())
        if cancellation is not None:
            if result is not None:
                cancellation.add_note(f"Compose cleanup also failed: {result}")
            raise cancellation
        if process_exit is not None:
            raise process_exit
        return result

    def _retire_session(self, session: _ComposeSession) -> None:
        """Release one lifecycle's ownership resources and clear it when active.

        Args:
            session: Lifecycle whose project ownership and lock should be released.
        """
        session.owns_project = False
        session.lock.release()
        if self._session is session:
            self._session = None

    async def _verify_project_destroyed(
        self,
        environment: Mapping[str, str],
    ) -> list[str]:
        """Check that managed Docker resources no longer exist.

        Args:
            environment: Environment forwarded to Docker inspection commands.

        Returns:
            Human-readable verification failures. Volumes are checked only when volume removal is enabled.
        """
        return await _compose_lifecycle._verify_project_destroyed(
            environment,
            remove_project_volumes=self.remove_project_volumes,
            managed_resource_names=self._managed_resource_names,
        )

    async def _managed_resource_names(
        self,
        kind: str,
        environment: Mapping[str, str],
    ) -> tuple[list[str], str | None]:
        """List Docker resources carrying this Compose project's label.

        Args:
            kind: Docker resource kind: ``container``, ``network``, or ``volume``.
            environment: Environment forwarded to the Docker CLI.

        Returns:
            Pair of resource names and an optional redacted inspection error.
        """
        return await _compose_lifecycle._managed_resource_names(
            self._cli,
            self._command_scope(),
            kind,
            environment,
            command_timeout_seconds=self.command_timeout_seconds,
        )

    async def _run_target_root(
        self,
        session: _ComposeSession,
        command: Sequence[str],
    ) -> ComposeCommandResult:
        """Run a command as root in the configured target service.

        Args:
            session: Active lifecycle identifying the target service and environment.
            command: Executable and arguments to append after the target service name.

        Returns:
            Captured result for the privileged Compose exec command.
        """
        return await _compose_transfer._run_target_root(
            self._cli,
            session,
            command,
            command_timeout_seconds=self.command_timeout_seconds,
        )

    async def _copy_to_service(
        self,
        handle: SandboxHandle,
        source: Path,
        target: str,
        *,
        directory: bool,
    ) -> None:
        """Copy a host path into the target service and repair ownership.

        Args:
            handle: Active handle returned by this provider.
            source: Host file or directory to copy.
            target: Destination path inside the target service.
            directory: When ``True``, create the full target and copy only ``source`` contents;
                otherwise create only the file's parent and copy the file itself.

        Raises:
            RuntimeError: If target preparation, copying, or ownership repair fails.
            SandboxCreateError: If the target service runtime identity cannot be determined.

        Example:
            With ``directory=True``, source ``/tmp/work`` is passed as ``/tmp/work/.`` so
            Docker merges its contents directly into the prepared target directory.
        """
        state = self._state(handle)
        await _compose_transfer._copy_to_service(
            self._cli,
            state,
            source,
            target,
            directory=directory,
            command_timeout_seconds=self.command_timeout_seconds,
        )

    async def _copy_from_service(
        self,
        handle: SandboxHandle,
        source: str,
        target: Path,
        *,
        directory: bool,
    ) -> None:
        """Copy a target-service path to a prepared host destination.

        Args:
            handle: Active handle returned by this provider.
            source: File or directory path inside the target service.
            target: Host destination path.
            directory: When ``True``, create the target directory and copy source contents;
                otherwise create only the target file's parent.

        Raises:
            RuntimeError: If the Compose copy command fails.
        """
        state = self._state(handle)
        await _compose_transfer._copy_from_service(
            self._cli,
            state,
            source,
            target,
            directory=directory,
            command_timeout_seconds=self.command_timeout_seconds,
        )

    async def _capture_diagnostics(
        self,
        environment: Mapping[str, str],
        *,
        reason: str,
    ) -> None:
        """Best-effort write redacted project state and recent logs.

        Args:
            environment: Environment used for Compose commands and secret redaction.
            reason: Filename-safe lifecycle label such as ``startup-failure`` or ``shutdown``.

        Diagnostics failures are logged and never replace the lifecycle error being investigated.
        """
        await _compose_lifecycle._capture_diagnostics(
            self._cli,
            environment,
            command_timeout_seconds=self.command_timeout_seconds,
            diagnostics_dir=self.diagnostics_dir,
            reason=reason,
        )

    async def _target_identity(self, session: _ComposeSession) -> str:
        """Read the runtime ``UID:GID`` of the target service user.

        Args:
            session: Active lifecycle identifying the target service and command environment.

        Returns:
            Numeric identity formatted as ``UID:GID`` for ``chown``.

        Raises:
            SandboxCreateError: If the identity command fails or emits an unexpected value.
        """
        return await _compose_transfer._target_identity(
            self._cli,
            session,
            command_timeout_seconds=self.command_timeout_seconds,
        )

    def _state(self, handle: SandboxHandle) -> _ComposeSession:
        """Validate a handle and return its provider-private state.

        Args:
            handle: Sandbox handle supplied to a provider operation.

        Returns:
            Compose state stored on the handle.

        Raises:
            ValueError: If the handle belongs to another provider or is not the active stack.
        """
        if handle.provider_name != self.name or not isinstance(handle.raw, _ComposeSession):
            raise ValueError("Sandbox handle does not belong to this Compose provider")
        if self._session is not handle.raw or handle.sandbox_id != handle.raw.session_id:
            raise ValueError("Sandbox handle is not the active Compose session")
        return handle.raw

    def _progress(self, message: str) -> None:
        """Publish a lifecycle progress message.

        Args:
            message: Human-readable progress text sent to the callback or module logger.
        """
        if self.progress_callback is not None:
            self.progress_callback(message)
        else:
            logger.info(message)
