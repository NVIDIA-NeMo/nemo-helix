# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenSandbox SDK calls ``NemoOpenSandboxEnvironment`` makes, with no Harbor imports.

Adapted from ``sandboxed_gym.backends._opensandbox_driver`` (create, exec, file I/O, kill by
metadata) and from Harbor 0.20's ``OpenSandboxEnvironment`` (create retries, a readiness wait that
fails fast, directory transfer). Keeping Harbor out of this module is what lets the environment
run on every Harbor runner, and leaves it ready to move into a shared package.

The SDK is only installed in the Harbor runner venvs, so it is imported lazily by ``load_sdk``.
Tests replace that mapping with mocks.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

LOGGER = logging.getLogger(__name__)

# Sandbox metadata label shared by every sandbox one create call made, including its retries.
CREATE_ATTEMPT_METADATA_KEY = "nemo-scaled-evals-create-attempt"
# Create attempts per sandbox, counting the first, on transient SDK errors only.
CREATE_ATTEMPTS = 3
# Transient SDK errors worth retrying a create on, matched by exact class name as Harbor 0.20 does.
# Input errors such as InvalidArgumentException are left out on purpose.
TRANSIENT_SDK_EXCEPTIONS = frozenset(
    {
        "SandboxApiException",
        "SandboxInternalException",
        "SandboxReadyTimeoutException",
        "SandboxUnhealthyException",
        "PoolAcquireFailedException",
        "PoolEmptyException",
        "PoolStateStoreUnavailableException",
        "PoolStateStoreContentionException",
    }
)
# Sandbox states that end a readiness wait with a failure.
_TERMINAL_STATES = frozenset({"Failed", "Terminated"})

# Returns a failure message once a sandbox can no longer become ready, else None.
FailureCheck = Callable[[Any], Awaitable[str | None]]


def load_sdk() -> dict[str, Any]:
    """Import the SDK classes the driver uses; only the Harbor runner venvs have the SDK."""
    from opensandbox import Sandbox  # ty: ignore[unresolved-import]  # Harbor runner venvs only
    from opensandbox.config import ConnectionConfig  # ty: ignore[unresolved-import]
    from opensandbox.manager import SandboxManager  # ty: ignore[unresolved-import]
    from opensandbox.models.execd import RunCommandOpts  # ty: ignore[unresolved-import]
    from opensandbox.models.filesystem import SearchEntry, WriteEntry  # ty: ignore[unresolved-import]
    from opensandbox.models.sandboxes import NetworkPolicy, SandboxFilter  # ty: ignore[unresolved-import]

    return {
        "Sandbox": Sandbox,
        "ConnectionConfig": ConnectionConfig,
        "SandboxManager": SandboxManager,
        "RunCommandOpts": RunCommandOpts,
        "SearchEntry": SearchEntry,
        "WriteEntry": WriteEntry,
        "NetworkPolicy": NetworkPolicy,
        "SandboxFilter": SandboxFilter,
    }


def harbor_backoff_sec(attempt: int) -> float:
    """Seconds to wait after create attempt ``attempt`` fails: 1, 2, 4 ... capped at 10, as in Harbor 0.20."""
    return min(10.0, float(2 ** (attempt - 1)))


def join_output(messages: Sequence[Any]) -> str:
    """Rejoin a command's output messages into one stream.

    execd sends one message per output line with its line ending stripped, and a blank line as
    ``"\n"``; joining with ``""`` would run every line together. Whether the output ended in a
    newline can't be recovered, so the result has none at the end.
    """
    return "\n".join(message.text.rstrip("\n") for message in messages)


def is_transient_sdk_error(exc: BaseException) -> bool:
    """True for SDK errors a fresh create attempt may get past."""
    return type(exc).__name__ in TRANSIENT_SDK_EXCEPTIONS


def sandbox_id(sandbox: Any) -> str | None:
    """Return the sandbox's ID from whichever of ``id`` or ``sandbox_id`` the SDK object exposes."""
    return getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None)


@dataclass(frozen=True)
class Connection:
    """Where the OpenSandbox server is and how to reach it."""

    domain: str | None
    # Kept out of repr, so logging or raising with a Connection never prints the key.
    api_key: str | None = field(repr=False)
    protocol: str = "https"
    use_server_proxy: bool = False
    # The SDK's 30 s default is too short for a server that pulls the image during create.
    request_timeout_sec: float | None = None


@dataclass(frozen=True)
class CreateRequest:
    """Everything one sandbox create sends. ``network_policy`` is the SDK's policy payload."""

    image: str
    env: Mapping[str, str]
    metadata: Mapping[str, str]
    resource: Mapping[str, str]
    network_policy: Mapping[str, Any] | None
    extensions: Mapping[str, str]
    sandbox_timeout_sec: float
    ready_timeout_sec: float
    health_check_poll_interval_sec: float


@dataclass(frozen=True)
class CommandResult:
    """A command's output streams and exit code."""

    stdout: str
    stderr: str
    exit_code: int


class OpenSandboxDriver:
    """Sandbox create, exec, file transfer and kill on one OpenSandbox server.

    Args:
        connection: The server to talk to.
        sdk: The SDK classes, as ``load_sdk`` returns them; loaded on first use when omitted.
        logger: Where warnings go.
        retry_wait: Seconds to wait after create attempt ``n`` fails (counting from 1); defaults
            to ``harbor_backoff_sec``.
    """

    def __init__(
        self,
        connection: Connection,
        *,
        sdk: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
        retry_wait: Callable[[int], float] | None = None,
    ) -> None:
        self._connection = connection
        self._sdk = dict(sdk) if sdk is not None else None
        self._logger = logger or LOGGER
        self._retry_wait = retry_wait or harbor_backoff_sec

    @property
    def sdk(self) -> dict[str, Any]:
        """The SDK classes, imported on first use."""
        if self._sdk is None:
            self._sdk = load_sdk()
        return self._sdk

    def connection_config(self) -> Any:
        """The SDK ``ConnectionConfig`` for this server."""
        kwargs: dict[str, Any] = {
            "domain": self._connection.domain,
            "api_key": self._connection.api_key,
            "protocol": self._connection.protocol,
            "use_server_proxy": self._connection.use_server_proxy,
        }
        if self._connection.request_timeout_sec is not None:
            kwargs["request_timeout"] = timedelta(seconds=self._connection.request_timeout_sec)
        return self.sdk["ConnectionConfig"](**kwargs)

    async def create(self, request: CreateRequest, *, failure_check: FailureCheck | None = None) -> Any:
        """Create a sandbox, wait until it is ready, and leave no other sandbox behind.

        Every attempt carries one create-attempt label, so a sandbox the server created but whose
        response was lost is still found and killed: all of them when the create fails, all but
        the returned one when it succeeds. Only transient SDK errors are retried.

        Args:
            request: What to create.
            failure_check: Extra check run on every readiness poll; a message fails the create at once.

        Raises:
            RuntimeError: The sandbox entered a terminal state, or ``failure_check`` reported a failure.
            TimeoutError: The sandbox did not become ready within ``request.ready_timeout_sec``.
        """
        attempt_id = uuid4().hex
        metadata = {**request.metadata, CREATE_ATTEMPT_METADATA_KEY: attempt_id}
        attempt = 1
        while True:
            try:
                sandbox = await self._create_once(request, metadata, failure_check)
                break
            except BaseException as exc:
                if attempt >= CREATE_ATTEMPTS or not is_transient_sdk_error(exc):
                    await self.kill_matching({CREATE_ATTEMPT_METADATA_KEY: attempt_id})
                    raise
                self._logger.warning(
                    "OpenSandbox create attempt %d of %d failed with %s; retrying",
                    attempt,
                    CREATE_ATTEMPTS,
                    type(exc).__name__,
                )
            await asyncio.sleep(self._retry_wait(attempt))
            attempt += 1

        await self.kill_matching({CREATE_ATTEMPT_METADATA_KEY: attempt_id}, keep=sandbox_id(sandbox))
        return sandbox

    async def _create_once(
        self, request: CreateRequest, metadata: Mapping[str, str], failure_check: FailureCheck | None
    ) -> Any:
        """One create call, then the readiness wait; the sandbox is killed if it never becomes ready."""
        sdk = self.sdk
        network_policy = (
            sdk["NetworkPolicy"].model_validate(dict(request.network_policy))
            if request.network_policy is not None
            else None
        )
        # skip_health_check returns the sandbox ID at once, so it is logged before the wait below,
        # which polls far less often than the SDK's own 200 ms health check.
        sandbox = await sdk["Sandbox"].create(
            request.image,
            timeout=timedelta(seconds=request.sandbox_timeout_sec),
            ready_timeout=timedelta(seconds=request.ready_timeout_sec),
            env=dict(request.env),
            metadata=dict(metadata),
            resource=dict(request.resource),
            network_policy=network_policy,
            extensions=dict(request.extensions),
            connection_config=self.connection_config(),
            skip_health_check=True,
        )
        self._logger.debug("OpenSandbox sandbox created sandbox_id=%s", sandbox_id(sandbox))
        try:
            await self._wait_until_ready(sandbox, request, failure_check)
        except BaseException:
            await self.safe_kill(sandbox)
            raise
        return sandbox

    async def _wait_until_ready(self, sandbox: Any, request: CreateRequest, failure_check: FailureCheck | None) -> None:
        """Poll until the sandbox is healthy; fail at once on a terminal state, or at the deadline."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + request.ready_timeout_sec
        while True:
            failure = await self._terminal_failure(sandbox)
            if failure is None and failure_check is not None:
                failure = await failure_check(sandbox)
            if failure is not None:
                raise RuntimeError(failure)
            try:
                healthy = await sandbox.is_healthy()
            except Exception:
                # A probe error must not escape as a transient SDK error: the create would be
                # retried and this sandbox orphaned. Treat it as not ready yet.
                healthy = False
            if healthy:
                return
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"OpenSandbox sandbox {sandbox_id(sandbox)} did not become ready within "
                    f"{request.ready_timeout_sec:g}s."
                )
            await asyncio.sleep(request.health_check_poll_interval_sec)

    async def _terminal_failure(self, sandbox: Any) -> str | None:
        """A message once the sandbox is Failed or Terminated, carrying the server's reason."""
        try:
            info = await sandbox.get_info()
        except Exception:
            # A transient read error: keep waiting for health.
            return None
        status = getattr(info, "status", None)
        state = str(getattr(status, "state", "") or "")
        if state in _TERMINAL_STATES:
            message = getattr(status, "message", "") or "(no message)"
            return f"OpenSandbox sandbox {sandbox_id(sandbox)} entered {state} before becoming ready: {message}"
        return None

    async def safe_kill(self, sandbox: Any) -> None:
        """Best-effort kill and close; never raises, so it can't mask the error that led here."""
        try:
            await sandbox.kill()
        except Exception:
            self._logger.warning(
                "Failed to kill OpenSandbox sandbox %s; it may linger until its server-side timeout.",
                sandbox_id(sandbox),
            )
        try:
            await sandbox.close()
        except Exception:
            pass

    async def kill_matching(self, metadata: Mapping[str, str], *, keep: str | None = None) -> list[str]:
        """Best-effort kill of every sandbox whose metadata includes ``metadata``, except ``keep``.

        Returns the IDs it killed. Failures are logged, not raised.
        """
        if not metadata:
            raise ValueError("killing sandboxes by metadata requires at least one label")
        sdk = self.sdk
        try:
            manager = await sdk["SandboxManager"].create(connection_config=self.connection_config())
        except Exception:
            self._logger.warning("Could not list OpenSandbox sandboxes matching %s", dict(metadata), exc_info=True)
            return []
        killed: list[str] = []
        try:
            matches: list[str] = []
            page = 1
            while True:
                result = await manager.list_sandbox_infos(sdk["SandboxFilter"](metadata=dict(metadata), page=page))
                matches.extend(info.id for info in result.sandbox_infos if info.id != keep)
                if not result.pagination.has_next_page:
                    break
                page += 1
            for match in matches:
                try:
                    await manager.kill_sandbox(match)
                except Exception:
                    # One sandbox the server already dropped must not spare the rest.
                    self._logger.warning("Could not kill OpenSandbox sandbox %s", match, exc_info=True)
                    continue
                killed.append(match)
            if killed:
                self._logger.warning("Killed OpenSandbox sandboxes matching %s: %s", dict(metadata), ", ".join(killed))
        except Exception:
            self._logger.warning("Could not list OpenSandbox sandboxes matching %s", dict(metadata), exc_info=True)
        finally:
            try:
                await manager.close()
            except Exception:
                pass
        return killed

    async def run(
        self,
        sandbox: Any,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_sec: float | None = None,
        uid: int | None = None,
    ) -> CommandResult:
        """Run a shell command in the foreground; ``uid=None`` runs it as the sandbox's default user.

        Foreground runs keep stdout and stderr apart; the background API merges them.
        """
        opts: dict[str, Any] = {
            "working_directory": cwd,
            "timeout": timedelta(seconds=timeout_sec) if timeout_sec is not None else None,
            "envs": dict(env) if env else None,
        }
        if uid is not None:
            opts["uid"] = uid
        execution = await sandbox.commands.run(command, opts=self.sdk["RunCommandOpts"](**opts))

        # The SDK infers the exit code of a foreground command. ``error.value`` is free text, so
        # only fall back to a generic failure when the SDK found no code at all.
        exit_code = execution.exit_code
        if exit_code is None:
            exit_code = 1 if execution.error is not None else 0
        return CommandResult(
            stdout=join_output(execution.logs.stdout),
            stderr=join_output(execution.logs.stderr),
            exit_code=exit_code,
        )

    async def create_directories(self, sandbox: Any, paths: Sequence[str], *, mode: int = 755) -> None:
        """Create directories, parents included. ``mode`` is octal written as decimal, as the SDK expects."""
        write_entry = self.sdk["WriteEntry"]
        await sandbox.files.create_directories([write_entry(path=path, mode=mode) for path in paths])

    async def write_files(self, sandbox: Any, files: Sequence[tuple[str, bytes, int]]) -> None:
        """Write ``(path, data, mode)`` files; parent directories must already exist."""
        write_entry = self.sdk["WriteEntry"]
        await sandbox.files.write_files([write_entry(path=path, data=data, mode=mode) for path, data, mode in files])

    async def read_bytes(self, sandbox: Any, path: str) -> bytes:
        """Read one file."""
        return await sandbox.files.read_bytes(path)

    async def list_tree(self, sandbox: Any, root: str) -> list[tuple[PurePosixPath, str, str]]:
        """Every entry under ``root`` as ``(relative path, absolute path, entry type)``.

        The type is the SDK's ``entry_type`` lower-cased, or ``""`` when the SDK left it unset.
        """
        matches = await sandbox.files.search(self.sdk["SearchEntry"](path=root, pattern="**"))
        base = PurePosixPath(root)
        return [
            (
                PurePosixPath(entry.path).relative_to(base),
                entry.path,
                str(getattr(entry, "entry_type", "") or "").lower(),
            )
            for entry in matches
        ]
