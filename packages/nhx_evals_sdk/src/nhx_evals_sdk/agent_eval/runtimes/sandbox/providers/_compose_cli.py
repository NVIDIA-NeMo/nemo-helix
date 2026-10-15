# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

"""Docker and Compose command execution for the Compose sandbox provider."""

from __future__ import annotations

import asyncio
import os
import re
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import IO

from ..base import SANDBOX_RUNTIME_RETURN_CODE
from ._compose_contracts import ComposeCleanupError, ComposeCommandResult
from ._compose_state import _ComposeCommandScope

_CLEANUP_ATTEMPTS = 3
_CLEANUP_RETRY_DELAY_SECONDS = 0.5
_SECRET_ENV_FRAGMENT = re.compile(r"(?:TOKEN|KEY|PASSWORD|SECRET)", re.IGNORECASE)
_INLINE_SECRET = re.compile(
    r"""(?ix)
    (?P<prefix>
        ["']?(?:authorization|x-api-key|api[_-]?key|token|password)["']?
        \s*[:=]\s*
    )
    (?:
        (?P<quote>["'])(?P<quoted>.*?)(?P=quote)
        |
        (?P<unquoted>[^\r\n,}\]]+)
    )
    """
)


class _ComposeCli:
    """Command gateway for one lifecycle-aware Compose provider."""

    def __init__(self, scope: Callable[[], _ComposeCommandScope]) -> None:
        """Bind command execution to a provider command-scope resolver.

        Args:
            scope: Zero-argument function returning active lifecycle settings, or the
                provider's current settings when no lifecycle is active.
        """
        self._scope = scope

    async def run_compose(
        self,
        args: Sequence[str],
        *,
        environment: Mapping[str, str],
        timeout: float,
        stdin: bytes | None = None,
        stream_output: IO[str] | None = None,
    ) -> ComposeCommandResult:
        """Run one project-scoped Docker Compose command.

        Args:
            args: Compose subcommand and arguments, excluding configured global options.
            environment: Complete subprocess environment.
            timeout: Command deadline in seconds.
            stdin: Optional bytes forwarded to standard input.
            stream_output: Optional text sink for line-buffered, redacted progress output.

        Returns:
            Captured command result with the fully rendered argument vector.
        """
        scope = self._scope()
        argv: tuple[str, ...] = (
            scope.docker_bin,
            "compose",
            "--ansi",
            "never",
            "--progress",
            "plain",
            "--project-directory",
            str(scope.project_directory),
            *(item for compose_file in scope.compose_files for item in ("--file", str(compose_file))),
            "--project-name",
            scope.project_name,
            *(item for profile in scope.profiles for item in ("--profile", profile)),
            *args,
        )
        return await _run_command(
            argv,
            cwd=scope.project_directory,
            environment=environment,
            timeout=timeout,
            stdin=stdin,
            stream_output=stream_output,
        )

    async def run_docker(
        self,
        args: Sequence[str],
        *,
        environment: Mapping[str, str],
        timeout: float,
    ) -> ComposeCommandResult:
        """Run one Docker CLI command outside the Compose subcommand.

        Args:
            args: Docker subcommand and arguments.
            environment: Complete subprocess environment.
            timeout: Command deadline in seconds.

        Returns:
            Captured command result.
        """
        scope = self._scope()
        return await _run_command(
            (scope.docker_bin, *args),
            cwd=scope.project_directory,
            environment=environment,
            timeout=timeout,
            stdin=None,
            stream_output=None,
        )

    async def retry_compose(
        self,
        args: Sequence[str],
        *,
        environment: Mapping[str, str],
        timeout: float,
    ) -> ComposeCommandResult:
        """Run a Compose operation with the cleanup retry policy.

        Args:
            args: Compose subcommand and arguments, excluding global project options.
            environment: Environment forwarded to Compose.
            timeout: Deadline applied independently to each attempt.

        Returns:
            First successful result or the final failed result after all attempts.
        """
        return await _retry_command(
            lambda: self.run_compose(
                args,
                environment=environment,
                timeout=timeout,
            )
        )

    @staticmethod
    def failure_message(
        prefix: str,
        result: ComposeCommandResult,
        environment: Mapping[str, str],
    ) -> str:
        """Build a redacted message from command output.

        Args:
            prefix: Human-readable operation description.
            result: Failed or timed-out command result.
            environment: Command environment used to identify secrets.

        Returns:
            Message containing the prefix, timeout state, and redacted output or return code.
        """
        captured = "\n".join(stream.strip() for stream in (result.stdout, result.stderr) if stream.strip())
        details = _redact(captured, environment)
        timeout = " (timed out)" if result.timed_out else ""
        return f"{prefix}{timeout}: {details or f'exit {result.return_code}'}"


async def _run_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
    stdin: bytes | None,
    stream_output: IO[str] | None = None,
) -> ComposeCommandResult:
    """Run a subprocess with process-group cancellation and timeout handling.

    Args:
        argv: Complete executable argument vector.
        cwd: Host working directory for the child process.
        environment: Complete child-process environment.
        timeout: Maximum runtime in seconds, clamped to a small positive value.
        stdin: Optional bytes written to the child process.
        stream_output: Optional text sink that receives redacted output as lines arrive.

    Returns:
        Captured subprocess result. Timeouts are returned as results rather than raised.

    Raises:
        asyncio.CancelledError: If the caller cancels execution after the process group is terminated.
        OSError: If the subprocess cannot be created.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=dict(environment),
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    communication = asyncio.create_task(
        _communicate_streaming(
            process,
            stdin=stdin,
            redact=_make_line_redactor(environment),
            stdout_chunks=stdout_chunks,
            stderr_chunks=stderr_chunks,
            output=stream_output,
        )
    )

    async def _abort() -> None:
        """Cancel communication, terminate the process group, and reap the task."""
        communication.cancel()
        await _terminate_process_group(process)
        await asyncio.gather(communication, return_exceptions=True)

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            communication,
            timeout=max(0.1, timeout),
        )
        return ComposeCommandResult(
            argv=argv,
            return_code=int(process.returncode or 0),
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )
    except asyncio.CancelledError:
        await _abort()
        raise
    except TimeoutError:
        await _abort()
        return ComposeCommandResult(
            argv=argv,
            return_code=SANDBOX_RUNTIME_RETURN_CODE,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=(
                b"".join(stderr_chunks).decode("utf-8", errors="replace") + f"Command timed out after {timeout:.1f}s"
            ),
            timed_out=True,
        )


async def _communicate_streaming(
    process: asyncio.subprocess.Process,
    *,
    stdin: bytes | None,
    redact: Callable[[str], str],
    stdout_chunks: list[bytes],
    stderr_chunks: list[bytes],
    output: IO[str] | None,
) -> tuple[bytes, bytes]:
    """Drain both subprocess streams while writing redacted progress lines.

    Args:
        process: Running subprocess with piped standard streams.
        stdin: Optional bytes to write before waiting for process completion.
        redact: Function that removes secrets from each emitted text line.
        stdout_chunks: Mutable accumulator for raw standard-output bytes.
        stderr_chunks: Mutable accumulator for raw standard-error bytes.
        output: Optional text sink for redacted progress lines from both streams.

    Returns:
        Complete raw standard-output and standard-error byte strings.

    Raises:
        RuntimeError: If required subprocess pipes are unavailable.
    """
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Streaming subprocess pipes are unavailable")
    readers = (
        asyncio.create_task(
            _drain_stream(
                process.stdout,
                stdout_chunks,
                redact=redact,
                output=output,
            )
        ),
        asyncio.create_task(
            _drain_stream(
                process.stderr,
                stderr_chunks,
                redact=redact,
                output=output,
            )
        ),
    )
    try:
        if stdin is not None:
            if process.stdin is None:
                raise RuntimeError("Streaming subprocess stdin is unavailable")
            process.stdin.write(stdin)
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()
        await process.wait()
        await asyncio.gather(*readers)
    finally:
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
    return b"".join(stdout_chunks), b"".join(stderr_chunks)


async def _drain_stream(
    stream: asyncio.StreamReader,
    chunks: list[bytes],
    *,
    redact: Callable[[str], str],
    output: IO[str] | None,
) -> None:
    """Capture raw stream bytes and emit each decoded line after redaction.

    Args:
        stream: Async subprocess stream to read until EOF.
        chunks: Mutable raw-byte accumulator used for the command result.
        redact: Function applied before a decoded line leaves the provider.
        output: Optional text sink that receives redacted lines and is flushed immediately.
    """
    if output is None:
        while chunk := await stream.read(64 * 1024):
            chunks.append(chunk)
        return

    pending_line = bytearray()
    scan_offset = 0
    while chunk := await stream.read(64 * 1024):
        chunks.append(chunk)
        pending_line.extend(chunk)
        while (newline := pending_line.find(b"\n", scan_offset)) >= 0:
            line = bytes(pending_line[: newline + 1])
            del pending_line[: newline + 1]
            scan_offset = 0
            output.write(redact(line.decode("utf-8", errors="replace")))
            output.flush()
        scan_offset = len(pending_line)
    if pending_line:
        output.write(redact(pending_line.decode("utf-8", errors="replace")))
        output.flush()


async def _retry_command(
    operation: Callable[[], Awaitable[ComposeCommandResult]],
) -> ComposeCommandResult:
    """Retry a cleanup command with short linear backoff.

    Args:
        operation: Zero-argument async function that starts one command attempt.

    Returns:
        First successful result or the final failed result after ``_CLEANUP_ATTEMPTS``.
    """
    result: ComposeCommandResult | None = None
    for attempt in range(_CLEANUP_ATTEMPTS):
        result = await operation()
        if result.ok:
            return result
        if attempt + 1 < _CLEANUP_ATTEMPTS:
            await asyncio.sleep(_CLEANUP_RETRY_DELAY_SECONDS * (attempt + 1))
    if result is None:  # pragma: no cover - attempts is a positive module constant
        raise RuntimeError("Cleanup command was not attempted")
    return result


async def _run_shielded(
    operation: Awaitable[ComposeCleanupError | None],
) -> tuple[ComposeCleanupError | None, asyncio.CancelledError | None]:
    """Let cleanup finish when its caller is cancelled.

    Args:
        operation: Cleanup awaitable that returns an optional aggregated error.

    Returns:
        Pair of the cleanup result and the first cancellation received by the caller.
        The caller decides when to restore that cancellation.
    """
    task = asyncio.ensure_future(operation)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            return result, cancellation
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            if task.done():
                return task.result(), cancellation


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    """Terminate a subprocess group, escalating from ``SIGTERM`` to ``SIGKILL``.

    Args:
        process: Process whose session process group should be stopped and reaped.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=2)
    except TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


def _make_line_redactor(environment: Mapping[str, str]) -> Callable[[str], str]:
    """Build a redactor that scans the environment's secrets only once.

    The returned callable is applied to every streamed log line, so the secret
    set and its (potentially large) alternation regex are compiled up front
    rather than rebuilt per line.

    Args:
        environment: Command environment whose secret-looking keys identify literal values to remove.

    Returns:
        Function that redacts known values and inline authorization, API key, token, and password assignments.

    Example:
        With ``{"API_KEY": "secret-value"}``, the returned function replaces both
        ``secret-value`` and ``Authorization: Bearer value``-style credentials.
    """
    secret_values = {
        value for key, value in environment.items() if _SECRET_ENV_FRAGMENT.search(key) and len(value) >= 4
    }
    parts = []
    for value in sorted(secret_values, key=len, reverse=True):
        prefix = r"(?<![A-Za-z0-9])" if value[0].isalnum() else ""
        suffix = r"(?![A-Za-z0-9])" if value[-1].isalnum() else ""
        parts.append(f"{prefix}{re.escape(value)}{suffix}")
    secret_pattern = re.compile("|".join(parts)) if parts else None

    def redact(text: str) -> str:
        """Redact one text fragment using the precompiled secret patterns.

        Args:
            text: Decoded command output or diagnostic text.

        Returns:
            Text with known and inline credential values replaced by ``<redacted>``.
        """
        if secret_pattern is not None:
            text = secret_pattern.sub("<redacted>", text)
        return _INLINE_SECRET.sub(_redact_inline_secret, text)

    return redact


def _redact(text: str, environment: Mapping[str, str]) -> str:
    """Redact known environment secrets and inline credentials from text.

    Args:
        text: Command output or diagnostics to sanitize.
        environment: Environment used to discover literal secret values.

    Returns:
        Sanitized text safe for errors, logs, and diagnostic files.
    """
    return _make_line_redactor(environment)(text)


def _redact_inline_secret(match: re.Match[str]) -> str:
    """Replace a matched inline credential value while preserving its prefix and quotes.

    Args:
        match: Match produced by ``_INLINE_SECRET``.

    Returns:
        Credential assignment with its value replaced by ``<redacted>``.
    """
    quote = match.group("quote") or ""
    return f"{match.group('prefix')}{quote}<redacted>{quote}"
