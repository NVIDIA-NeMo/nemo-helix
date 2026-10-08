# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

"""File-transfer preparation and ownership repair for Compose sandboxes."""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Sequence
from pathlib import Path

from ..base import SandboxCreateError
from ._compose_cli import _ComposeCli
from ._compose_contracts import ComposeCommandResult
from ._compose_state import _ComposeSession

_FILE_TARGET_OPERATION = "nemo-compose-file-target"
_REPAIR_FILE_OPERATION = "nemo-compose-file-repair"
_DIRECTORY_TARGET_OPERATION = "nemo-compose-directory-target"
_PREPARE_FILE_TARGET_SCRIPT = """\
parent=$1
target=$2
mkdir -p "$parent" || exit 1
if [ -L "$target" ] || [ -d "$target" ]; then
    exit 1
fi
if [ -e "$target" ] && [ ! -f "$target" ]; then
    exit 1
fi
"""
_REPAIR_FILE_SCRIPT = """\
target=$1
identity=$2
[ -f "$target" ] && [ ! -L "$target" ] || exit 1
chown -h "$identity" "$target" || exit 1
chmod u+w "$target" || exit 1
[ -f "$target" ] && [ ! -L "$target" ] || exit 1
"""
_PREPARE_DIRECTORY_TARGET_SCRIPT = """\
target=$1
[ ! -L "$target" ] || exit 1
mkdir -p "$target" || exit 1
[ -d "$target" ] && [ ! -L "$target" ] || exit 1
"""


async def _run_target_root(
    cli: _ComposeCli,
    session: _ComposeSession,
    command: Sequence[str],
    *,
    command_timeout_seconds: float,
) -> ComposeCommandResult:
    """Run a command as root in the configured target service.

    Args:
        cli: Command gateway bound to the active Compose lifecycle.
        session: Active lifecycle identifying the target service and environment.
        command: Executable and arguments to append after the target service name.
        command_timeout_seconds: Deadline for the privileged command.

    Returns:
        Captured result for the privileged Compose exec command.
    """
    return await cli.run_compose(
        ["exec", "--no-TTY", "--user", "0", session.target_service, *command],
        environment=session.environment,
        timeout=command_timeout_seconds,
    )


async def _prepare_file_target(
    cli: _ComposeCli,
    session: _ComposeSession,
    parent: str,
    target: str,
    *,
    command_timeout_seconds: float,
) -> None:
    """Prepare and validate a file target as the configured service user.

    Args:
        cli: Command gateway bound to the active Compose lifecycle.
        session: Active lifecycle identifying the target service and environment.
        parent: Normalized absolute parent path for the uploaded file.
        target: Normalized absolute file path to validate.
        command_timeout_seconds: Deadline for target preparation.

    Raises:
        RuntimeError: If the service user cannot prepare a safe exact target.
    """
    result = await cli.run_compose(
        [
            "exec",
            "--no-TTY",
            session.target_service,
            "sh",
            "-c",
            _PREPARE_FILE_TARGET_SCRIPT,
            _FILE_TARGET_OPERATION,
            parent,
            target,
        ],
        environment=session.environment,
        timeout=command_timeout_seconds,
    )
    if not result.ok:
        raise RuntimeError(
            cli.failure_message(
                "Compose upload target preparation failed",
                result,
                session.environment,
            )
        )


async def _repair_uploaded_file(
    cli: _ComposeCli,
    session: _ComposeSession,
    target: str,
    identity: str,
    *,
    command_timeout_seconds: float,
) -> None:
    """Validate and repair only the exact uploaded regular-file leaf."""
    ownership = await _run_target_root(
        cli,
        session,
        [
            "sh",
            "-c",
            _REPAIR_FILE_SCRIPT,
            _REPAIR_FILE_OPERATION,
            target,
            identity,
        ],
        command_timeout_seconds=command_timeout_seconds,
    )
    if not ownership.ok:
        raise RuntimeError(
            cli.failure_message(
                "Compose upload ownership repair failed",
                ownership,
                session.environment,
            )
        )


async def _prepare_directory_target(
    cli: _ComposeCli,
    session: _ComposeSession,
    target: str,
    *,
    command_timeout_seconds: float,
) -> None:
    """Prepare a dedicated non-symlink directory target as root."""
    prepared = await _run_target_root(
        cli,
        session,
        [
            "sh",
            "-c",
            _PREPARE_DIRECTORY_TARGET_SCRIPT,
            _DIRECTORY_TARGET_OPERATION,
            target,
        ],
        command_timeout_seconds=command_timeout_seconds,
    )
    if not prepared.ok:
        raise RuntimeError(
            cli.failure_message(
                "Compose upload target preparation failed",
                prepared,
                session.environment,
            )
        )


async def _copy_to_service(
    cli: _ComposeCli,
    session: _ComposeSession,
    source: Path,
    target: str,
    *,
    directory: bool,
    command_timeout_seconds: float,
) -> None:
    """Copy a host path into the target service and repair ownership.

    Args:
        cli: Command gateway bound to the active Compose lifecycle.
        session: Active lifecycle identifying the target service and environment.
        source: Host file or directory to copy.
        target: Destination path inside the target service.
        directory: When ``True``, create the full target and copy only ``source`` contents;
            otherwise create only the file's parent and copy the file itself.
        command_timeout_seconds: Deadline for each Compose operation.

    Raises:
        RuntimeError: If target preparation, copying, or ownership repair fails.
        SandboxCreateError: If the target service runtime identity cannot be determined.

    Example:
        With ``directory=True``, source ``/tmp/work`` is passed as ``/tmp/work/.`` so
        Docker merges its contents directly into the prepared target directory.
    """
    container_target = _normalized_upload_target(target, directory=directory)
    remote_directory = container_target if directory else posixpath.dirname(container_target)
    if directory:
        await _prepare_directory_target(
            cli,
            session,
            remote_directory,
            command_timeout_seconds=command_timeout_seconds,
        )
    else:
        await _prepare_file_target(
            cli,
            session,
            remote_directory,
            container_target,
            command_timeout_seconds=command_timeout_seconds,
        )
    copy_source = f"{source}{os.sep}." if directory else str(source)
    result = await cli.run_compose(
        ["cp", copy_source, f"{session.target_service}:{container_target}"],
        environment=session.environment,
        timeout=command_timeout_seconds,
    )
    if not result.ok:
        raise RuntimeError(cli.failure_message("Compose upload failed", result, session.environment))
    if session.target_identity is None:
        session.target_identity = await _target_identity(
            cli,
            session,
            command_timeout_seconds=command_timeout_seconds,
        )
    if directory:
        ownership = await _run_target_root(
            cli,
            session,
            ["chown", "-R", session.target_identity, "--", container_target],
            command_timeout_seconds=command_timeout_seconds,
        )
        if not ownership.ok:
            raise RuntimeError(
                cli.failure_message(
                    "Compose upload ownership repair failed",
                    ownership,
                    session.environment,
                )
            )
    else:
        await _repair_uploaded_file(
            cli,
            session,
            container_target,
            session.target_identity,
            command_timeout_seconds=command_timeout_seconds,
        )


async def _copy_from_service(
    cli: _ComposeCli,
    session: _ComposeSession,
    source: str,
    target: Path,
    *,
    directory: bool,
    command_timeout_seconds: float,
) -> None:
    """Copy a target-service path to a prepared host destination.

    Args:
        cli: Command gateway bound to the active Compose lifecycle.
        session: Active lifecycle identifying the target service and environment.
        source: File or directory path inside the target service.
        target: Host destination path.
        directory: When ``True``, create the target directory and copy source contents;
            otherwise create only the target file's parent.
        command_timeout_seconds: Deadline for the Compose copy command.

    Raises:
        RuntimeError: If the Compose copy command fails.
    """
    if directory:
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    copy_source = posixpath.join(source, ".") if directory else source
    result = await cli.run_compose(
        ["cp", f"{session.target_service}:{copy_source}", str(target)],
        environment=session.environment,
        timeout=command_timeout_seconds,
    )
    if not result.ok:
        raise RuntimeError(cli.failure_message("Compose download failed", result, session.environment))


async def _target_identity(
    cli: _ComposeCli,
    session: _ComposeSession,
    *,
    command_timeout_seconds: float,
) -> str:
    """Read the runtime ``UID:GID`` of the target service user.

    Args:
        cli: Command gateway bound to the active Compose lifecycle.
        session: Active lifecycle identifying the target service and command environment.
        command_timeout_seconds: Deadline for the identity command.

    Returns:
        Numeric identity formatted as ``UID:GID`` for ``chown``.

    Raises:
        SandboxCreateError: If the identity command fails or emits an unexpected value.
    """
    result = await cli.run_compose(
        [
            "exec",
            "--no-TTY",
            session.target_service,
            "sh",
            "-c",
            'printf "%s:%s" "$(id -u)" "$(id -g)"',
        ],
        environment=session.environment,
        timeout=command_timeout_seconds,
    )
    identity = result.stdout.strip()
    if not result.ok or not re.fullmatch(r"\d+:\d+", identity):
        raise SandboxCreateError(
            cli.failure_message(
                "Could not determine target service identity",
                result,
                session.environment,
            )
        )
    return identity


def _absolute_container_path(path: str) -> str:
    """Normalize a Docker container path against its root directory.

    Docker copy commands interpret relative container paths from ``/``, while commands
    executed in a container interpret them from the image or service working directory.
    Normalizing once keeps preparation, copy, and ownership repair on the same target.

    Args:
        path: Absolute or root-relative POSIX container path.

    Returns:
        Normalized absolute POSIX path.

    Raises:
        ValueError: If ``path`` is empty.

    Example:
        ``work/output.txt`` becomes ``/work/output.txt``.
    """
    if not path:
        raise ValueError("Container path cannot be empty")
    return posixpath.normpath(f"/{path.lstrip('/')}")


def _normalized_upload_target(target: str, *, directory: bool) -> str:
    """Normalize and validate an exact upload destination."""
    if not target:
        raise ValueError("Container path cannot be empty")
    normalized = _absolute_container_path(target)
    if normalized == "/":
        kind = "Directory" if directory else "File"
        raise ValueError(f"{kind} upload target cannot be the container root")
    if not directory and target.endswith("/"):
        raise ValueError("File upload target must name an exact file")
    return normalized
