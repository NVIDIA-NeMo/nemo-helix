# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent lifecycle primitives for Docker Compose sandboxes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from ..base import SandboxCreateError
from ._compose_cli import _ComposeCli, _redact
from ._compose_contracts import ComposeCommandResult, ComposeServiceTopology, logger
from ._compose_inspection import (
    _find_port_conflicts,
    _parse_compose_config,
    _parse_json_rows,
    _published_ports,
    _services_ready,
)
from ._compose_state import _ComposeCommandScope


async def _preflight(
    cli: _ComposeCli,
    command_scope: _ComposeCommandScope,
    service_topology: ComposeServiceTopology,
    environment: Mapping[str, str],
    *,
    command_timeout_seconds: float,
    port_override_hints: Mapping[str, str],
) -> None:
    """Validate rendered topology, project ownership, and published host ports.

    Args:
        cli: Command gateway bound to the lifecycle command scope.
        command_scope: Immutable Docker and Compose project settings.
        service_topology: Exact active services and lifecycle roles.
        environment: Fully merged environment used for Compose interpolation.
        command_timeout_seconds: Deadline for preflight Compose commands.
        port_override_hints: Service-specific hints shown for occupied host ports.

    Raises:
        SandboxCreateError: If the configuration is invalid, the project already has
            containers, service roles differ, or a published host port is unavailable.
    """
    config_task = asyncio.create_task(
        cli.run_compose(
            ["config", "--format", "json"],
            environment=environment,
            timeout=command_timeout_seconds,
        )
    )
    existing_task = asyncio.create_task(
        cli.run_compose(
            ["ps", "--all", "--quiet"],
            environment=environment,
            timeout=command_timeout_seconds,
        )
    )
    try:
        config, existing = await asyncio.gather(config_task, existing_task)
    except BaseException:
        config_task.cancel()
        existing_task.cancel()
        drain_task = asyncio.gather(config_task, existing_task, return_exceptions=True)
        cancellation: asyncio.CancelledError | None = None
        while not drain_task.done():
            try:
                await asyncio.shield(drain_task)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
        if cancellation is not None:
            raise cancellation
        raise
    if not config.ok:
        raise SandboxCreateError(cli.failure_message("Invalid Compose configuration", config, environment))
    if not existing.ok:
        raise SandboxCreateError(cli.failure_message("Could not inspect managed project", existing, environment))
    if existing.stdout.strip():
        raise SandboxCreateError(
            f"Managed Compose project {command_scope.project_name!r} already has containers; "
            "refusing to adopt or remove them"
        )
    try:
        services = _parse_compose_config(config.stdout)
        published_ports = _published_ports(services)
        active_services = frozenset(str(service) for service in services)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SandboxCreateError(f"Could not inspect rendered Compose configuration: {exc}") from exc
    expected_services = service_topology.active_services
    if active_services != expected_services:
        missing = sorted(expected_services - active_services)
        unexpected = sorted(active_services - expected_services)
        raise SandboxCreateError(
            "Rendered Compose service topology does not match the provider configuration: "
            f"missing={missing}, unexpected={unexpected}"
        )
    conflicts = await _find_port_conflicts(published_ports)
    if conflicts:
        details = "\n".join(
            f"- {port.service}: {port.host_ip}:{port.published} -> "
            f"{port.target}/{port.protocol} "
            f"(override {port_override_hints.get(port.service, 'its Compose port mapping')})"
            for port in conflicts
        )
        raise SandboxCreateError(
            "Managed Compose host ports are unavailable:\n"
            f"{details}\n"
            "Stop the conflicting stack or override every occupied port."
        )


async def _assert_ready(
    cli: _ComposeCli,
    service_topology: ComposeServiceTopology,
    environment: Mapping[str, str],
    *,
    command_timeout_seconds: float,
) -> None:
    """Require every configured service to satisfy its lifecycle role.

    Args:
        cli: Command gateway bound to the active lifecycle.
        service_topology: Service roles expected after startup.
        environment: Environment used to query Compose state.
        command_timeout_seconds: Deadline for the state query.

    Raises:
        SandboxCreateError: If a long-running or one-shot service is not ready.
    """
    rows = await _compose_ps(
        cli,
        environment,
        command_timeout_seconds=command_timeout_seconds,
    )
    problem = _services_ready(rows, service_topology)
    if problem is not None:
        raise SandboxCreateError(problem)


async def _compose_ps(
    cli: _ComposeCli,
    environment: Mapping[str, str],
    *,
    command_timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Return parsed state rows for all project services.

    Args:
        cli: Command gateway bound to the active lifecycle.
        environment: Environment forwarded to ``docker compose ps``.
        command_timeout_seconds: Deadline for the state query.

    Returns:
        Parsed JSON objects emitted for service containers.

    Raises:
        RuntimeError: If Compose cannot inspect the project.
        json.JSONDecodeError: If Compose emits malformed JSON.
    """
    result = await cli.run_compose(
        ["ps", "--all", "--format", "json"],
        environment=environment,
        timeout=command_timeout_seconds,
    )
    if not result.ok:
        raise RuntimeError(cli.failure_message("Could not inspect Compose services", result, environment))
    return _parse_json_rows(result.stdout)


async def _capture_diagnostics(
    cli: _ComposeCli,
    environment: Mapping[str, str],
    *,
    command_timeout_seconds: float,
    diagnostics_dir: Path | None,
    reason: str,
) -> None:
    """Best-effort write redacted project state and recent logs.

    Args:
        cli: Command gateway bound to the active lifecycle.
        environment: Environment used for Compose commands and secret redaction.
        command_timeout_seconds: Deadline for each diagnostics command.
        diagnostics_dir: Optional output directory for diagnostics.
        reason: Filename-safe lifecycle label such as ``startup-failure`` or ``shutdown``.

    Diagnostics failures are logged and never replace the lifecycle error being investigated.
    """
    if diagnostics_dir is None:
        return
    try:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        ps_result, logs_result = await asyncio.gather(
            cli.run_compose(
                ["ps", "--all"],
                environment=environment,
                timeout=command_timeout_seconds,
            ),
            cli.run_compose(
                ["logs", "--no-color", "--tail", "200"],
                environment=environment,
                timeout=command_timeout_seconds,
            ),
        )
        ps_text = _redact(f"{ps_result.stdout}\n{ps_result.stderr}", environment)
        logs_text = _redact(f"{logs_result.stdout}\n{logs_result.stderr}", environment)
        (diagnostics_dir / f"compose-{reason}-ps.txt").write_text(
            ps_text,
            encoding="utf-8",
        )
        (diagnostics_dir / f"compose-{reason}-logs.txt").write_text(
            logs_text,
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - diagnostics must not mask lifecycle errors
        logger.exception("Could not capture Compose diagnostics")


async def _managed_resource_names(
    cli: _ComposeCli,
    command_scope: _ComposeCommandScope,
    kind: str,
    environment: Mapping[str, str],
    *,
    command_timeout_seconds: float,
) -> tuple[list[str], str | None]:
    """List Docker resources carrying this Compose project's label.

    Args:
        cli: Command gateway bound to the active lifecycle.
        command_scope: Immutable Compose project settings.
        kind: Docker resource kind: ``container``, ``network``, or ``volume``.
        environment: Environment forwarded to the Docker CLI.
        command_timeout_seconds: Deadline for the resource query.

    Returns:
        Pair of resource names and an optional redacted inspection error.
    """
    args = [kind, "ls"]
    if kind == "container":
        args.append("--all")
    args.extend(
        [
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={command_scope.project_name}",
        ]
    )
    result = await cli.run_docker(
        args,
        environment=environment,
        timeout=command_timeout_seconds,
    )
    if not result.ok:
        return [], cli.failure_message(
            f"Could not inspect managed {kind}s",
            result,
            environment,
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()], None


async def _compose_down(
    cli: _ComposeCli,
    environment: Mapping[str, str],
    *,
    shutdown_timeout_seconds: float,
    command_timeout_seconds: float,
    remove_project_volumes: bool,
) -> ComposeCommandResult:
    """Stop the managed project using the provider's cleanup policy.

    Args:
        cli: Command gateway bound to the active lifecycle.
        environment: Environment forwarded to Docker Compose.
        shutdown_timeout_seconds: Grace period supplied to ``compose down``.
        command_timeout_seconds: Additional deadline for the cleanup command.
        remove_project_volumes: Whether ``compose down`` should remove project volumes.

    Returns:
        Result of the final successful or exhausted cleanup attempt.
    """
    down_args = [
        "down",
        "--remove-orphans",
        "--timeout",
        str(max(1, int(shutdown_timeout_seconds))),
    ]
    if remove_project_volumes:
        down_args.append("--volumes")
    return await cli.retry_compose(
        down_args,
        environment=environment,
        timeout=shutdown_timeout_seconds + command_timeout_seconds,
    )


async def _verify_project_destroyed(
    environment: Mapping[str, str],
    *,
    remove_project_volumes: bool,
    managed_resource_names: Callable[
        [str, Mapping[str, str]],
        Awaitable[tuple[list[str], str | None]],
    ],
) -> list[str]:
    """Check that managed Docker resources no longer exist.

    Args:
        environment: Environment forwarded to Docker inspection commands.
        remove_project_volumes: Whether managed volumes must also be absent.
        managed_resource_names: Resource-inspection callback bound to the active scope.

    Returns:
        Human-readable verification failures. Volumes are checked only when volume removal is enabled.
    """
    kinds = ["container", "network"]
    if remove_project_volumes:
        kinds.append("volume")

    errors: list[str] = []
    for kind in kinds:
        names, query_error = await managed_resource_names(kind, environment)
        if query_error is not None:
            errors.append(query_error)
        elif names:
            errors.append(f"Managed Compose {kind}s remain after teardown: {', '.join(names)}")
    return errors
