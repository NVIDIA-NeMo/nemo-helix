# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rendered Compose configuration and service-state inspection."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._compose_contracts import ComposeServiceTopology


@dataclass(frozen=True, order=True)
class _PublishedPort:
    """One rendered host-to-container port publication.

    Attributes:
        service: Compose service that declares the publication.
        host_ip: Host address on which Docker publishes the port.
        published: Published host port number.
        target: Container port number.
        protocol: Lowercase transport protocol such as ``tcp`` or ``udp``.
    """

    service: str
    host_ip: str
    published: int
    target: int
    protocol: str


def _parse_json_rows(text: str) -> list[dict[str, Any]]:
    """Parse Compose JSON-array, JSON-object, or JSON-lines output.

    Args:
        text: Raw output from a Compose command such as ``ps --format json``.

    Returns:
        Dictionary rows; non-object JSON values are ignored.

    Raises:
        json.JSONDecodeError: If neither the complete payload nor an individual line is valid JSON.

    Example:
        A JSON array and newline-delimited JSON objects both produce a list of service
        dictionaries.
    """
    stripped = text.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line in stripped.splitlines():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        return rows
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return [payload] if isinstance(payload, dict) else []


def _parse_compose_config(text: str) -> dict[str, Any]:
    """Parse and validate the service mapping from rendered Compose configuration.

    Args:
        text: JSON object emitted by ``docker compose config --format json``.

    Returns:
        Rendered service names mapped to their service configuration values.

    Raises:
        json.JSONDecodeError: If ``text`` is not valid JSON.
        TypeError: If the root or ``services`` value is not an object.
    """
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Compose config JSON must be an object")
    services = payload.get("services", {})
    if not isinstance(services, dict):
        raise TypeError("Compose config services must be an object")
    return services


def _published_ports(services: Mapping[str, Any]) -> list[_PublishedPort]:
    """Extract fixed host-port publications from rendered Compose services.

    Args:
        services: Validated rendered Compose service mapping.

    Returns:
        Sorted, de-duplicated publications. Dynamically assigned host ports are omitted.

    Raises:
        ValueError: If a published or target port is not numeric.
    """
    published_ports: set[_PublishedPort] = set()
    for service, service_config in services.items():
        if not isinstance(service_config, dict):
            continue
        ports = service_config.get("ports", [])
        if not isinstance(ports, list):
            continue
        for port in ports:
            if not isinstance(port, dict) or port.get("published") in {None, ""}:
                continue
            published = int(port["published"])
            if published == 0:
                continue
            published_ports.add(
                _PublishedPort(
                    service=str(service),
                    host_ip=str(port.get("host_ip") or "0.0.0.0"),
                    published=published,
                    target=int(port.get("target") or published),
                    protocol=str(port.get("protocol") or "tcp").casefold(),
                )
            )
    return sorted(published_ports)


async def _find_port_conflicts(
    published_ports: list[_PublishedPort],
) -> list[_PublishedPort]:
    """Probe host-port availability without blocking the event loop.

    Args:
        published_ports: Rendered fixed host-port publications to probe.

    Returns:
        Publications whose host address and port cannot be bound locally.
    """
    availability = await asyncio.gather(
        *(asyncio.to_thread(_published_port_available, published_port) for published_port in published_ports)
    )
    return [
        published_port
        for published_port, available in zip(
            published_ports,
            availability,
            strict=True,
        )
        if not available
    ]


def _published_port_available(published_port: _PublishedPort) -> bool:
    """Check whether one host address and port can be bound.

    Args:
        published_port: Publication describing address family, protocol, and host port.

    Returns:
        ``True`` when a temporary matching socket can bind the host endpoint.
    """
    family = socket.AF_INET6 if ":" in published_port.host_ip else socket.AF_INET
    socket_type = socket.SOCK_DGRAM if published_port.protocol == "udp" else socket.SOCK_STREAM
    with socket.socket(family, socket_type) as probe:
        if socket_type == socket.SOCK_STREAM:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((published_port.host_ip, published_port.published))
        except OSError:
            return False
    return True


def _service_is_running(rows: list[dict[str, Any]], service: str) -> bool:
    """Check whether any Compose state row reports a service as running.

    Args:
        rows: Parsed Compose service-state rows.
        service: Service name to locate.

    Returns:
        ``True`` when a matching row has state ``running``.
    """
    return any(str(row.get("Service")) == service and str(row.get("State", "")).casefold() == "running" for row in rows)


def _services_ready(
    rows: list[dict[str, Any]],
    topology: ComposeServiceTopology,
) -> str | None:
    """Validate service rows against long-running and one-shot expectations.

    Args:
        rows: Parsed Compose service-state rows.
        topology: Exact service roles expected after startup.

    Returns:
        ``None`` when every role is ready; otherwise a concise failure description.
    """
    services: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        services.setdefault(str(row.get("Service")), []).append(row)
    missing = sorted(topology.active_services - services.keys())
    if missing:
        return f"Compose services missing after startup: {missing}"
    unexpected = sorted(services.keys() - topology.active_services)
    if unexpected:
        return f"Unexpected Compose services after startup: {unexpected}"
    for service in sorted(topology.long_running_services):
        for row in services[service]:
            if str(row.get("State", "")).casefold() != "running":
                return f"Compose service {service!r} is not running: {row.get('State')}"
            health = str(row.get("Health", "")).casefold()
            if health and health != "healthy":
                return f"Compose service {service!r} is not healthy: {row.get('Health')}"
    for service in sorted(topology.one_shot_services):
        for row in services[service]:
            state = str(row.get("State", "")).casefold()
            try:
                exit_code = int(row.get("ExitCode", 1))
            except (TypeError, ValueError):
                exit_code = 1
            if state != "exited" or exit_code != 0:
                return f"Compose one-shot service {service!r} did not exit successfully"
    return None
