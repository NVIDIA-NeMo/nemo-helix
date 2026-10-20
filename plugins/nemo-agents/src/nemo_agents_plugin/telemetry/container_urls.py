# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate telemetry destinations in Platform authoring and translated Fabric configs."""

from collections.abc import Iterator
from itertools import chain
from typing import Any


def _entries(config: dict[str, Any], *path: str) -> Iterator[dict[str, Any]]:
    """Read a known exporter list, ignoring malformed optional sections."""
    current: Any = config
    for key in path:
        if not isinstance(current, dict):
            return
        current = current.get(key)
    if isinstance(current, list):
        yield from (entry for entry in current if isinstance(entry, dict))


def _atif_url_fields(telemetry: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """ATIF sends HTTP storage to endpoint; file and S3 storage stay untouched."""
    for storage in _entries(telemetry, "atif", "storage"):
        if storage.get("type") == "http":
            yield storage, "endpoint"


def _otlp_url_fields(telemetry: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """OTLP traces, logs and metrics each have an explicit endpoint list."""
    endpoints = chain(
        _entries(telemetry, "opentelemetry", "endpoints"),
        _entries(telemetry, "opentelemetry", "logs", "endpoints"),
        _entries(telemetry, "opentelemetry", "metrics", "endpoints"),
    )
    for endpoint in endpoints:
        yield endpoint, "endpoint"


def _atof_url_fields(telemetry: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """ATOF accepts endpoint/url entries and stream sinks; file sinks stay untouched."""
    for endpoint in _entries(telemetry, "atof", "endpoints"):
        yield endpoint, "url" if "url" in endpoint else "endpoint"
    for sink in _entries(telemetry, "atof", "sinks"):
        if sink.get("type") == "stream":
            yield sink, "url"


def telemetry_url_fields(config: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """Locate supported exporter fields in authoring and translated Relay configs.

    Return references to the original field holders. Callers decide which
    origins to route and can update those holders without traversing again.
    """
    relay = config.get("relay")
    translated = relay.get("observability") if isinstance(relay, dict) else None
    for telemetry in (config.get("telemetry"), translated):
        if not isinstance(telemetry, dict):
            continue
        yield from _atif_url_fields(telemetry)
        yield from _otlp_url_fields(telemetry)
        yield from _atof_url_fields(telemetry)
