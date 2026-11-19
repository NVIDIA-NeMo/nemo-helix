# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selection and request-building helpers for calling registered external plugins.

An external plugin's endpoints come from its manifest, so the CLI derives a short,
typeable name for each one (the last non-template path segment) instead of making
the user type the gateway path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import click
from nemo_helix_plugin.community_plugins.registry import EndpointMapping, ExternalPlugin

_PLACEHOLDER = re.compile(r"\{([^{}/]+)\}")
_BODYLESS_METHODS = frozenset({"GET", "HEAD", "DELETE"})


def endpoint_name(path: str) -> str:
    """Short name for an endpoint: its last path segment that is not a ``{template}``."""
    segments = [s for s in path.split("/") if s and not _PLACEHOLDER.fullmatch(s)]
    return segments[-1] if segments else ""


def gateway_path(plugin_name: str, endpoint_path: str) -> str:
    """The platform path a client calls for a manifest endpoint path."""
    return f"/apis/plugins/{plugin_name}{endpoint_path}"


@dataclass(frozen=True)
class EndpointRow:
    plugin: str
    endpoint: str
    method: str
    access: str
    path: str
    example: str


def _selects(plugin: ExternalPlugin, name: str | None, method: str | None, endpoint: EndpointMapping) -> bool:
    """Whether ``plugins call`` with these arguments would pick exactly ``endpoint``."""
    try:
        return select_endpoint(plugin, name, method) is endpoint
    except click.UsageError:
        return False


def _example(plugin: ExternalPlugin, endpoint: EndpointMapping) -> str:
    """A command the user can paste: only as many arguments as needed to pick this endpoint."""
    parts = ["nemo plugins call", plugin.name]
    if not _selects(plugin, None, None, endpoint):
        name = endpoint_name(endpoint.path)
        parts.append(name)
        if not _selects(plugin, name, None, endpoint):
            parts.append(f"-X {endpoint.method}")
    if endpoint.method not in _BODYLESS_METHODS:
        parts.append("--data '{...}'")
    return " ".join(parts)


def endpoint_rows(plugins: Iterable[ExternalPlugin]) -> list[EndpointRow]:
    """One row per callable endpoint of every enabled plugin."""
    rows: list[EndpointRow] = []
    for plugin in plugins:
        if not plugin.enabled:
            continue
        for endpoint in plugin.endpoints:
            name = endpoint_name(endpoint.path)
            rows.append(
                EndpointRow(
                    plugin=plugin.name,
                    endpoint=name,
                    method=endpoint.method,
                    access=endpoint.access,
                    path=gateway_path(plugin.name, endpoint.path),
                    example=_example(plugin, endpoint),
                )
            )
    return rows


def _describe(endpoints: Iterable[EndpointMapping]) -> str:
    return ", ".join(f"{endpoint_name(e.path)} ({e.method})" for e in endpoints) or "none"


def select_endpoint(plugin: ExternalPlugin, name: str | None, method: str | None) -> EndpointMapping:
    """Pick the manifest endpoint a ``plugins call`` refers to, or explain the choices."""
    candidates = list(plugin.endpoints)
    if name:
        wanted = name.strip("/")
        candidates = [e for e in candidates if endpoint_name(e.path) == wanted or e.path.strip("/") == wanted]
        if not candidates:
            raise click.UsageError(
                f"Plugin '{plugin.name}' has no endpoint '{name}'. Available: {_describe(plugin.endpoints)}. "
                f"See `nemo plugins endpoints {plugin.name}`."
            )
    if method:
        candidates = [e for e in candidates if e.method == method.upper()]
        if not candidates:
            raise click.UsageError(
                f"Plugin '{plugin.name}' has no {method.upper()} endpoint"
                f"{f' named {name!r}' if name else ''}. Available: {_describe(plugin.endpoints)}."
            )
    if len(candidates) > 1:
        if name is None and method is None:
            # `nemo plugins call my-plugin` means its main endpoint, the one named like the plugin.
            main = [e for e in candidates if endpoint_name(e.path) == plugin.name]
            if len(main) == 1:
                return main[0]
        gets = [e for e in candidates if e.method == "GET"]
        if method is None and len(gets) == 1:
            return gets[0]
        raise click.UsageError(
            f"Plugin '{plugin.name}' has several endpoints; name one"
            f"{' and pass -X METHOD' if name else ''}. Choices: {_describe(candidates)}."
        )
    return candidates[0]


def fill_path(path: str, *, workspace: str, params: dict[str, str]) -> str:
    """Substitute ``{workspace}`` and ``--param`` values, quoting each as one path segment."""
    values = {"workspace": workspace, **params}
    missing = [p for p in _PLACEHOLDER.findall(path) if p not in values]
    if missing:
        flags = " ".join(f"--param {p}=VALUE" for p in missing)
        raise click.UsageError(f"This endpoint needs path parameter(s) {', '.join(missing)}: pass {flags}.")
    return _PLACEHOLDER.sub(lambda m: quote(values[m.group(1)], safe=""), path)


def parse_pairs(items: Sequence[str] | None, *, option: str) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` options."""
    result: dict[str, str] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise click.UsageError(f"{option} expects KEY=VALUE, got {item!r}.")
        result[key] = value
    return result


def parse_body(data: str | None, *, method: str) -> bytes | None:
    """JSON request body from ``--data '{...}'`` or ``--data @file.json``."""
    if data is None:
        return None
    if method in _BODYLESS_METHODS:
        raise click.UsageError(f"--data cannot be sent with {method}.")
    text = Path(data[1:]).read_text() if data.startswith("@") else data
    try:
        return json.dumps(json.loads(text)).encode()
    except json.JSONDecodeError as exc:
        raise click.UsageError(f"--data is not valid JSON: {exc}") from exc


def scalar_columns(items: Sequence[dict], limit: int = 6) -> list[str]:
    """First few top-level keys whose values are scalars, for a readable default table."""
    columns: list[str] = []
    for item in items:
        for key, value in item.items():
            if key not in columns and not isinstance(value, (dict, list)):
                columns.append(key)
    return columns[:limit]
