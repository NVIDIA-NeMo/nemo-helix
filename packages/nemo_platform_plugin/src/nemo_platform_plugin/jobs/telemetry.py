# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for carrying anonymous job telemetry context on platform jobs."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

TELEMETRY_CUSTOM_FIELDS_KEY = "_nemo_telemetry"
TELEMETRY_SESSION_ID_KEY = "session_id"
TELEMETRY_PLUGINS_KEY = "plugins"

_MAX_PLUGIN_NAME_LENGTH = 64
_MAX_PLUGINS = 16
_PLUGIN_NAME_RE = re.compile(r"[^a-z0-9.-]+")


def build_job_telemetry_custom_fields(session_id: str, *, plugins: Iterable[str] | None = None) -> dict[str, Any]:
    telemetry: dict[str, Any] = {TELEMETRY_SESSION_ID_KEY: session_id}
    plugin_names = _normalize_plugin_names(plugins)
    if plugin_names:
        telemetry[TELEMETRY_PLUGINS_KEY] = plugin_names
    return {TELEMETRY_CUSTOM_FIELDS_KEY: telemetry}


def get_job_telemetry_session_id(custom_fields: dict[str, Any] | None) -> str | None:
    telemetry = _get_job_telemetry_data(custom_fields)
    if telemetry is None:
        return None
    session_id = telemetry.get(TELEMETRY_SESSION_ID_KEY)
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    return session_id


def get_job_telemetry_plugins(custom_fields: dict[str, Any] | None) -> list[str]:
    telemetry = _get_job_telemetry_data(custom_fields)
    if telemetry is None:
        return []
    return _normalize_plugin_names(telemetry.get(TELEMETRY_PLUGINS_KEY))


def merge_job_telemetry_custom_fields(
    custom_fields: dict[str, Any] | None,
    telemetry_custom_fields: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(custom_fields or {})
    existing = merged.get(TELEMETRY_CUSTOM_FIELDS_KEY)
    telemetry = dict(existing) if isinstance(existing, dict) else {}
    incoming = telemetry_custom_fields.get(TELEMETRY_CUSTOM_FIELDS_KEY)
    if isinstance(incoming, dict):
        existing_plugins = telemetry.get(TELEMETRY_PLUGINS_KEY)
        incoming_plugins = incoming.get(TELEMETRY_PLUGINS_KEY)
        telemetry.update(incoming)
        plugin_names = _normalize_plugin_names(
            [*_iter_plugin_names(existing_plugins), *_iter_plugin_names(incoming_plugins)]
        )
        if plugin_names:
            telemetry[TELEMETRY_PLUGINS_KEY] = plugin_names
        else:
            telemetry.pop(TELEMETRY_PLUGINS_KEY, None)
    if telemetry:
        merged[TELEMETRY_CUSTOM_FIELDS_KEY] = telemetry
    return merged


def stamp_job_telemetry_plugins(custom_fields: dict[str, Any] | None, plugin_name: str) -> dict[str, Any]:
    merged = dict(custom_fields or {})
    existing = merged.get(TELEMETRY_CUSTOM_FIELDS_KEY)
    telemetry = dict(existing) if isinstance(existing, dict) else {}
    plugin_names = _normalize_plugin_names([plugin_name])
    if plugin_names:
        telemetry[TELEMETRY_PLUGINS_KEY] = plugin_names
    else:
        telemetry.pop(TELEMETRY_PLUGINS_KEY, None)
    if telemetry:
        merged[TELEMETRY_CUSTOM_FIELDS_KEY] = telemetry
    return merged


def _get_job_telemetry_data(custom_fields: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(custom_fields, dict):
        return None
    telemetry = custom_fields.get(TELEMETRY_CUSTOM_FIELDS_KEY)
    if not isinstance(telemetry, dict):
        return None
    return telemetry


def _iter_plugin_names(raw_plugins: object) -> list[object]:
    if raw_plugins is None:
        return []
    if isinstance(raw_plugins, str):
        return [raw_plugins]
    if isinstance(raw_plugins, Iterable):
        return list(raw_plugins)
    return []


def _normalize_plugin_names(raw_plugins: object) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw_plugin in _iter_plugin_names(raw_plugins):
        plugin = _normalize_plugin_name(raw_plugin)
        if plugin is None or plugin in seen:
            continue
        names.append(plugin)
        seen.add(plugin)
        if len(names) >= _MAX_PLUGINS:
            break
    return names


def _normalize_plugin_name(raw_plugin: object) -> str | None:
    if not isinstance(raw_plugin, str):
        return None
    plugin = raw_plugin.strip().lower().replace("_", "-")
    if not plugin:
        return None
    plugin = _PLUGIN_NAME_RE.sub("-", plugin).strip(".-")
    if plugin.startswith("nemo-"):
        plugin = plugin[len("nemo-") :]
    if plugin.endswith("-plugin"):
        plugin = plugin[: -len("-plugin")]
    plugin = plugin.strip(".-")[:_MAX_PLUGIN_NAME_LENGTH].strip(".-")
    return plugin or None
