# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the shared telemetry handler."""

from __future__ import annotations

from nemo_platform_plugin.telemetry.handler import (
    CLIENT_ID,
    DEFAULT_ENDPOINT,
    MAX_RETRIES,
    NEMO_TELEMETRY_VERSION,
    SEND_TIMEOUT_SECONDS,
    QueuedEvent,
    TelemetryHandler,
    _cpu_architecture,
    _get_iso_timestamp,
    _redact_endpoint,
    _redacted_netloc,
    _session_prefix,
    _telemetry_enabled,
    _telemetry_endpoint,
    build_payload,
    httpx,
    logger,
    platform,
)

__all__ = [
    "CLIENT_ID",
    "DEFAULT_ENDPOINT",
    "MAX_RETRIES",
    "NEMO_TELEMETRY_VERSION",
    "SEND_TIMEOUT_SECONDS",
    "QueuedEvent",
    "TelemetryHandler",
    "_cpu_architecture",
    "_get_iso_timestamp",
    "_redact_endpoint",
    "_redacted_netloc",
    "_session_prefix",
    "_telemetry_enabled",
    "_telemetry_endpoint",
    "build_payload",
    "httpx",
    "logger",
    "platform",
]
