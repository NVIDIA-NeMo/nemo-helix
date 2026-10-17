# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate telemetry destinations in Platform authoring and translated Fabric configs."""

from collections.abc import Iterator
from typing import Any


def telemetry_url_fields(config: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield network destinations without treating container-local sidecars as host services.

    The public telemetry envelope contains independent ATIF, ATOF and OpenTelemetry
    exporters. Fabric translation puts those exporters under relay.observability.
    This function identifies fields only; the caller decides which origins to route.
    """
    relay = config.get("relay") or {}
    platform_telemetry = config.get("telemetry") or {}
    relay_observability = relay.get("observability") or {}
    for telemetry in (platform_telemetry, relay_observability):
        atif = telemetry.get("atif") or {}
        for storage in atif.get("storage") or []:
            if isinstance(storage, dict) and storage.get("type") == "http":
                yield storage, "endpoint"
        opentelemetry = telemetry.get("opentelemetry") or {}
        for endpoint in opentelemetry.get("endpoints") or []:
            if isinstance(endpoint, dict):
                yield endpoint, "endpoint"
        atof = telemetry.get("atof") or {}
        for endpoint in atof.get("endpoints") or []:
            if isinstance(endpoint, dict):
                yield endpoint, "url" if "url" in endpoint else "endpoint"
        for sink in atof.get("sinks") or []:
            if isinstance(sink, dict) and sink.get("type") == "stream":
                yield sink, "url"
