# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arize Phoenix (Elastic License 2.0: customer self-hosted only, never offered as a Helix-managed service)."""

from __future__ import annotations

from typing import Any

from nemo_observability_plugin.adapters.base import ProviderAdapter
from nemo_observability_plugin.config import ObservabilityConnection

_DEFAULT_PROJECT = "default"


class PhoenixAdapter(ProviderAdapter):
    provider = "phoenix"
    display_name = "Phoenix"
    # With PHOENIX_HOST_ROOT_PATH set, Phoenix still serves at its root and only prefixes the URLs it emits.
    strip_prefix = True
    health_path = "/healthz"

    def landing_path(self, connection: ObservabilityConnection) -> str:
        return "/projects"

    def upstream_headers(self, credential: str | None) -> dict[str, str]:
        return {"Authorization": f"Bearer {credential}"} if credential else {}

    def otlp_endpoint(self, connection: ObservabilityConnection) -> dict[str, Any]:
        return {
            "type": "openinference",
            "transport": "http_binary",
            "endpoint": connection.otlp_endpoint or f"{connection.base_url}/v1/traces",
            "resource_attributes": {"openinference.project.name": connection.default_project or _DEFAULT_PROJECT},
        }

    def upstream_setup(self, connection: ObservabilityConnection, prefix: str) -> list[str]:
        return [
            f"PHOENIX_HOST_ROOT_PATH={prefix}",
            f"PHOENIX_ROOT_URL=<Studio origin>{prefix}",
        ]
