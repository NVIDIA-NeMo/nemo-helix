# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What differs between trace explorers, kept in one small class per provider.

Everything else (sessions, the proxy, routes, the Studio page) is provider-agnostic.
Onboarding a new platform means adding an adapter here plus a runbook entry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from nemo_observability_plugin.config import ObservabilityConnection


class ProviderAdapter(ABC):
    provider: ClassVar[str]
    display_name: ClassVar[str]
    #: Whether the upstream expects the proxy prefix removed (it serves at its root and only
    #: *generates* prefixed URLs) or kept (it registers its routes under the prefix).
    strip_prefix: ClassVar[bool]

    @abstractmethod
    def landing_path(self, connection: ObservabilityConnection) -> str:
        """Path (and fragment) under the proxy prefix that the Studio frame opens."""

    @abstractmethod
    def upstream_headers(self, credential: str | None) -> dict[str, str]:
        """Headers the proxy adds to every upstream request; *credential* comes from ``secret_env``."""

    @abstractmethod
    def otlp_endpoint(self, connection: ObservabilityConnection, prefix: str) -> dict[str, Any]:
        """One ``telemetry.opentelemetry.endpoints`` entry for an agent's ``agent.yaml``.

        *prefix* is the proxy prefix the upstream is configured to serve under.
        """

    @abstractmethod
    def upstream_setup(self, connection: ObservabilityConnection, prefix: str) -> list[str]:
        """Human-readable settings the upstream needs so it can be served under *prefix*."""
