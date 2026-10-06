# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent protocols for garak_plugin SDK sub-resources.

Sub-resources (``configs``, ``targets``) only need their parent's HTTP client
and URL builder. Typing against these protocols instead of importing the
concrete ``GarakPluginResource`` classes keeps the import graph acyclic.
"""

from __future__ import annotations

from typing import Protocol

import httpx


class GarakPluginResourceParent(Protocol):
    """Sync parent surface used by sub-resources."""

    _http_client: httpx.Client

    def _url(self, path: str) -> str:
        """Build the absolute request URL for ``path``."""
        raise NotImplementedError


class AsyncGarakPluginResourceParent(Protocol):
    """Async parent surface used by sub-resources."""

    _http_client: httpx.AsyncClient

    def _url(self, path: str) -> str:
        """Build the absolute request URL for ``path``."""
        raise NotImplementedError
