# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity store client — re-exports from nemo_helix_plugin with platform extensions.

The canonical EntityBase, EntityClient base, and related types live in
nemo_helix_plugin.entities. This module re-exports them and extends EntityClient
with the platform-specific ``as_service()`` method.
"""

from __future__ import annotations

# Re-export types from nemo-helix-plugin (canonical source) that have
# consumers through this module's path.
from nemo_helix_plugin.entities import EntityBase as EntityBase
from nemo_helix_plugin.entities import EntityClient as _PluginEntityClient
from nemo_helix_plugin.entities import EntityConflictError as EntityConflictError
from nemo_helix_plugin.entities import EntityNotFoundError as EntityNotFoundError
from nemo_helix_plugin.entities import EntityStoreError as EntityStoreError
from nemo_helix_plugin.entities import EntityValidationError as EntityValidationError
from nemo_helix_plugin.entities import ListResponse as ListResponse
from nemo_helix_plugin.entities import PaginationInfo as PaginationInfo
from nemo_helix_plugin.entities import SyncEntityClient as _PluginSyncEntityClient
from nemo_helix_plugin.entities import parse_qualified_name as parse_qualified_name


def _service_principal_headers(service_name: str, *, internal: bool) -> dict[str, str]:
    from nhx.common.observability import MARK_INTERNAL_REQUEST_HEADERS

    headers: dict[str, str] = {
        "X-NHX-Principal-Id": f"service:{service_name}",
        "X-NHX-Actor-Account-Id": "",
        "X-NHX-Actor-Aliases": f"service:{service_name}",
        # ``with_options`` merges defaults. Explicitly clear any delegation
        # inherited from a request-scoped client so this is true elevation.
        "X-NHX-Principal-On-Behalf-Of": "",
        "X-NHX-Principal-On-Behalf-Of-Email": "",
        "X-NHX-Principal-On-Behalf-Of-Groups": "",
        "X-NHX-Subject-Account-Id": "",
        "X-NHX-Subject-Aliases": "",
    }
    if internal:
        headers.update(MARK_INTERNAL_REQUEST_HEADERS)
    return headers


class EntityClient(_PluginEntityClient):
    """Extended entity client with platform-specific capabilities.

    Adds ``as_service()`` for service-principal credential elevation,
    which requires ``nhx.common.observability``.
    """

    def as_service(self, service_name: str, *, internal: bool = False) -> "EntityClient":
        """Return a copy with service principal credentials baked in.

        Use this for background tasks, startup code, or permission elevation
        where you need service-level access. The returned client has service
        principal headers (X-NHX-Principal-Id: service:<name>) baked into its
        underlying SDK, so all requests are authenticated as the service.

        Args:
            service_name: The service name (e.g., "auth", "evaluator")
            internal: If True, mark requests as internal to suppress access logging.

        Returns:
            A new EntityClient backed by an SDK with service principal headers.
        """
        # with_options merges headers into the client's defaults and shares the
        # underlying httpx transport (connection pool, auth), so this is cheap.
        # It clones via copy.copy, so the platform URL resolver carries over and
        # no request-router fixup is needed the way the Stainless path required.
        service_client = self._client.with_options(headers=_service_principal_headers(service_name, internal=internal))
        return EntityClient(service_client)


class SyncEntityClient(_PluginSyncEntityClient):
    """Synchronous entity client with platform-specific capabilities."""

    def as_service(self, service_name: str, *, internal: bool = False) -> "SyncEntityClient":
        """Return a copy with service principal credentials baked in."""
        service_client = self._client.with_options(headers=_service_principal_headers(service_name, internal=internal))
        return SyncEntityClient(service_client)
