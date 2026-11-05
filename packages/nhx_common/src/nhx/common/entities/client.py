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
from nemo_helix_plugin.entities.client import AsyncEntitiesClient, EntitiesClient
from nhx.common.auth.headers import AUTHENTICATION_CONTEXT_HEADERS
from nhx.common.config import get_auth_config
from nhx.common.observability import MARK_INTERNAL_REQUEST_HEADERS


def _service_principal_headers(service_name: str, *, internal: bool) -> dict[str, str]:
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


def _without_authentication_context(existing_headers: dict[str, str], *, internal: bool) -> dict[str, str]:
    headers = {
        name: value for name, value in existing_headers.items() if name.lower() not in AUTHENTICATION_CONTEXT_HEADERS
    }
    if internal:
        headers.update(MARK_INTERNAL_REQUEST_HEADERS)
    return headers


def _uses_service_workload_token() -> bool:
    config = get_auth_config()
    return bool(config.enabled and config.oidc.workload is not None)


def _async_client_as_service(
    client: AsyncEntitiesClient,
    service_name: str,
    *,
    internal: bool,
) -> AsyncEntitiesClient:
    from nhx.common.auth.workload_tokens import ServiceWorkloadAccessTokenProvider

    config = get_auth_config()
    bearer = _uses_service_workload_token()
    headers = _without_authentication_context(client.default_headers, internal=internal)
    if not bearer:
        headers.update(_service_principal_headers(service_name, internal=internal))
    return client.__class__(
        base_url=client.base_url,
        workspace=client.workspace,
        auth=ServiceWorkloadAccessTokenProvider(config, service_name) if bearer else None,
        default_headers=headers or None,
        timeout=client._timeout,
        retry=client._retry,
        http_client=client._http,
        owns_http_client=False,
        client_runtime=client.nemo_client_runtime,
    )


def _sync_client_as_service(
    client: EntitiesClient,
    service_name: str,
    *,
    internal: bool,
) -> EntitiesClient:
    from nhx.common.auth.workload_tokens import ServiceWorkloadAccessTokenProvider

    config = get_auth_config()
    bearer = _uses_service_workload_token()
    headers = _without_authentication_context(client.default_headers, internal=internal)
    if not bearer:
        headers.update(_service_principal_headers(service_name, internal=internal))
    return client.__class__(
        base_url=client.base_url,
        workspace=client.workspace,
        auth=ServiceWorkloadAccessTokenProvider(config, service_name) if bearer else None,
        default_headers=headers or None,
        timeout=client._timeout,
        retry=client._retry,
        http_client=client._http,
        owns_http_client=False,
        client_runtime=client.nemo_client_runtime,
    )


class EntityClient(_PluginEntityClient):
    """Extended entity client with platform-specific capabilities.

    Adds ``as_service()`` for service-principal credential elevation,
    which requires ``nhx.common.observability``.
    """

    def as_service(self, service_name: str, *, internal: bool = False) -> "EntityClient":
        """Return a copy with service principal credentials baked in.

        Use this for background tasks, startup code, or permission elevation
        where you need service-level access. In trusted-header mode the
        returned client has service principal headers baked in; in workload
        token-exchange mode it attaches a service bearer token per request.

        Args:
            service_name: The service name (e.g., "auth", "evaluator")
            internal: If True, mark requests as internal to suppress access logging.

        Returns:
            A new EntityClient backed by an SDK with service principal headers.
        """
        service_client = _async_client_as_service(self._client, service_name, internal=internal)
        return EntityClient(service_client)


class SyncEntityClient(_PluginSyncEntityClient):
    """Synchronous entity client with platform-specific capabilities."""

    def as_service(self, service_name: str, *, internal: bool = False) -> "SyncEntityClient":
        """Return a copy with service principal credentials baked in."""
        service_client = _sync_client_as_service(self._client, service_name, internal=internal)
        return SyncEntityClient(service_client)
