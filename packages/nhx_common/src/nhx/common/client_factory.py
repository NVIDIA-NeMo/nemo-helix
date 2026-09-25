# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rich NemoClient factory backed by platform internals.

This is the :class:`~nemo_helix_plugin.client.client.NemoClient` sibling of
:mod:`nhx.common.sdk_factory`.  It builds typed clients that reuse the same
platform machinery the SDK factory uses:

- base URL from :class:`~nhx.common.config.Configuration`;
- per-service URL routing via :class:`~nhx.common.platform_endpoint.HelixEndpoint`;
- endpoint-aware sync/async HTTP clients;
- principal / auth + runtime context via :mod:`nhx.common.platform_client_context`;
- OTEL trace-propagation headers captured on the current request.

:class:`HelixNemoClientProvider` is registered under the ``nemo.client_provider``
entry-point group so :func:`nemo_helix_plugin.client_provider.get_nemo_client`
discovers it automatically whenever ``nhx-common`` is installed.
"""

from __future__ import annotations

import logging

import httpx
from nemo_helix_plugin.client.client import AsyncNemoClient, NemoClient
from nemo_helix_plugin.client.constants import is_workload_identity_token_file_set
from nhx.common.auth.models import Principal
from nhx.common.auth.tasks import principal_from_env
from nhx.common.platform_client_context import build_platform_client_context
from nhx.common.platform_endpoint import resolve_platform_endpoint

logger = logging.getLogger(__name__)


def get_nemo_client(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | Principal | None = None,
    workspace: str | None = None,
    http_client: httpx.Client | None = None,
) -> NemoClient:
    """Build a sync :class:`NemoClient` configured with platform internals.

    Args:
        as_service: If provided, authenticate as ``service:{as_service}``.
            If ``None``, propagate the current request's / env principal.
        internal: Mark requests as internal (service-to-service).
        on_behalf_of: Principal (or id) to act on behalf of.  Passing a
            :class:`~nhx.common.auth.Principal` (rather than a bare id string)
            is only reachable through this direct entry point; the plugin-facing
            :class:`~nemo_helix_plugin.client_provider.NemoClientProvider`
            protocol narrows ``on_behalf_of`` to ``str | None``.
        workspace: Default workspace used to fill ``{workspace}`` path params.
        http_client: Optional sync HTTP client; defaults to an endpoint-aware client.

    Note:
        OTEL trace-propagation headers are captured once, at construction, from
        the current request context.  Build a fresh client per request scope
        rather than caching one across requests, or its ``traceparent`` will be
        stale (mirrors ``get_platform_sdk``).
    """
    context = build_platform_client_context(
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        has_explicit_http_client=http_client is not None,
        include_otel_headers=True,
    )
    base_url = context.base_url
    return NemoClient(
        base_url=base_url,
        workspace=workspace,
        auth=context.nemo_client_auth(),
        default_headers=context.default_headers_or_none(),
        http_client=context.runtime_context.sync_nemo_http_client(http_client=http_client),
        client_runtime=context.runtime,
    )


def get_async_nemo_client(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | Principal | None = None,
    workspace: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> AsyncNemoClient:
    """Async counterpart of :func:`get_nemo_client`.

    Uses the explicitly provided ``http_client`` (e.g. from a test fixture), or
    creates one from the resolved platform endpoint.
    """
    context = build_platform_client_context(
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        has_explicit_http_client=http_client is not None,
        include_otel_headers=True,
    )
    base_url = context.base_url
    return AsyncNemoClient(
        base_url=base_url,
        workspace=workspace,
        auth=context.nemo_client_auth(),
        default_headers=context.default_headers_or_none(),
        http_client=context.runtime_context.async_nemo_http_client(http_client=http_client),
        client_runtime=context.runtime,
    )


def get_task_nemo_client(
    service_name: str,
    *,
    workspace: str | None = None,
    http_client: httpx.Client | None = None,
) -> NemoClient:
    """Build a sync :class:`NemoClient` for use inside a task container.

    NemoClient counterpart of :func:`nhx.common.sdk_factory.get_task_sdk`:
    reads the job creator's principal from ``NHX_PRINCIPAL`` and authenticates
    as ``service:{service_name}`` while acting on behalf of that creator, or --
    when ``NHX_WORKLOAD_IDENTITY_TOKEN_FILE`` is set -- bootstraps
    workload-identity bearer-token exchange (via :func:`get_nemo_client` with
    ``internal=True``) instead of trusted ``X-NHX-*`` principal headers.
    """
    if http_client is None and is_workload_identity_token_file_set():
        return get_nemo_client(internal=True, workspace=workspace)
    if http_client is None:
        http_client = resolve_platform_endpoint().sync_sdk_http_client()
    principal = principal_from_env()
    if principal is None:
        logger.warning(
            "NHX_PRINCIPAL not set; task NemoClient will authenticate as service:%s without on-behalf-of delegation",
            service_name,
        )
    return get_nemo_client(
        as_service=service_name,
        internal=True,
        on_behalf_of=principal.effective_principal if principal else None,
        workspace=workspace,
        http_client=http_client,
    )


def get_async_task_nemo_client(
    service_name: str,
    *,
    workspace: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> AsyncNemoClient:
    """Async counterpart of :func:`get_task_nemo_client`. Wire-identical."""
    if http_client is None and is_workload_identity_token_file_set():
        return get_async_nemo_client(internal=True, workspace=workspace)
    if http_client is None:
        http_client = resolve_platform_endpoint().async_sdk_http_client()
    principal = principal_from_env()
    if principal is None:
        logger.warning(
            "NHX_PRINCIPAL not set; async task NemoClient will authenticate as service:%s without on-behalf-of delegation",
            service_name,
        )
    return get_async_nemo_client(
        as_service=service_name,
        internal=True,
        on_behalf_of=principal.effective_principal if principal else None,
        workspace=workspace,
        http_client=http_client,
    )


# ---------------------------------------------------------------------------
# Entry-point provider for nemo_helix_plugin.client_provider
# ---------------------------------------------------------------------------


class HelixNemoClientProvider:
    """Rich :class:`~nemo_helix_plugin.client_provider.NemoClientProvider`
    that uses platform internals (endpoint-aware HTTP clients, URL routing, OTEL
    headers, auth context).

    Registered as a ``nemo.client_provider`` entry-point so it is discovered
    automatically when ``nhx-common`` is installed.
    """

    def get_nemo_client(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | Principal | None = None,
        workspace: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> NemoClient:
        return get_nemo_client(
            as_service=as_service,
            internal=internal,
            on_behalf_of=on_behalf_of,
            workspace=workspace,
            http_client=http_client,
        )

    def get_async_nemo_client(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | Principal | None = None,
        workspace: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> AsyncNemoClient:
        return get_async_nemo_client(
            as_service=as_service,
            internal=internal,
            on_behalf_of=on_behalf_of,
            workspace=workspace,
            http_client=http_client,
        )

    def get_task_nemo_client(
        self,
        service_name: str,
        *,
        workspace: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> NemoClient:
        return get_task_nemo_client(service_name, workspace=workspace, http_client=http_client)

    def get_async_task_nemo_client(
        self,
        service_name: str,
        *,
        workspace: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> AsyncNemoClient:
        return get_async_task_nemo_client(service_name, workspace=workspace, http_client=http_client)
