# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rich NemoClient factory backed by platform internals.

Builds :class:`~nemo_helix_plugin.client.client.NemoClient` /
:class:`~nemo_helix_plugin.client.client.AsyncNemoClient` handles for services,
controllers, and tasks running inside the platform, using:

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
from typing import TypeVar

import httpx
from nemo_helix_plugin.client.client import AsyncNemoClient, BaseNemoClient, NemoClient, NemoClientRuntime
from nemo_helix_plugin.client.constants import is_workload_identity_token_file_set
from nemo_helix_plugin.client.types import RetryPolicy
from nhx.common.auth.models import Principal
from nhx.common.auth.tasks import principal_from_env
from nhx.common.client_runtime import build_platform_client_runtime
from nhx.common.platform_client_context import (
    DELEGATED_PRINCIPAL_HEADERS,
    PRINCIPAL_OBO_HEADER,
    HelixClientContext,
    build_platform_client_context,
    delegated_principal_headers,
)
from nhx.common.platform_endpoint import HelixEndpoint, resolve_platform_endpoint

logger = logging.getLogger(__name__)

#: Default retry behavior for platform clients, matching the generated SDK's
#: contract: up to 2 retries with exponential backoff on request timeouts
#: (408), lock conflicts (409), rate limits (429) and server errors (>=500),
#: honoring the server's ``Retry-After`` and ``x-should-retry`` verdicts.
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    retryable_status_codes=(408, 409, 429),
    retry_all_server_errors=True,
    respect_retry_after_headers=True,
    respect_retry_decision_headers=True,
)

ClientT = TypeVar("ClientT", bound=BaseNemoClient)


def _unrouted_endpoint(context: HelixClientContext) -> HelixEndpoint:
    endpoint = context.endpoint
    return HelixEndpoint(
        connect_base_url=endpoint.connect_base_url,
        socket_path=endpoint.socket_path,
        transport=endpoint.transport,
    )


def _nemo_client_runtime(context: HelixClientContext, base_url: str | None) -> NemoClientRuntime:
    if base_url is None:
        return context.runtime
    return build_platform_client_runtime(_unrouted_endpoint(context))


def _sync_nemo_http_client(
    context: HelixClientContext,
    http_client: httpx.Client | None,
    base_url: str | None,
    *,
    limits: httpx.Limits | None,
    follow_redirects: bool | None,
) -> httpx.Client:
    if http_client is not None:
        return http_client
    endpoint = context.endpoint if base_url is None else _unrouted_endpoint(context)
    return endpoint.sync_sdk_http_client(limits=limits, follow_redirects=follow_redirects)


def _async_nemo_http_client(
    context: HelixClientContext,
    http_client: httpx.AsyncClient | None,
    base_url: str | None,
    *,
    limits: httpx.Limits | None,
    follow_redirects: bool | None,
) -> httpx.AsyncClient:
    if http_client is not None:
        return http_client
    endpoint = context.endpoint if base_url is None else _unrouted_endpoint(context)
    return endpoint.async_sdk_http_client(limits=limits, follow_redirects=follow_redirects)


def get_nemo_client(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | Principal | None = None,
    workspace: str | None = None,
    http_client: httpx.Client | None = None,
    base_url: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    limits: httpx.Limits | None = None,
    follow_redirects: bool | None = None,
    retry: RetryPolicy | None = DEFAULT_RETRY_POLICY,
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
        base_url: Optional platform base URL; defaults to the configured endpoint.
            Requests go to it directly instead of through service endpoint routing.
        timeout: Per-request timeout applied by the client (also honoured when
            ``http_client`` is shared).
        limits: Connection-pool limits for the HTTP client built by this factory.
            Ignored when ``http_client`` is supplied.
        follow_redirects: Redirect policy for the HTTP client built by this
            factory. Ignored when ``http_client`` is supplied.
        retry: Client-level retry policy; defaults to :data:`DEFAULT_RETRY_POLICY`.
            ``None`` disables retries.

    Note:
        OTEL trace-propagation headers are captured once, at construction, from
        the current request context.  Build a fresh client per request scope
        rather than caching one across requests, or its ``traceparent`` will be
        stale.
    """
    context = build_platform_client_context(
        base_url=base_url,
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        has_explicit_http_client=http_client is not None,
        include_otel_headers=True,
    )
    resolved_base_url = context.base_url
    return NemoClient(
        base_url=resolved_base_url,
        workspace=workspace,
        auth=context.nemo_client_auth(),
        default_headers=context.default_headers_or_none(),
        timeout=timeout,
        retry=retry,
        http_client=_sync_nemo_http_client(
            context, http_client, base_url, limits=limits, follow_redirects=follow_redirects
        ),
        client_runtime=_nemo_client_runtime(context, base_url),
    )


def get_async_nemo_client(
    *,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | Principal | None = None,
    workspace: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    limits: httpx.Limits | None = None,
    follow_redirects: bool | None = None,
    retry: RetryPolicy | None = DEFAULT_RETRY_POLICY,
) -> AsyncNemoClient:
    """Async counterpart of :func:`get_nemo_client`.

    Uses the explicitly provided ``http_client`` (e.g. from a test fixture), or
    creates one from the resolved platform endpoint.
    """
    context = build_platform_client_context(
        base_url=base_url,
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        has_explicit_http_client=http_client is not None,
        include_otel_headers=True,
    )
    resolved_base_url = context.base_url
    return AsyncNemoClient(
        base_url=resolved_base_url,
        workspace=workspace,
        auth=context.nemo_client_auth(),
        default_headers=context.default_headers_or_none(),
        timeout=timeout,
        retry=retry,
        http_client=_async_nemo_http_client(
            context, http_client, base_url, limits=limits, follow_redirects=follow_redirects
        ),
        client_runtime=_nemo_client_runtime(context, base_url),
    )


def get_nemo_client_on_behalf_of(client: ClientT, on_behalf_of: str | Principal) -> ClientT:
    """Return a copy of *client* that acts on behalf of *on_behalf_of*.

    Replaces any ``X-NHX-Principal-On-Behalf-Of*`` / ``X-NHX-Subject-*`` default
    headers on *client* with the delegated identity and keeps every other
    header. The copy shares the transport of *client* and keeps its type, so a
    typed service client (e.g. ``AsyncSecretsClient``) stays typed.

    Delegation is expressed with trusted headers. A client that authenticates
    with a service workload token carries its delegated identity in the token;
    build that one with ``get_async_nemo_client(as_service=..., on_behalf_of=...)``
    instead.
    """
    return client.without_headers((PRINCIPAL_OBO_HEADER, *DELEGATED_PRINCIPAL_HEADERS)).with_headers(
        delegated_principal_headers(on_behalf_of)
    )


def get_task_nemo_client(
    service_name: str,
    *,
    workspace: str | None = None,
    http_client: httpx.Client | None = None,
) -> NemoClient:
    """Build a sync :class:`NemoClient` for use inside a task container.

    Reads the job creator's principal from ``NHX_PRINCIPAL`` and authenticates
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
