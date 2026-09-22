# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK factory functions for creating NeMo Helix SDK instances."""

import logging
from collections.abc import Callable, Mapping

import httpx
from httpx._types import TimeoutTypes
from nemo_helix import AsyncNeMoHelix, NeMoHelix, Omit
from nemo_helix_plugin.client.constants import is_workload_identity_token_file_set
from nhx.common import platform_client_context
from nhx.common.auth import Principal, principal_from_env
from nhx.common.config import Configuration, HelixConfig
from nhx.common.platform_client_context import (
    DELEGATED_PRINCIPAL_HEADERS,
    PRINCIPAL_OBO_HEADER,
    AsyncRequestHook,
    HelixClientContext,
    SyncRequestHook,
)
from nhx.common.platform_endpoint import HelixEndpoint

logger = logging.getLogger(__name__)


def _get_platform_config() -> HelixConfig:
    platform_config = Configuration.get_platform_config()
    if not isinstance(platform_config, HelixConfig):
        raise TypeError("Expected HelixConfig from Configuration.get_platform_config()")
    return platform_config


def _sync_workload_identity_http_client_factory(
    endpoint: HelixEndpoint,
    *,
    timeout: TimeoutTypes | None,
    limits: httpx.Limits | None,
    follow_redirects: bool | None,
) -> Callable[[SyncRequestHook, str | bool], httpx.Client]:
    def create_http_client(request_hook: SyncRequestHook, verify: str | bool) -> httpx.Client:
        return endpoint.sync_sdk_http_client(
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
            request_hooks=(request_hook,),
            verify=verify,
        )

    return create_http_client


def _async_workload_identity_http_client_factory(
    endpoint: HelixEndpoint,
    *,
    timeout: TimeoutTypes | None,
    limits: httpx.Limits | None,
    follow_redirects: bool | None,
) -> Callable[[AsyncRequestHook, str | bool], httpx.AsyncClient]:
    def create_http_client(request_hook: AsyncRequestHook, verify: str | bool) -> httpx.AsyncClient:
        return endpoint.async_sdk_http_client(
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
            request_hooks=(request_hook,),
            verify=verify,
        )

    return create_http_client


def _workload_identity_platform_sdk(
    *,
    context: HelixClientContext,
    base_url: str | None,
    headers: Mapping[str, str],
    timeout: TimeoutTypes | None,
    limits: httpx.Limits | None,
    follow_redirects: bool | None,
) -> NeMoHelix:
    from nemo_helix_ext.client.factory import build_client_init_kwargs

    client_init_kwargs = build_client_init_kwargs(
        base_url=base_url,
        extra_headers=headers or None,
        http_client_factory=_sync_workload_identity_http_client_factory(
            context.endpoint,
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
        ),
    )
    http_client = client_init_kwargs.http_client
    if http_client is not None and not isinstance(http_client, httpx.Client):
        raise TypeError("Expected httpx.Client from sync client factory")
    if http_client is None:
        http_client = context.endpoint.sync_sdk_http_client(
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
            verify=client_init_kwargs.client_verify,
        )
    return NeMoHelix(
        workspace=client_init_kwargs.workspace,
        base_url=client_init_kwargs.base_url,
        default_headers=client_init_kwargs.default_headers,
        http_client=http_client,
        nemo_client_runtime=context.runtime,
    )


def _async_workload_identity_platform_sdk(
    *,
    context: HelixClientContext,
    base_url: str | None,
    headers: Mapping[str, str],
    timeout: TimeoutTypes | None,
    limits: httpx.Limits | None,
    follow_redirects: bool | None,
) -> AsyncNeMoHelix:
    from nemo_helix_ext.client.factory import build_async_client_init_kwargs

    client_init_kwargs = build_async_client_init_kwargs(
        base_url=base_url,
        extra_headers=headers or None,
        http_client_factory=_async_workload_identity_http_client_factory(
            context.endpoint,
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
        ),
    )
    http_client = client_init_kwargs.http_client
    if http_client is not None and not isinstance(http_client, httpx.AsyncClient):
        raise TypeError("Expected httpx.AsyncClient from async client factory")
    if http_client is None:
        http_client = context.endpoint.async_sdk_http_client(
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
            verify=client_init_kwargs.client_verify,
        )
    return AsyncNeMoHelix(
        workspace=client_init_kwargs.workspace,
        base_url=client_init_kwargs.base_url,
        default_headers=client_init_kwargs.default_headers,
        http_client=http_client,
        nemo_client_runtime=context.runtime,
    )


def get_platform_sdk(
    as_service: str | None = None,
    internal: bool = False,
    http_client: httpx.Client | None = None,
    on_behalf_of: str | Principal | None = None,
    base_url: str | None = None,
    timeout: TimeoutTypes | None = None,
    limits: httpx.Limits | None = None,
    follow_redirects: bool | None = None,
) -> NeMoHelix:
    """
    Returns an instance of the NeMoHelix SDK configured with the platform's base URL.

    Args:
        as_service: If provided, use service principal headers (service:{name}).
                   Use this for internal service operations without user context
                   (e.g., startup, background tasks, controllers).
                   If None and auth is enabled, propagates the current user's auth context.
        internal: If True, mark all requests from this SDK as internal requests.
                 Use this for controllers and background tasks that make internal API calls.
        http_client: Optional sync HTTP client to use for requests.
        on_behalf_of: Optional principal ID to use for on-behalf-of authorization.
        base_url: Optional platform base URL. Defaults to configured platform base URL.

    Returns:
        Configured NeMoHelix SDK instance.
    """
    context = platform_client_context.build_platform_client_context(
        platform_config=_get_platform_config(),
        base_url=base_url,
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        has_explicit_http_client=http_client is not None,
    )
    resolved_base_url = context.base_url
    if context.auth_plan.uses_workload_identity_bootstrap:
        return _workload_identity_platform_sdk(
            context=context,
            base_url=resolved_base_url,
            headers=context.default_headers,
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
        )
    sdk = NeMoHelix(
        base_url=resolved_base_url,
        http_client=context.sync_sdk_http_client(
            http_client,
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
        ),
        default_headers=context.default_headers_or_none(),
        nemo_client_runtime=context.runtime,
    )
    return sdk.set_nemo_client_auth(context.nemo_client_auth())


def get_task_sdk(as_service: str, http_client: httpx.Client | None = None) -> NeMoHelix:
    """Create an SDK for use inside a task container with on-behalf-of auth.

    Reads the job creator's principal from the NHX_PRINCIPAL environment variable
    (set by the jobs backend when launching task containers) and creates an SDK
    that authenticates as the given service while acting on behalf of the job creator.

    Args:
        as_service: Service name for the service principal (e.g., "customizer").
        http_client: Optional sync HTTP client to use for requests.

    Returns:
        Configured NeMoHelix SDK with internal + on-behalf-of headers.
    """
    if http_client is None and is_workload_identity_token_file_set():
        return get_platform_sdk(internal=True)

    principal = principal_from_env()
    if principal is None:
        logger.warning(
            "NHX_PRINCIPAL not set; task SDK will authenticate as service:%s without on-behalf-of delegation",
            as_service,
        )
    return get_platform_sdk(
        as_service=as_service,
        internal=True,
        http_client=http_client,
        on_behalf_of=principal.effective_principal if principal else None,
    )


def get_async_task_sdk(as_service: str, http_client: httpx.AsyncClient | None = None) -> AsyncNeMoHelix:
    """Async counterpart of :func:`get_task_sdk` for use inside a task container.

    Reads the job creator's principal from ``NHX_PRINCIPAL`` and creates an async SDK that
    authenticates as the given service while acting on behalf of the job creator with the full
    delegated identity (on-behalf-of id, email, and groups). Wire-identical to :func:`get_task_sdk`.

    Args:
        as_service: Service name for the service principal (e.g., "evaluator").
        http_client: Optional async HTTP client to use for requests.

    Returns:
        Configured AsyncNeMoHelix SDK with internal + on-behalf-of headers.
    """
    if http_client is None and is_workload_identity_token_file_set():
        return get_async_platform_sdk(internal=True)

    principal = principal_from_env()
    if principal is None:
        logger.warning(
            "NHX_PRINCIPAL not set; async task SDK will authenticate as service:%s without on-behalf-of delegation",
            as_service,
        )
    return get_async_platform_sdk(
        as_service=as_service,
        internal=True,
        http_client=http_client,
        on_behalf_of=principal.effective_principal if principal else None,
    )


def get_async_platform_sdk(
    as_service: str | None = None,
    internal: bool = False,
    http_client: httpx.AsyncClient | None = None,
    on_behalf_of: str | Principal | None = None,
    base_url: str | None = None,
    timeout: TimeoutTypes | None = None,
    limits: httpx.Limits | None = None,
    follow_redirects: bool | None = None,
) -> AsyncNeMoHelix:
    """
    Returns an instance of the AsyncNeMoHelix SDK configured with the platform's base URL.

    Args:
        as_service: If provided, use service principal headers (service:{name}).
                   Use this for internal service operations without user context
                   (e.g., startup, background tasks, controllers).
                   If None and auth is enabled, propagates the current user's auth context.
        internal: If True, mark all requests from this SDK as internal requests.
                 Use this for controllers and background tasks that make internal API calls.
        http_client: Optional HTTP client to use for requests. Used for test injection
                    via DependencyProvider. See architecture/docs/http-client-injection.md.
        on_behalf_of: Optional principal ID to use for on-behalf-of authorization.
        base_url: Optional platform base URL. Defaults to configured platform base URL.
    Returns:
        Configured AsyncNeMoHelix SDK instance.
    """
    context = platform_client_context.build_platform_client_context(
        platform_config=_get_platform_config(),
        base_url=base_url,
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        has_explicit_http_client=http_client is not None,
    )
    resolved_base_url = context.base_url
    if context.auth_plan.uses_workload_identity_bootstrap:
        return _async_workload_identity_platform_sdk(
            context=context,
            base_url=resolved_base_url,
            headers=context.default_headers,
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
        )
    sdk = AsyncNeMoHelix(
        base_url=resolved_base_url,
        http_client=context.async_sdk_http_client(
            http_client,
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
        ),
        default_headers=context.default_headers_or_none(),
        nemo_client_runtime=context.runtime,
    )
    return sdk.set_nemo_client_auth(context.nemo_client_auth())


def get_request_scoped_sdk(
    base_sdk: AsyncNeMoHelix,
) -> AsyncNeMoHelix:
    """Create a request-scoped SDK with current auth and observability headers.

    Takes a base SDK (with shared HTTP client) and returns a new SDK instance
    with the current request's auth headers applied via .with_options().

    This is lightweight - the underlying HTTP client is reused.

    Args:
        base_sdk: The base SDK instance (typically cached by DependencyProvider)

    Returns:
        SDK instance with auth + OTEL headers, or base_sdk if no headers to add.

    Usage:
        This is called by DependencyProvider to create per-request SDK instances
        for FastAPI dependency injection.
    """

    # Combine OTEL headers (tracing) + auth headers (user identity)
    headers = platform_client_context.forwardable_otel_headers(include_internal=True)
    headers.update(platform_client_context.current_principal_auth_headers())

    # If we have headers to add, create a new SDK with them
    # This reuses the underlying HTTP client (lightweight operation)
    if headers:
        return base_sdk.with_options(default_headers=headers)

    return base_sdk


def get_request_scoped_sync_sdk(
    base_sdk: NeMoHelix,
) -> NeMoHelix:
    """Create a request-scoped sync SDK with current auth and observability headers."""

    headers = platform_client_context.forwardable_otel_headers(include_internal=True)
    headers.update(platform_client_context.current_principal_auth_headers())

    if headers:
        return base_sdk.with_options(default_headers=headers)

    return base_sdk


def get_sdk_on_behalf_of(
    base_sdk: NeMoHelix | AsyncNeMoHelix,
    on_behalf_of: str | Principal,
) -> NeMoHelix | AsyncNeMoHelix:
    """Create an SDK with on-behalf-of headers for delegated access.

    Takes an existing SDK (typically created as a service principal) and returns
    a new SDK instance with X-NHX-Principal-On-Behalf-Of header added. This enables
    service principals to act on behalf of users while checking the delegated user's
    permissions.

    This is lightweight - the underlying HTTP client is reused, and all original
    headers are preserved.

    Args:
        base_sdk: The base SDK instance (typically created with as_service)
        on_behalf_of: The principal ID to act on behalf of (e.g., user email)

    Returns:
        SDK instance with on-behalf-of header added and all original headers preserved.

    Usage:
        ```python
        # Create a service SDK
        service_sdk = get_platform_sdk(as_service="my-service")

        # Create delegated SDK for accessing resources on behalf of a user
        delegated_sdk = get_sdk_on_behalf_of(service_sdk, "user@example.com")

        # Create delegated SDK for accessing resources on behalf of a principal
        delegated_sdk = get_sdk_on_behalf_of(service_sdk, Principal(id="user@example.com", groups=["group1", "group2"], email="user@example.com"))

        # Secret access will check user@example.com's permissions
        secret = delegated_sdk.secrets.access("my-secret", workspace="workspace-name")
        ```
    """
    # Merge existing headers with the new on-behalf-of header
    merged_headers: dict[str, str | Omit] = dict(base_sdk._custom_headers)
    for header in (PRINCIPAL_OBO_HEADER, *DELEGATED_PRINCIPAL_HEADERS):
        merged_headers.pop(header, None)
    merged_headers.update(platform_client_context.delegated_principal_headers(on_behalf_of))
    return base_sdk.with_options(set_default_headers=merged_headers)


def get_entity_parts(name: str, default_workspace: str | None = None) -> tuple[str, str]:
    """Get the workspace and name parts of an entity reference."""
    if "/" in name:
        parts = name.split("/", 1)
        return parts[0], parts[1]
    if default_workspace is None:
        raise ValueError(
            f"Entity reference '{name}' is not qualified with a workspace, and no workspace to default to was provided. Must be in the format $workspace/$entity_name or a default workspace must be provided to fall back to."
        )
    return default_workspace, name


# ---------------------------------------------------------------------------
# Entry-point provider for nemo_helix_plugin.sdk_provider
# ---------------------------------------------------------------------------


class HelixSDKProvider:
    """Rich :class:`~nemo_helix_plugin.sdk_provider.SDKProvider` that uses
    platform internals (shared HTTP clients, URL routing, OTEL headers, auth
    context vars).

    Registered as a ``nemo.sdk_provider`` entry-point so it is
    discovered automatically when ``nhx-common`` is installed.
    """

    def get_task_sdk(self, service_name: str, http_client: httpx.Client | None = None) -> NeMoHelix:
        return get_task_sdk(service_name, http_client=http_client)

    def get_async_task_sdk(self, service_name: str, http_client: httpx.AsyncClient | None = None) -> AsyncNeMoHelix:
        return get_async_task_sdk(service_name, http_client=http_client)

    def get_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        http_client: httpx.Client | None = None,
        on_behalf_of: str | Principal | None = None,
        base_url: str | None = None,
    ) -> NeMoHelix:
        return get_platform_sdk(
            as_service=as_service,
            internal=internal,
            http_client=http_client,
            on_behalf_of=on_behalf_of,
            base_url=base_url,
        )

    def get_async_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        on_behalf_of: str | Principal | None = None,
        base_url: str | None = None,
    ) -> AsyncNeMoHelix:
        return get_async_platform_sdk(
            as_service=as_service,
            internal=internal,
            on_behalf_of=on_behalf_of,
            base_url=base_url,
        )
