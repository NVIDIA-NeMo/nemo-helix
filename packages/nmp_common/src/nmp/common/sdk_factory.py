# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK factory functions for creating NeMo Platform SDK instances."""

import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Optional, override

import httpx
from nemo_platform import DEFAULT_MAX_RETRIES, AsyncNeMoPlatform, NeMoPlatform, NotGiven, Omit, Timeout, not_given
from nemo_platform_plugin.client.constants import is_workload_identity_token_file_set
from nmp.common.auth import Principal, get_principal_auth_headers, principal_from_env
from nmp.common.config import Configuration, PlatformConfig
from nmp.common.immutable_http_client import ImmutableDefaultAsyncHttpxClient, ImmutableDefaultHttpxClient
from nmp.common.observability import MARK_INTERNAL_REQUEST_HEADERS
from nmp.common.observability.otel import get_otel_headers
from nmp.common.platform_endpoint import PlatformEndpoint, resolve_platform_endpoint

logger = logging.getLogger(__name__)
_SyncRequestHook = Callable[[httpx.Request], None]
_AsyncRequestHook = Callable[[httpx.Request], Awaitable[None]]


def _get_platform_config() -> PlatformConfig:
    platform_config = Configuration.get_platform_config()
    if not isinstance(platform_config, PlatformConfig):
        raise TypeError("Expected PlatformConfig from Configuration.get_platform_config()")
    return platform_config


def _sync_sdk_http_client(
    endpoint: PlatformEndpoint,
    base_url: str | None,
    http_client: httpx.Client | None,
) -> httpx.Client:
    if http_client is not None:
        return endpoint.sync_sdk_http_client(http_client=http_client)
    if base_url is not None and not endpoint.service_endpoints:
        return ImmutableDefaultHttpxClient()
    return endpoint.sync_sdk_http_client()


def _async_sdk_http_client(
    endpoint: PlatformEndpoint,
    base_url: str | None,
    http_client: httpx.AsyncClient | None,
) -> httpx.AsyncClient:
    if http_client is not None:
        return endpoint.async_sdk_http_client(http_client=http_client)
    if base_url is not None and not endpoint.service_endpoints:
        return ImmutableDefaultAsyncHttpxClient()
    return endpoint.async_sdk_http_client()


def _sync_workload_identity_http_client_factory(
    endpoint: PlatformEndpoint,
) -> Callable[[_SyncRequestHook, str | bool], httpx.Client]:
    def create_http_client(request_hook: _SyncRequestHook, verify: str | bool) -> httpx.Client:
        return endpoint.sync_sdk_http_client(request_hooks=(request_hook,), verify=verify)

    return create_http_client


def _async_workload_identity_http_client_factory(
    endpoint: PlatformEndpoint,
) -> Callable[[_AsyncRequestHook, str | bool], httpx.AsyncClient]:
    def create_http_client(request_hook: _AsyncRequestHook, verify: str | bool) -> httpx.AsyncClient:
        return endpoint.async_sdk_http_client(request_hooks=(request_hook,), verify=verify)

    return create_http_client


class _WorkloadIdentityRoutedNeMoPlatform(NeMoPlatform):
    _platform_endpoint: PlatformEndpoint

    def __init__(
        self,
        *,
        workspace: str | None = None,
        base_url: str | httpx.URL | None = None,
        inference_base_url: str | httpx.URL | None = None,
        config_path: Path | None = None,
        context_name: str | None = None,
        access_token: str | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str | Omit] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.Client | None = None,
        _strict_response_validation: bool = False,
        platform_config: PlatformConfig | None = None,
        platform_endpoint: PlatformEndpoint | None = None,
    ) -> None:
        resolved_config = platform_config or _get_platform_config()
        self._platform_endpoint = platform_endpoint or resolve_platform_endpoint(resolved_config)
        if http_client is None:
            from nemo_platform_ext.client.factory import build_client_init_kwargs

            client_init_kwargs = build_client_init_kwargs(
                config_path=config_path,
                base_url=base_url,
                context_name=context_name,
                access_token=access_token,
                extra_headers=default_headers,
                http_client_factory=_sync_workload_identity_http_client_factory(self._platform_endpoint),
            )
            base_url = client_init_kwargs.base_url
            if workspace is None:
                workspace = client_init_kwargs.workspace
            default_headers = client_init_kwargs.default_headers
            if client_init_kwargs.http_client is not None and not isinstance(
                client_init_kwargs.http_client, httpx.Client
            ):
                raise TypeError("Expected httpx.Client from sync client factory")
            if client_init_kwargs.http_client is not None:
                http_client = client_init_kwargs.http_client
            else:
                http_client = self._platform_endpoint.sync_sdk_http_client(verify=client_init_kwargs.client_verify)
        super().__init__(
            workspace=workspace,
            base_url=base_url,
            inference_base_url=inference_base_url,
            config_path=config_path,
            context_name=context_name,
            access_token=access_token,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

    @override
    def _prepare_url(self, url: str) -> httpx.URL:
        return self._platform_endpoint.route_request_url(super()._prepare_url(url)).url


class _WorkloadIdentityRoutedAsyncNeMoPlatform(AsyncNeMoPlatform):
    _platform_endpoint: PlatformEndpoint

    def __init__(
        self,
        *,
        workspace: str | None = None,
        base_url: str | httpx.URL | None = None,
        inference_base_url: str | httpx.URL | None = None,
        config_path: Path | None = None,
        context_name: str | None = None,
        access_token: str | None = None,
        timeout: float | Timeout | None | NotGiven = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str | Omit] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.AsyncClient | None = None,
        _strict_response_validation: bool = False,
        platform_config: PlatformConfig | None = None,
        platform_endpoint: PlatformEndpoint | None = None,
    ) -> None:
        resolved_config = platform_config or _get_platform_config()
        self._platform_endpoint = platform_endpoint or resolve_platform_endpoint(resolved_config)
        if http_client is None:
            from nemo_platform_ext.client.factory import build_async_client_init_kwargs

            client_init_kwargs = build_async_client_init_kwargs(
                config_path=config_path,
                base_url=base_url,
                context_name=context_name,
                access_token=access_token,
                extra_headers=default_headers,
                http_client_factory=_async_workload_identity_http_client_factory(self._platform_endpoint),
            )
            base_url = client_init_kwargs.base_url
            if workspace is None:
                workspace = client_init_kwargs.workspace
            default_headers = client_init_kwargs.default_headers
            if client_init_kwargs.http_client is not None and not isinstance(
                client_init_kwargs.http_client,
                httpx.AsyncClient,
            ):
                raise TypeError("Expected httpx.AsyncClient from async client factory")
            if client_init_kwargs.http_client is not None:
                http_client = client_init_kwargs.http_client
            else:
                http_client = self._platform_endpoint.async_sdk_http_client(verify=client_init_kwargs.client_verify)
        super().__init__(
            workspace=workspace,
            base_url=base_url,
            inference_base_url=inference_base_url,
            config_path=config_path,
            context_name=context_name,
            access_token=access_token,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

    @override
    def _prepare_url(self, url: str) -> httpx.URL:
        return self._platform_endpoint.route_request_url(super()._prepare_url(url)).url


def _should_bootstrap_workload_identity(
    *,
    as_service: str | None,
    on_behalf_of: str | Principal | None,
    http_client: httpx.Client | httpx.AsyncClient | None,
    endpoint: PlatformEndpoint,
) -> bool:
    return (
        as_service is None
        and on_behalf_of is None
        and http_client is None
        and endpoint.transport != "uds"
        and is_workload_identity_token_file_set()
    )


def _workload_identity_extra_headers(*, internal: bool) -> dict[str, str]:
    return MARK_INTERNAL_REQUEST_HEADERS.copy() if internal else {}


def _get_default_headers(
    as_service: str | None = None, internal: bool = False, on_behalf_of: str | Principal | None = None
) -> dict[str, str]:
    """Get default headers for SDK requests.

    Args:
        as_service: If provided, use service principal headers (service:{name}).
                   If None, use the current request's auth context.
        internal: If True, include headers to mark requests as internal
                 (used for controller/background task requests).

    Returns:
        Headers dict combining auth and internal markers as needed.
    """
    headers: dict[str, str] = {}

    # Add internal request marker if requested
    if internal:
        headers.update(MARK_INTERNAL_REQUEST_HEADERS)

    # Add auth headers
    if as_service is not None:
        # Use service principal
        headers["X-NMP-Principal-Id"] = f"service:{as_service}"

        if on_behalf_of is not None:
            if isinstance(on_behalf_of, Principal):
                effective_principal = on_behalf_of.effective_principal
                headers["X-NMP-Principal-On-Behalf-Of"] = effective_principal.id
                if effective_principal.groups:
                    headers["X-NMP-Principal-On-Behalf-Of-Groups"] = ",".join(effective_principal.groups)
                if effective_principal.email:
                    headers["X-NMP-Principal-On-Behalf-Of-Email"] = effective_principal.email
            else:
                headers["X-NMP-Principal-On-Behalf-Of"] = on_behalf_of
    else:
        # Propagate the current user's auth context
        auth_headers = get_principal_auth_headers()
        if auth_headers:
            headers.update(auth_headers)

        elif (principal := principal_from_env()) is not None:
            # If we don't have auth_headers set yet, try loading them from env
            headers.update(principal.get_headers())

        if on_behalf_of is not None:
            headers.pop("X-NMP-Principal-On-Behalf-Of-Groups", None)
            headers.pop("X-NMP-Principal-On-Behalf-Of-Email", None)
            if isinstance(on_behalf_of, Principal):
                effective_principal = on_behalf_of.effective_principal
                headers["X-NMP-Principal-On-Behalf-Of"] = effective_principal.id
                if effective_principal.groups:
                    headers["X-NMP-Principal-On-Behalf-Of-Groups"] = ",".join(effective_principal.groups)
                if effective_principal.email:
                    headers["X-NMP-Principal-On-Behalf-Of-Email"] = effective_principal.email
            else:
                headers["X-NMP-Principal-On-Behalf-Of"] = on_behalf_of

    return headers


def get_platform_sdk(
    as_service: str | None = None,
    internal: bool = False,
    http_client: httpx.Client | None = None,
    on_behalf_of: str | Principal | None = None,
    base_url: str | None = None,
) -> NeMoPlatform:
    """
    Returns an instance of the NeMoPlatform SDK configured with the platform's base URL.

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
        Configured NeMoPlatform SDK instance.
    """
    platform_config = _get_platform_config()
    endpoint = resolve_platform_endpoint(platform_config)
    if _should_bootstrap_workload_identity(
        as_service=as_service,
        on_behalf_of=on_behalf_of,
        http_client=http_client,
        endpoint=endpoint,
    ):
        headers = _workload_identity_extra_headers(internal=internal)
        return _WorkloadIdentityRoutedNeMoPlatform(
            base_url=base_url or endpoint.connect_base_url,
            default_headers=headers if headers else None,
            platform_config=platform_config,
            platform_endpoint=endpoint,
        )

    headers = _get_default_headers(as_service, internal, on_behalf_of)
    return NeMoPlatform(
        base_url=base_url or endpoint.connect_base_url,
        http_client=_sync_sdk_http_client(endpoint, base_url, http_client),
        default_headers=headers if headers else None,
    )


def get_task_sdk(as_service: str, http_client: httpx.Client | None = None) -> NeMoPlatform:
    """Create an SDK for use inside a task container with on-behalf-of auth.

    Reads the job creator's principal from the NMP_PRINCIPAL environment variable
    (set by the jobs backend when launching task containers) and creates an SDK
    that authenticates as the given service while acting on behalf of the job creator.

    Args:
        as_service: Service name for the service principal (e.g., "customizer").
        http_client: Optional sync HTTP client to use for requests.

    Returns:
        Configured NeMoPlatform SDK with internal + on-behalf-of headers.
    """
    if http_client is None and is_workload_identity_token_file_set():
        return get_platform_sdk(internal=True)

    principal = principal_from_env()
    if principal is None:
        logger.warning(
            "NMP_PRINCIPAL not set; task SDK will authenticate as service:%s without on-behalf-of delegation",
            as_service,
        )
    return get_platform_sdk(
        as_service=as_service,
        internal=True,
        http_client=http_client,
        on_behalf_of=principal.effective_principal if principal else None,
    )


def get_async_task_sdk(as_service: str, http_client: Optional[httpx.AsyncClient] = None) -> AsyncNeMoPlatform:
    """Async counterpart of :func:`get_task_sdk` for use inside a task container.

    Reads the job creator's principal from ``NMP_PRINCIPAL`` and creates an async SDK that
    authenticates as the given service while acting on behalf of the job creator with the full
    delegated identity (on-behalf-of id, email, and groups). Wire-identical to :func:`get_task_sdk`.

    Args:
        as_service: Service name for the service principal (e.g., "evaluator").
        http_client: Optional async HTTP client to use for requests.

    Returns:
        Configured AsyncNeMoPlatform SDK with internal + on-behalf-of headers.
    """
    if http_client is None and is_workload_identity_token_file_set():
        return get_async_platform_sdk(internal=True)

    principal = principal_from_env()
    if principal is None:
        logger.warning(
            "NMP_PRINCIPAL not set; async task SDK will authenticate as service:%s without on-behalf-of delegation",
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
    http_client: Optional[httpx.AsyncClient] = None,
    on_behalf_of: Optional[str | Principal] = None,
    base_url: str | None = None,
) -> AsyncNeMoPlatform:
    """
    Returns an instance of the AsyncNeMoPlatform SDK configured with the platform's base URL.

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
        Configured AsyncNeMoPlatform SDK instance.
    """
    platform_config = _get_platform_config()
    endpoint = resolve_platform_endpoint(platform_config)
    if _should_bootstrap_workload_identity(
        as_service=as_service,
        on_behalf_of=on_behalf_of,
        http_client=http_client,
        endpoint=endpoint,
    ):
        headers = _workload_identity_extra_headers(internal=internal)
        return _WorkloadIdentityRoutedAsyncNeMoPlatform(
            base_url=base_url or endpoint.connect_base_url,
            default_headers=headers if headers else None,
            platform_config=platform_config,
            platform_endpoint=endpoint,
        )

    headers = _get_default_headers(as_service, internal, on_behalf_of)
    return AsyncNeMoPlatform(
        base_url=base_url or endpoint.connect_base_url,
        http_client=_async_sdk_http_client(endpoint, base_url, http_client),
        default_headers=headers if headers else None,
    )


def get_request_scoped_sdk(
    base_sdk: AsyncNeMoPlatform,
) -> AsyncNeMoPlatform:
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
    headers = get_otel_headers().copy()
    headers.update(get_principal_auth_headers())

    # If we have headers to add, create a new SDK with them
    # This reuses the underlying HTTP client (lightweight operation)
    if headers:
        return base_sdk.with_options(default_headers=headers)

    return base_sdk


def get_request_scoped_sync_sdk(
    base_sdk: NeMoPlatform,
) -> NeMoPlatform:
    """Create a request-scoped sync SDK with current auth and observability headers."""

    headers = get_otel_headers().copy()
    headers.update(get_principal_auth_headers())

    if headers:
        return base_sdk.with_options(default_headers=headers)

    return base_sdk


def get_sdk_on_behalf_of(
    base_sdk: NeMoPlatform | AsyncNeMoPlatform,
    on_behalf_of: str | Principal,
) -> NeMoPlatform | AsyncNeMoPlatform:
    """Create an SDK with on-behalf-of headers for delegated access.

    Takes an existing SDK (typically created as a service principal) and returns
    a new SDK instance with X-NMP-Principal-On-Behalf-Of header added. This enables
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
    if isinstance(on_behalf_of, Principal):
        merged_headers["X-NMP-Principal-On-Behalf-Of"] = on_behalf_of.effective_principal.id
        if on_behalf_of.effective_principal.email:
            merged_headers["X-NMP-Principal-On-Behalf-Of-Email"] = on_behalf_of.effective_principal.email
        if on_behalf_of.effective_principal.groups:
            merged_headers["X-NMP-Principal-On-Behalf-Of-Groups"] = ",".join(on_behalf_of.effective_principal.groups)
    else:
        merged_headers["X-NMP-Principal-On-Behalf-Of"] = on_behalf_of
        merged_headers.pop("X-NMP-Principal-On-Behalf-Of-Groups", None)
        merged_headers.pop("X-NMP-Principal-On-Behalf-Of-Email", None)
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
# Entry-point provider for nemo_platform_plugin.sdk_provider
# ---------------------------------------------------------------------------


class PlatformSDKProvider:
    """Rich :class:`~nemo_platform_plugin.sdk_provider.SDKProvider` that uses
    platform internals (shared HTTP clients, URL routing, OTEL headers, auth
    context vars).

    Registered as a ``nemo.sdk_provider`` entry-point so it is
    discovered automatically when ``nmp-common`` is installed.
    """

    def get_task_sdk(self, service_name: str, http_client: httpx.Client | None = None) -> NeMoPlatform:
        return get_task_sdk(service_name, http_client=http_client)

    def get_async_task_sdk(self, service_name: str, http_client: httpx.AsyncClient | None = None) -> AsyncNeMoPlatform:
        return get_async_task_sdk(service_name, http_client=http_client)

    def get_platform_sdk(
        self,
        *,
        as_service: str | None = None,
        internal: bool = False,
        http_client: httpx.Client | None = None,
        on_behalf_of: str | Principal | None = None,
        base_url: str | None = None,
    ) -> NeMoPlatform:
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
    ) -> AsyncNeMoPlatform:
        return get_async_platform_sdk(
            as_service=as_service,
            internal=internal,
            on_behalf_of=on_behalf_of,
            base_url=base_url,
        )
