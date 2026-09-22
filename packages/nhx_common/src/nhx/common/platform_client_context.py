# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared platform client construction context.

This module is the dependency tree for in-process platform clients.  It resolves
the configured platform endpoint, the typed-client runtime policy, and the auth
plan that should be used for the current call site.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from httpx._types import TimeoutTypes
from nemo_helix_plugin.client.auth import TokenProvider
from nemo_helix_plugin.client.client import NemoClientRuntime
from nemo_helix_plugin.client.constants import (
    WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR,
    is_workload_identity_token_file_set,
)
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.headers import AUTHORIZATION_HEADER
from nhx.common.auth.models import Principal
from nhx.common.auth.tasks import principal_from_env
from nhx.common.auth.workload_tokens import ServiceWorkloadAccessTokenProvider
from nhx.common.client_runtime import build_platform_client_runtime
from nhx.common.config import AuthConfig, Configuration, HelixConfig, get_auth_config
from nhx.common.observability import INTERNAL_REQUEST_HEADER, MARK_INTERNAL_REQUEST_HEADERS
from nhx.common.observability.otel import get_otel_headers
from nhx.common.platform_endpoint import HelixEndpoint, resolve_platform_endpoint

PRINCIPAL_ID_HEADER = "X-NHX-Principal-Id"
ACTOR_ALIASES_HEADER = "X-NHX-Actor-Aliases"
PRINCIPAL_OBO_HEADER = "X-NHX-Principal-On-Behalf-Of"
PRINCIPAL_OBO_GROUPS_HEADER = "X-NHX-Principal-On-Behalf-Of-Groups"
PRINCIPAL_OBO_EMAIL_HEADER = "X-NHX-Principal-On-Behalf-Of-Email"
SUBJECT_ACCOUNT_ID_HEADER = "X-NHX-Subject-Account-Id"
SUBJECT_ALIASES_HEADER = "X-NHX-Subject-Aliases"
DELEGATED_PRINCIPAL_HEADERS = (
    PRINCIPAL_OBO_GROUPS_HEADER,
    PRINCIPAL_OBO_EMAIL_HEADER,
    SUBJECT_ACCOUNT_ID_HEADER,
    SUBJECT_ALIASES_HEADER,
)

SyncRequestHook = Callable[[httpx.Request], None]
AsyncRequestHook = Callable[[httpx.Request], Awaitable[None]]


@dataclass(frozen=True)
class HelixRuntimeContext:
    """Platform-scoped runtime derived from one effective endpoint config."""

    platform_config: HelixConfig
    endpoint: HelixEndpoint
    runtime: NemoClientRuntime

    @property
    def base_url(self) -> str:
        return self.endpoint.connect_base_url

    def sync_sdk_http_client(
        self,
        *,
        http_client: httpx.Client | None,
        timeout: TimeoutTypes | None = None,
        limits: httpx.Limits | None = None,
        follow_redirects: bool | None = None,
        request_hooks: tuple[SyncRequestHook, ...] = (),
    ) -> httpx.Client:
        if http_client is not None:
            return self.endpoint.sync_sdk_http_client(http_client=http_client, request_hooks=request_hooks)
        return self.endpoint.sync_sdk_http_client(
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
            request_hooks=request_hooks,
        )

    def async_sdk_http_client(
        self,
        *,
        http_client: httpx.AsyncClient | None,
        timeout: TimeoutTypes | None = None,
        limits: httpx.Limits | None = None,
        follow_redirects: bool | None = None,
        request_hooks: tuple[AsyncRequestHook, ...] = (),
    ) -> httpx.AsyncClient:
        if http_client is not None:
            return self.endpoint.async_sdk_http_client(http_client=http_client, request_hooks=request_hooks)
        return self.endpoint.async_sdk_http_client(
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
            request_hooks=request_hooks,
        )

    def sync_nemo_http_client(self, *, http_client: httpx.Client | None) -> httpx.Client:
        if http_client is not None:
            return http_client
        return self.endpoint.sync_sdk_http_client()

    def async_nemo_http_client(self, *, http_client: httpx.AsyncClient | None) -> httpx.AsyncClient:
        if http_client is not None:
            return http_client
        return self.endpoint.async_sdk_http_client()


class HelixAuthPlan(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...

    @property
    def uses_workload_identity_bootstrap(self) -> bool: ...

    def nemo_client_auth(self) -> TokenProvider | None: ...

    def sync_sdk_request_hooks(self) -> tuple[SyncRequestHook, ...]: ...

    def async_sdk_request_hooks(self) -> tuple[AsyncRequestHook, ...]: ...


@dataclass(frozen=True)
class TrustedHeaderAuthPlan:
    headers: dict[str, str]

    @property
    def uses_workload_identity_bootstrap(self) -> bool:
        return False

    def nemo_client_auth(self) -> TokenProvider | None:
        return None

    def sync_sdk_request_hooks(self) -> tuple[SyncRequestHook, ...]:
        return ()

    def async_sdk_request_hooks(self) -> tuple[AsyncRequestHook, ...]:
        return ()


@dataclass(frozen=True)
class ServiceWorkloadTokenAuthPlan:
    headers: dict[str, str]
    provider: ServiceWorkloadAccessTokenProvider

    @property
    def uses_workload_identity_bootstrap(self) -> bool:
        return False

    def nemo_client_auth(self) -> ServiceWorkloadAccessTokenProvider:
        return self.provider

    def sync_sdk_request_hooks(self) -> tuple[SyncRequestHook, ...]:
        return (_sync_service_workload_token_request_hook(self.provider),)

    def async_sdk_request_hooks(self) -> tuple[AsyncRequestHook, ...]:
        return (_async_service_workload_token_request_hook(self.provider),)


@dataclass(frozen=True)
class WorkloadIdentityBootstrapAuthPlan:
    headers: dict[str, str]
    subject_token_file: Path
    base_url: str

    @property
    def uses_workload_identity_bootstrap(self) -> bool:
        return True

    def nemo_client_auth(self) -> TokenProvider:
        from nemo_helix_plugin.client.oidc_factory import resolve_workload_exchange_provider

        return resolve_workload_exchange_provider(base_url=self.base_url, subject_token_file=self.subject_token_file)

    def sync_sdk_request_hooks(self) -> tuple[SyncRequestHook, ...]:
        return ()

    def async_sdk_request_hooks(self) -> tuple[AsyncRequestHook, ...]:
        return ()


@dataclass(frozen=True)
class HelixClientContext:
    runtime_context: HelixRuntimeContext
    auth_plan: HelixAuthPlan
    default_headers: dict[str, str]

    @property
    def endpoint(self) -> HelixEndpoint:
        return self.runtime_context.endpoint

    @property
    def runtime(self) -> NemoClientRuntime:
        return self.runtime_context.runtime

    @property
    def base_url(self) -> str:
        return self.runtime_context.base_url

    def default_headers_or_none(self) -> dict[str, str] | None:
        return self.default_headers or None

    def nemo_client_auth(self) -> TokenProvider | None:
        return self.auth_plan.nemo_client_auth()

    def sync_sdk_request_hooks(self) -> tuple[SyncRequestHook, ...]:
        return self.auth_plan.sync_sdk_request_hooks()

    def async_sdk_request_hooks(self) -> tuple[AsyncRequestHook, ...]:
        return self.auth_plan.async_sdk_request_hooks()

    def sync_sdk_http_client(
        self,
        http_client: httpx.Client | None,
        *,
        timeout: TimeoutTypes | None = None,
        limits: httpx.Limits | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Client:
        return self.runtime_context.sync_sdk_http_client(
            http_client=http_client,
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
            request_hooks=self.sync_sdk_request_hooks(),
        )

    def async_sdk_http_client(
        self,
        http_client: httpx.AsyncClient | None,
        *,
        timeout: TimeoutTypes | None = None,
        limits: httpx.Limits | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.AsyncClient:
        return self.runtime_context.async_sdk_http_client(
            http_client=http_client,
            timeout=timeout,
            limits=limits,
            follow_redirects=follow_redirects,
            request_hooks=self.async_sdk_request_hooks(),
        )


def _sync_service_workload_token_request_hook(
    provider: ServiceWorkloadAccessTokenProvider,
) -> SyncRequestHook:
    def set_authorization(request: httpx.Request) -> None:
        if "Authorization" not in request.headers:
            request.headers["Authorization"] = f"Bearer {provider.get_access_token()}"

    return set_authorization


def _async_service_workload_token_request_hook(
    provider: ServiceWorkloadAccessTokenProvider,
) -> AsyncRequestHook:
    async def set_authorization(request: httpx.Request) -> None:
        if "Authorization" not in request.headers:
            request.headers["Authorization"] = f"Bearer {await provider.get_access_token_async()}"

    return set_authorization


def forwardable_otel_headers(*, include_internal: bool) -> dict[str, str]:
    internal_header = INTERNAL_REQUEST_HEADER.lower()
    return {
        name: value
        for name, value in get_otel_headers().items()
        if (include_internal and name.lower() == internal_header) or not name.lower().startswith("x-nhx-")
    }


def should_bootstrap_workload_identity(
    *,
    as_service: str | None,
    on_behalf_of: str | Principal | None,
    has_explicit_http_client: bool,
    endpoint: HelixEndpoint,
) -> bool:
    return (
        as_service is None
        and on_behalf_of is None
        and not has_explicit_http_client
        and endpoint.transport != "uds"
        and is_workload_identity_token_file_set()
    )


def should_use_service_workload_token(
    config: AuthConfig,
    *,
    as_service: str | None,
) -> bool:
    return bool(as_service is not None and config.enabled and config.oidc.workload_token_exchange_enabled)


def internal_request_headers(*, internal: bool) -> dict[str, str]:
    return MARK_INTERNAL_REQUEST_HEADERS.copy() if internal else {}


def current_principal_auth_headers() -> dict[str, str]:
    auth_client = auth_client_context.get()
    if auth_client is None or not auth_client.principal:
        return {}
    if auth_client.config.enabled and auth_client.config.oidc.workload_token_exchange_enabled:
        if auth_client.bearer_token:
            return {AUTHORIZATION_HEADER: f"Bearer {auth_client.bearer_token}"}
        return {}
    return auth_client.principal.get_headers()


def delegated_principal_headers(principal: str | Principal) -> dict[str, str]:
    if not isinstance(principal, Principal):
        return {PRINCIPAL_OBO_HEADER: principal}

    effective_principal = principal.effective_principal
    headers = {PRINCIPAL_OBO_HEADER: effective_principal.id}
    if effective_principal.groups:
        headers[PRINCIPAL_OBO_GROUPS_HEADER] = ",".join(effective_principal.groups)
    if effective_principal.email:
        headers[PRINCIPAL_OBO_EMAIL_HEADER] = effective_principal.email
    if effective_principal.account_id:
        headers[SUBJECT_ACCOUNT_ID_HEADER] = effective_principal.account_id
    if effective_principal.authz_aliases:
        headers[SUBJECT_ALIASES_HEADER] = ",".join(effective_principal.authz_aliases)
    return headers


def _trusted_service_principal_headers(
    service_name: str,
    *,
    on_behalf_of: str | Principal | None,
) -> dict[str, str]:
    service_principal_id = f"service:{service_name}"
    headers = {
        PRINCIPAL_ID_HEADER: service_principal_id,
        ACTOR_ALIASES_HEADER: service_principal_id,
    }
    if on_behalf_of is not None:
        headers.update(delegated_principal_headers(on_behalf_of))
    return headers


def _request_principal_headers(*, on_behalf_of: str | Principal | None) -> dict[str, str]:
    headers = current_principal_auth_headers().copy()
    if not headers:
        principal = principal_from_env()
        if principal is not None:
            headers = principal.get_headers()

    if on_behalf_of is not None:
        for header in DELEGATED_PRINCIPAL_HEADERS:
            headers.pop(header, None)
        headers.update(delegated_principal_headers(on_behalf_of))

    return headers


def platform_auth_headers(
    config: AuthConfig,
    *,
    as_service: str | None,
    internal: bool,
    on_behalf_of: str | Principal | None,
) -> dict[str, str]:
    headers = internal_request_headers(internal=internal)
    if should_use_service_workload_token(config, as_service=as_service):
        return headers

    if as_service is not None:
        headers.update(_trusted_service_principal_headers(as_service, on_behalf_of=on_behalf_of))
        return headers

    headers.update(_request_principal_headers(on_behalf_of=on_behalf_of))
    return headers


def service_principal_auth_headers(service_name: str) -> dict[str, str]:
    auth_client = auth_client_context.get()
    config = auth_client.config if auth_client is not None else get_auth_config()
    on_behalf_of: Principal | None = None
    if auth_client is not None and auth_client.principal and auth_client.principal.id:
        effective = auth_client.principal.effective_principal
        if effective.caller_kind != "service_principal":
            on_behalf_of = effective

    if config.enabled and config.oidc.workload_token_exchange_enabled:
        token = ServiceWorkloadAccessTokenProvider(config, service_name, on_behalf_of).get_access_token()
        return {AUTHORIZATION_HEADER: f"Bearer {token}"}

    return _trusted_service_principal_headers(service_name, on_behalf_of=on_behalf_of)


def build_platform_auth_plan(
    config: AuthConfig,
    *,
    endpoint: HelixEndpoint,
    as_service: str | None,
    internal: bool,
    on_behalf_of: str | Principal | None,
    has_explicit_http_client: bool,
) -> HelixAuthPlan:
    if should_use_service_workload_token(config, as_service=as_service):
        service_name = as_service or "unknown"
        return ServiceWorkloadTokenAuthPlan(
            headers=internal_request_headers(internal=internal),
            provider=ServiceWorkloadAccessTokenProvider(config, service_name, on_behalf_of),
        )

    if should_bootstrap_workload_identity(
        as_service=as_service,
        on_behalf_of=on_behalf_of,
        has_explicit_http_client=has_explicit_http_client,
        endpoint=endpoint,
    ):
        return WorkloadIdentityBootstrapAuthPlan(
            headers=internal_request_headers(internal=internal),
            subject_token_file=Path(os.environ[WORKLOAD_IDENTITY_TOKEN_FILE_ENVVAR]),
            base_url=endpoint.connect_base_url,
        )

    return TrustedHeaderAuthPlan(
        headers=platform_auth_headers(
            config,
            as_service=as_service,
            internal=internal,
            on_behalf_of=on_behalf_of,
        )
    )


def _get_platform_config(platform_config: HelixConfig | None) -> HelixConfig:
    if platform_config is not None:
        return platform_config
    configured = Configuration.get_platform_config()
    if not isinstance(configured, HelixConfig):
        raise TypeError("Expected HelixConfig from Configuration.get_platform_config()")
    return configured


def build_platform_runtime_context(
    *,
    platform_config: HelixConfig | None = None,
    base_url: str | None = None,
) -> HelixRuntimeContext:
    effective_config = _get_platform_config(platform_config)
    endpoint = resolve_platform_endpoint(effective_config, base_url=base_url)
    return HelixRuntimeContext(
        platform_config=effective_config,
        endpoint=endpoint,
        runtime=build_platform_client_runtime(endpoint),
    )


def build_platform_client_context(
    *,
    platform_config: HelixConfig | None = None,
    runtime_context: HelixRuntimeContext | None = None,
    base_url: str | None = None,
    as_service: str | None = None,
    internal: bool = False,
    on_behalf_of: str | Principal | None = None,
    has_explicit_http_client: bool = False,
    include_otel_headers: bool = False,
) -> HelixClientContext:
    if runtime_context is None:
        runtime_context = build_platform_runtime_context(platform_config=platform_config, base_url=base_url)
    auth_plan = build_platform_auth_plan(
        get_auth_config(),
        endpoint=runtime_context.endpoint,
        as_service=as_service,
        internal=internal,
        on_behalf_of=on_behalf_of,
        has_explicit_http_client=has_explicit_http_client,
    )
    headers = dict(auth_plan.headers)
    if include_otel_headers:
        headers.update(forwardable_otel_headers(include_internal=False))
    return HelixClientContext(
        runtime_context=runtime_context,
        auth_plan=auth_plan,
        default_headers=headers,
    )
