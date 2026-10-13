# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split platform auth from headers copied onto every guardrails model call.

Trace and caller headers propagate to every model. Platform identity is attached
only when the model URL has the same origin as the resolved platform endpoint.
"""

from __future__ import annotations

from contextvars import ContextVar

import httpx
from nhx.common.auth.headers import AUTHENTICATION_CONTEXT_HEADERS
from nhx.common.observability.otel import get_otel_headers
from nhx.common.platform_client_context import service_principal_auth_headers_async
from nhx.common.platform_endpoint import resolve_platform_endpoint
from nhx.guardrails.app.utils.context_utils import (
    get_request_default_headers_from_context,
    set_request_default_headers_into_context,
)
from nhx.guardrails.entities.values._private import ModelParameters, RailsConfig

_platform_auth_headers_var: ContextVar[dict[str, str]] = ContextVar("platform_auth_headers", default={})


def strip_platform_auth_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop Authorization and trusted principal headers from a propagated bag."""
    return {name: value for name, value in headers.items() if name.lower() not in AUTHENTICATION_CONTEXT_HEADERS}


def set_platform_auth_headers_into_context(headers: dict[str, str]) -> None:
    """Store platform auth headers for this request without mixing them into model defaults."""
    _platform_auth_headers_var.set(dict(headers))


def get_platform_auth_headers_from_context() -> dict[str, str]:
    """Return platform auth headers minted for this request."""
    return dict(_platform_auth_headers_var.get())


def is_platform_endpoint_url(url: str | None) -> bool:
    """Return whether ``url`` has the same HTTP origin as the platform endpoint."""
    candidate = _http_origin(url)
    if candidate is None:
        return False
    try:
        endpoint = resolve_platform_endpoint()
    except ValueError:
        return False
    return candidate == _http_origin(endpoint.connect_base_url)


def headers_for_model_endpoint(endpoint_url: str | None) -> dict[str, str]:
    """Return propagated headers, plus platform auth when ``endpoint_url`` is the platform."""
    headers = dict(get_request_default_headers_from_context())
    if is_platform_endpoint_url(endpoint_url):
        headers.update(get_platform_auth_headers_from_context())
    return headers


async def publish_downstream_request_headers(custom_headers: dict[str, str]) -> None:
    """Publish trace and caller headers, and mint platform auth into a separate context."""
    propagated = strip_platform_auth_headers(custom_headers)
    set_request_default_headers_into_context({**propagated, **get_otel_headers()})
    set_platform_auth_headers_into_context(await service_principal_auth_headers_async("guardrails"))


def apply_platform_auth_headers(config: RailsConfig) -> RailsConfig:
    """Add platform auth to models whose base URL is the platform endpoint."""
    platform_headers = get_platform_auth_headers_from_context()
    if not platform_headers or not config.models:
        return config

    models = []
    for model in config.models:
        raw = model.parameters.model_dump(exclude_none=True) if model.parameters is not None else {}
        base_url = raw.get("base_url")
        if isinstance(base_url, str) and is_platform_endpoint_url(base_url):
            default_headers = dict(raw.get("default_headers") or {})
            default_headers.update(platform_headers)
            raw["default_headers"] = default_headers
            model = model.model_copy(update={"parameters": ModelParameters.model_validate(raw)})
        models.append(model)
    return config.model_copy(update={"models": models})


def _http_origin(url: str | None) -> tuple[str, str, int] | None:
    if not url:
        return None
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.host:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return (parsed.scheme, parsed.host.lower(), port)
