# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared provider request policy helpers."""

from collections.abc import Mapping

from fastapi import HTTPException, status
from jinja2 import Environment
from nemo_helix_plugin.models.types import ModelProvider
from nhx.common.auth.bearer import MalformedBearerTokenError
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.middleware import API_KEY_HEADER_PATH_PREFIXES, request_bearer_token

_DEFAULT_AUTH_HEADER_FORMAT = "Authorization: Bearer {{ auth_secret }}"
# The only scope family a forwarded caller credential may carry: it reaches inference and nothing else on the platform.
_FORWARDABLE_SCOPE_FAMILY = "inference"
# Renders HTTP headers, not HTML. Autoescape would corrupt some secret values.
_JINJA_ENV = Environment(autoescape=False)  # noqa: S701  # nosec B701


def render_auth_header(secret_value: str, auth_header_format: str | None) -> tuple[str, str]:
    """Render a provider authentication header from its validated template."""
    template = auth_header_format or _DEFAULT_AUTH_HEADER_FORMAT
    rendered = _JINJA_ENV.from_string(template).render(auth_secret=secret_value)
    header_name, _, header_value = rendered.partition(": ")
    return header_name, header_value


def upstream_credential(provider: ModelProvider, secret_value: str | None, headers: Mapping[str, str]) -> str | None:
    """The credential to send ``provider``: its stored key, or, with ``forward_caller_credential``, the caller's own.

    Raises:
        HTTPException: 403 when platform auth is on and the caller's credential isn't limited to the ``inference``
            scope, so a broader credential never reaches a provider.
    """
    if not provider.forward_caller_credential:
        return secret_value
    lowered = {name.lower(): value for name, value in headers.items()}
    try:
        credential = request_bearer_token(API_KEY_HEADER_PATH_PREFIXES[0], lowered)
    except MalformedBearerTokenError:
        return None
    auth_client = auth_client_context.get()
    if credential is None or auth_client is None or not auth_client.auth_enabled:
        return credential
    resolved = auth_client.resolved_bearer_token
    scopes = resolved.scopes if resolved is not None else lowered.get("x-nhx-scopes", "").split()
    if not scopes or any(scope.split(":", 1)[0] != _FORWARDABLE_SCOPE_FAMILY for scope in scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Model provider {provider.workspace}/{provider.name} calls models as its caller, so it accepts only "
                "a credential limited to the inference scope, such as an access key with scope inference."
            ),
        )
    return credential
