# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared provider request policy helpers."""

import re
from collections.abc import Mapping

from fastapi import HTTPException, status
from nemo_helix_plugin.models.types import ModelProvider
from nhx.common.auth.bearer import MalformedBearerTokenError
from nhx.common.auth.dependencies import auth_client_context
from nhx.common.auth.middleware import API_KEY_HEADER_PATH_PREFIXES, request_bearer_token

_DEFAULT_AUTH_HEADER_FORMAT = "Authorization: Bearer <auth_secret>"

# ``auth_header_format`` is literal text around exactly one ``<auth_secret>`` placeholder (or the legacy
# ``{{ auth_secret }}`` spelling). It is
# never evaluated by a template engine: providers stored before this was enforced may carry hostile
# templates, so the format is re-validated here on every use. Keep in sync with the models service
# validator (``nhx.core.models.schemas._validate_auth_header_format``).
_AUTH_SECRET_PLACEHOLDER_RE = re.compile(r"<auth_secret>|\{\{[ \t]*auth_secret[ \t]*\}\}")
_AUTH_HEADER_FORBIDDEN_CHARS = frozenset("{}\r\n")
# The only scope a forwarded caller credential may carry: it reaches inference and nothing else on the platform.
_FORWARDABLE_SCOPES = frozenset({"inference"})


def render_auth_header(secret_value: str, auth_header_format: str | None) -> tuple[str, str]:
    """Build a provider authentication header from its ``auth_header_format``.

    Raises:
        ValueError: If the format is not literal text with exactly one ``<auth_secret>``
            (or legacy ``{{ auth_secret }}``) placeholder. The offending format is not echoed back.
    """
    template = auth_header_format or _DEFAULT_AUTH_HEADER_FORMAT
    # Splitting on the placeholder yields [prefix, suffix] iff it appears exactly once.
    parts = _AUTH_SECRET_PLACEHOLDER_RE.split(template)
    if len(parts) != 2 or any(_AUTH_HEADER_FORBIDDEN_CHARS.intersection(part) for part in parts):
        raise ValueError(
            "Provider auth_header_format is invalid: it must contain exactly one '<auth_secret>' "
            "(or legacy '{{ auth_secret }}') placeholder"
        )
    rendered = parts[0] + secret_value + parts[1]
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
    if not scopes or not set(scopes) <= _FORWARDABLE_SCOPES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Model provider {provider.workspace}/{provider.name} calls models as its caller, so it accepts only "
                "a credential limited to the inference scope, such as an access key with scope inference."
            ),
        )
    return credential
