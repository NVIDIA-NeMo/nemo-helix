# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Documentation-only authentication metadata; no runtime security dependencies."""

from copy import deepcopy

BEARER_SCHEME_NAME = "HelixBearer"
BEARER_AUTH_OPENAPI = {"security": [{BEARER_SCHEME_NAME: []}]}
BEARER_SECURITY_SCHEME = {
    "type": "http",
    "scheme": "bearer",
    "description": (
        "Send `Authorization: Bearer <token>` when setting the HTTP header directly (for example, with curl). "
        "In a bearer-auth token field, supply only the raw Helix access token; "
        "the client adds the Bearer prefix."
    ),
    "x-fern-bearer": {
        "name": "accessToken",
        "env": "NHX_ACCESS_TOKEN",
        "placeholder": "YOUR_ACCESS_TOKEN",
    },
}


def register_bearer_security_scheme(spec: dict) -> dict:
    """Complete explicitly annotated operations without imposing global security.

    Platform and service OpenAPI hooks call this while constructing their cached
    schema. No route dependencies, authentication logic, or deployment configuration
    are changed.
    """
    if any(
        BEARER_SCHEME_NAME in requirement
        for item in spec.get("paths", {}).values()
        for operation in item.values()
        if isinstance(operation, dict)
        for requirement in operation.get("security", [])
    ):
        schemes = spec.setdefault("components", {}).setdefault("securitySchemes", {})
        if BEARER_SCHEME_NAME in schemes and schemes[BEARER_SCHEME_NAME] != BEARER_SECURITY_SCHEME:
            raise ValueError(f"Conflicting security scheme: {BEARER_SCHEME_NAME}")
        schemes[BEARER_SCHEME_NAME] = deepcopy(BEARER_SECURITY_SCHEME)
    return spec
