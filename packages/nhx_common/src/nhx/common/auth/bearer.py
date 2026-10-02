# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class MalformedBearerTokenError(ValueError):
    """Raised when an Authorization header uses Bearer but has invalid credentials."""


def parse_bearer_authorization_header(auth_header: str | None) -> str | None:
    """Return the bearer token, None for non-bearer auth, or raise for malformed bearer auth."""
    if auth_header is None:
        return None

    parts = auth_header.strip().split()
    if not parts:
        return None
    if parts[0].lower() != "bearer":
        return None
    if len(parts) != 2:
        raise MalformedBearerTokenError("Bearer authorization must include exactly one token")
    return parts[1]
