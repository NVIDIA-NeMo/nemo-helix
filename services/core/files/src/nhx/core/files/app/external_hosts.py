# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External host validation for the Files service."""

from __future__ import annotations

from urllib.parse import urlparse

from nhx.common.config import get_service_config


class ExternalHostNotAllowedError(ValueError):
    """Raised when a host URL is not in the allowed external hosts list."""


class ExternalHostInvalidError(ValueError):
    """Raised when a host URL is malformed or otherwise invalid for external host validation."""


def _normalize_authority(url_str: str) -> str:
    """Return a comparable (scheme, netloc) key for the URL, with default ports normalized."""
    parsed = urlparse(url_str)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url_str!r}")
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    # Normalize default ports so https://host and https://host:443 match
    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        if (scheme == "https" and port == "443") or (scheme == "http" and port == "80"):
            netloc = host
    return f"{scheme}://{netloc}"


def validate_external_host(host_url: str, allowed_hosts: list[str] | None = None) -> None:
    """Validate that host_url is in the allowed external hosts list.

    Comparison is by scheme + netloc (authority), so trailing slashes and
    default ports match. The caller (e.g. Files service) passes the list
    from its own config.

    Args:
        host_url: The host or base URL to validate (e.g. NGC host or HuggingFace endpoint).
        allowed_hosts: List of allowed base URLs (e.g. from FilesConfig.get_allowed_external_hosts()).

    Raises:
        ExternalHostInvalidError: If host_url is not a valid URL.
        ExternalHostNotAllowedError: If host_url's authority is not in the allowed list.
    """
    if allowed_hosts is None:
        from nhx.core.files.config import FilesConfig

        allowed_hosts = get_service_config(FilesConfig).get_allowed_external_hosts()

    try:
        candidate = _normalize_authority(host_url)
    except ValueError as e:
        raise ExternalHostInvalidError(f"Invalid URL for external host: {host_url!r}") from e

    allowed_set = set()
    for allowed in allowed_hosts:
        allowed = allowed.strip()
        if not allowed:
            continue
        try:
            allowed_set.add(_normalize_authority(allowed))
        except ValueError:
            continue

    if candidate not in allowed_set:
        raise ExternalHostNotAllowedError(f"Host {host_url!r} is not in the allowed external hosts list.")
