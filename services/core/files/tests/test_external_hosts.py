# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for external host validation."""

import pytest
from nhx.core.files.app.external_hosts import (
    ExternalHostInvalidError,
    ExternalHostNotAllowedError,
    validate_external_host,
)


class TestValidateExternalHost:
    """Tests for validate_external_host with explicit allowed_hosts."""

    def test_allowed_host_matches_exact(self) -> None:
        """URL with matching scheme+netloc does not raise."""
        validate_external_host(
            "https://api.ngc.nvidia.com",
            allowed_hosts=["https://api.ngc.nvidia.com"],
        )

    def test_allowed_host_matches_with_trailing_slash(self) -> None:
        """URL with trailing slash matches same host."""
        validate_external_host(
            "https://api.ngc.nvidia.com/",
            allowed_hosts=["https://api.ngc.nvidia.com"],
        )
        validate_external_host(
            "https://huggingface.co",
            allowed_hosts=["https://huggingface.co/"],
        )

    def test_allowed_host_matches_with_default_port(self) -> None:
        """Explicit :443 for https and :80 for http normalize to match."""
        validate_external_host(
            "https://api.ngc.nvidia.com:443",
            allowed_hosts=["https://api.ngc.nvidia.com"],
        )
        validate_external_host(
            "http://internal.example.com:80",
            allowed_hosts=["http://internal.example.com"],
        )

    def test_disallowed_host_raises(self) -> None:
        """URL whose authority is not in the list raises ExternalHostNotAllowedError."""
        with pytest.raises(ExternalHostNotAllowedError, match="not in the allowed"):
            validate_external_host(
                "https://disallowed.example.com",
                allowed_hosts=["https://api.ngc.nvidia.com", "https://huggingface.co"],
            )

    def test_invalid_url_raises_external_host_invalid_error(self) -> None:
        """Invalid URL (e.g. missing scheme or netloc) raises ExternalHostInvalidError."""
        with pytest.raises(ExternalHostInvalidError, match="Invalid URL"):
            validate_external_host(
                "not-a-valid-url",
                allowed_hosts=["https://api.ngc.nvidia.com"],
            )
        with pytest.raises(ExternalHostInvalidError, match="Invalid URL"):
            validate_external_host(
                "://missing-scheme.com",
                allowed_hosts=["https://api.ngc.nvidia.com"],
            )

    def test_empty_allowed_list_rejects(self) -> None:
        """When allowed list is empty (or only invalid entries), any host is rejected."""
        with pytest.raises(ExternalHostNotAllowedError):
            validate_external_host(
                "https://api.ngc.nvidia.com",
                allowed_hosts=[],
            )

    def test_whitespace_in_allowed_list_stripped(self) -> None:
        """Allowed list entries are stripped; candidate must match normalized form."""
        validate_external_host(
            "https://custom.example.com",
            allowed_hosts=["  https://custom.example.com  "],
        )
