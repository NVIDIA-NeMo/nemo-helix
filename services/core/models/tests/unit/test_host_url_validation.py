# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ModelProvider.host_url scheme validation.

The validator (`_validate_host_url`) rejects a scheme-less URL like
`"inference-api.nvidia.com"` at the API boundary. Without it, the entity is
accepted at create time but model discovery fails later with an opaque 502
backend networking error. Validating early gives the user a clear 4xx naming
the missing scheme instead.
"""

import pytest
from nmp.core.models.schemas import (
    CreateModelProviderRequest,
    UpsertModelProviderRequest,
    _validate_host_url,
)
from pydantic import ValidationError


class TestHostUrlValidationHelper:
    @pytest.mark.parametrize(
        "value",
        [
            "https://integrate.api.nvidia.com/v1",
            "http://mock.local",
            "https://inference-api.nvidia.com",
            "http://localhost:8000",
        ],
    )
    def test_accepts_urls_with_scheme(self, value: str) -> None:
        assert _validate_host_url(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "inference-api.nvidia.com",  # bare host — the NMP-186 repro
            "integrate.api.nvidia.com/v1",
            "localhost:8000",
            "ftp://inference-api.nvidia.com",  # wrong scheme
            "//inference-api.nvidia.com",  # scheme-relative
        ],
    )
    def test_rejects_urls_without_http_scheme(self, value: str) -> None:
        with pytest.raises(ValueError, match="must include a scheme"):
            _validate_host_url(value)

    def test_error_names_the_offending_value(self) -> None:
        with pytest.raises(ValueError, match="inference-api.nvidia.com"):
            _validate_host_url("inference-api.nvidia.com")


class TestHostUrlValidationAtModelBoundary:
    """The validator is wired onto the request/response schemas, so a
    scheme-less URL is rejected when the model is constructed (i.e. at the
    API boundary, before the entity is ever persisted)."""

    def test_create_request_rejects_schemeless_url(self) -> None:
        with pytest.raises(ValidationError, match="must include a scheme"):
            CreateModelProviderRequest(name="my-provider", host_url="inference-api.nvidia.com")

    def test_create_request_accepts_https_url(self) -> None:
        request = CreateModelProviderRequest(name="my-provider", host_url="https://inference-api.nvidia.com")
        assert request.host_url == "https://inference-api.nvidia.com"

    def test_upsert_request_rejects_schemeless_url(self) -> None:
        with pytest.raises(ValidationError, match="must include a scheme"):
            UpsertModelProviderRequest(host_url="inference-api.nvidia.com")

    def test_upsert_request_accepts_https_url(self) -> None:
        request = UpsertModelProviderRequest(host_url="https://inference-api.nvidia.com")
        assert request.host_url == "https://inference-api.nvidia.com"
