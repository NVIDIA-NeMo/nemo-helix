# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for model provider host_url scheme validation.

The validator (`_validate_host_url`) rejects a scheme-less URL like
`"inference-api.nvidia.com"` at the write boundary (create/upsert request
schemas). Without it, the entity is accepted at create time but model discovery
fails later with an opaque 502 backend networking error. Validating early gives
the user a clear 4xx naming the missing scheme instead.

It is deliberately NOT applied to the response/domain `ModelProvider` schema,
which is rebuilt from persisted rows on every read — re-validating there would
turn a provider stored before this check landed into an unreadable 500.
"""

from datetime import datetime, timezone

import pytest
from nhx.core.models.schemas import (
    CreateModelProviderRequest,
    ModelProvider,
    UpsertModelProviderRequest,
    _validate_host_url,
)
from pydantic import ValidationError

# Builders for the two write-boundary request schemas. Create requires a name;
# upsert does not — otherwise host_url validation is identical, so the boundary
# tests below run against both.
_WRITE_REQUEST_BUILDERS = [
    pytest.param(lambda url: CreateModelProviderRequest(name="my-provider", host_url=url), id="create"),
    pytest.param(lambda url: UpsertModelProviderRequest(host_url=url), id="upsert"),
]


class TestHostUrlValidationHelper:
    @pytest.mark.parametrize(
        "value",
        [
            "https://integrate.api.nvidia.com/v1",
            "http://mock.local",
            "https://inference-api.nvidia.com",
            "http://localhost:8000",
            "HTTPS://inference-api.nvidia.com",  # uppercase scheme — urlparse lowercases .scheme, accepted
        ],
    )
    def test_accepts_urls_with_scheme(self, value: str) -> None:
        assert _validate_host_url(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "inference-api.nvidia.com",  # bare host — the reported repro
            "integrate.api.nvidia.com/v1",
            "localhost:8000",
            "ftp://inference-api.nvidia.com",  # wrong scheme
            "//inference-api.nvidia.com",  # scheme-relative
            "",  # empty
            "https:inference-api.nvidia.com",  # scheme but no authority/hostname
            "https://",  # scheme + delimiter but no host
            "http:/v1/models",  # single-slash, hostless
        ],
    )
    def test_rejects_urls_without_http_scheme(self, value: str) -> None:
        with pytest.raises(ValueError, match="must include a scheme"):
            _validate_host_url(value)

    def test_error_names_the_offending_value(self) -> None:
        with pytest.raises(ValueError, match="inference-api.nvidia.com"):
            _validate_host_url("inference-api.nvidia.com")


class TestHostUrlValidationAtWriteBoundary:
    """The validator is wired onto the create/upsert REQUEST schemas only, so a
    scheme-less URL is rejected when a write is attempted — before the entity is
    ever persisted."""

    @pytest.mark.parametrize("build", _WRITE_REQUEST_BUILDERS)
    def test_request_rejects_schemeless_url(self, build) -> None:
        with pytest.raises(ValidationError, match="must include a scheme"):
            build("inference-api.nvidia.com")

    @pytest.mark.parametrize("build", _WRITE_REQUEST_BUILDERS)
    def test_request_accepts_https_url(self, build) -> None:
        request = build("https://inference-api.nvidia.com")
        assert request.host_url == "https://inference-api.nvidia.com"


class TestResponseSchemaDoesNotRevalidate:
    """The response/domain ModelProvider schema is rebuilt from persisted rows on
    every read (via _entity_to_schema). It must NOT re-validate host_url, or a
    provider stored before this check landed (exactly the reported scenario)
    would become an unreadable/un-listable 500. This pins that reads of a legacy
    scheme-less host_url still succeed."""

    def test_response_schema_accepts_legacy_schemeless_url(self) -> None:
        now = datetime.now(timezone.utc)
        provider = ModelProvider(
            name="legacy-provider",
            workspace="default",
            host_url="inference-api.nvidia.com",  # no scheme — a pre-existing row
            created_at=now,
            updated_at=now,
        )
        assert provider.host_url == "inference-api.nvidia.com"
