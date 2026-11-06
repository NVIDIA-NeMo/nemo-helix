# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Upload failures retain their HTTP status without exposing provider details."""

import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from nhx.core.files.api.endpoint_helpers import UPSTREAM_STORAGE_ACCESS_DETAIL
from nhx.core.files.api.v2.filesets import endpoints
from nhx.core.files.exceptions import StorageAccessError


@pytest.mark.parametrize(
    "backend_detail",
    [
        "Access denied for arn:aws:iam::123456789012:role/storage on arn:aws:s3:::internal-bucket/file",
        "ExpiredToken: the upstream storage session token has expired",
    ],
)
async def test_storage_authorization_failure_is_not_retryable(monkeypatch, caplog, backend_detail):
    """Keep authorization failures non-retryable and provider diagnostics in server logs."""
    fileset = SimpleNamespace(name="files", storage=None, custom_fields={})
    error = StorageAccessError(backend_detail)
    caplog.set_level(logging.ERROR, logger=endpoints.__name__)
    monkeypatch.setattr(endpoints, "get_fileset", AsyncMock(return_value=fileset))
    monkeypatch.setattr(endpoints, "resolve_storage_secrets_for_user", AsyncMock(return_value={}))
    monkeypatch.setattr(
        endpoints, "storage_impl_factory", lambda *args: SimpleNamespace(upload=AsyncMock(side_effect=error))
    )
    monkeypatch.setattr(endpoints, "scoped_app_ctx", lambda *args: nullcontext())

    monkeypatch.setattr(endpoints, "streaming_file_upload", lambda *args: nullcontext())
    with pytest.raises(HTTPException) as caught:
        await endpoints.upload_file(
            "ws",
            "files",
            "file",
            Request(scope={"type": "http", "headers": [(b"content-length", b"5")]}),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
    assert caught.value.status_code == 403
    assert caught.value.detail == UPSTREAM_STORAGE_ACCESS_DETAIL
    assert backend_detail not in caught.value.detail
    assert backend_detail in caplog.text
