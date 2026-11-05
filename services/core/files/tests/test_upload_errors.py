# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request
from nhx.core.files.api.v2.filesets import endpoints
from nhx.core.files.exceptions import StorageAccessError


async def test_storage_authorization_failure_is_not_retryable(monkeypatch):
    fileset = SimpleNamespace(name="files", storage=None, custom_fields={})
    error = StorageAccessError("Access denied")
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
    assert caught.value.detail == str(error)
