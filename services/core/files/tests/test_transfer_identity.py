# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit ID references cannot retarget a replacement with the old name."""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.core.files.api.endpoint_helpers import get_fileset
from nhx.core.files.app.backends.local import LocalStorageConfig
from nhx.core.files.entities import Fileset


async def test_explicit_id_uses_identity_and_validates_workspace():
    client = AsyncMock(spec=EntityClient)
    fileset = Fileset(
        purpose="generic", name="source", workspace="default", storage=LocalStorageConfig(path="/tmp/transfer-identity")
    )
    client.get_by_id.return_value = fileset
    assert await get_fileset("default", "id:selected", client) == fileset
    client.get_by_id.assert_awaited_once_with(Fileset, "selected")
    client.get.assert_not_awaited()
    with pytest.raises(HTTPException) as denied:
        await get_fileset("other", "id:selected", client)
    assert denied.value.status_code == 404


async def test_deleted_id_never_falls_back_to_name():
    client = AsyncMock(spec=EntityClient)
    client.get_by_id.side_effect = EntityNotFoundError("deleted")
    with pytest.raises(HTTPException) as missing:
        await get_fileset("default", "id:deleted", client)
    assert missing.value.status_code == 404
    client.get.assert_not_awaited()
