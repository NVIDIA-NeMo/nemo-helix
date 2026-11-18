# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure bounds for revision persistence."""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException
from nemo_helix_plugin.client.errors import NemoTransportError
from nhx.common.entities.client import EntityClient, EntityConflictError, EntityNotFoundError, EntityStoreError
from nhx.core.files.app.revision import advance_fileset_revision
from nhx.core.files.entities import Fileset


def _fileset() -> Fileset:
    result = Fileset(
        name="files", workspace="default", purpose="generic", storage={"type": "local", "path": "/tmp/files"}
    )
    result._id = "original-id"
    return result


async def test_conflicts_are_bounded():
    store = AsyncMock(spec=EntityClient)
    original = _fileset()
    store.update.side_effect = EntityConflictError("conflict")
    store.get_by_id.return_value = original
    with pytest.raises(HTTPException) as caught:
        await advance_fileset_revision(original, store)
    assert caught.value.status_code == 503
    assert store.update.await_count == 3
    assert store.get_by_id.await_count == 2
    store.get_by_id.assert_awaited_with(Fileset, original.id)
    assert len({call.args[0].revision for call in store.update.await_args_list}) == 1


@pytest.mark.parametrize("error, status", [(EntityNotFoundError("gone"), 404), (EntityStoreError("unavailable"), 503)])
async def test_persistence_errors_are_not_retried(error: EntityStoreError, status: int):
    store = AsyncMock(spec=EntityClient)
    store.update.side_effect = error
    with pytest.raises(HTTPException) as caught:
        await advance_fileset_revision(_fileset(), store)
    assert caught.value.status_code == status
    store.update.assert_awaited_once()
    store.get_by_id.assert_not_awaited()


async def test_refetch_cannot_move_the_operation_to_another_workspace():
    store = AsyncMock(spec=EntityClient)
    store.update.side_effect = EntityConflictError("conflict")
    moved = _fileset()
    moved.workspace = "other"
    store.get_by_id.return_value = moved
    with pytest.raises(HTTPException) as caught:
        await advance_fileset_revision(_fileset(), store)
    assert caught.value.status_code == 404
    store.update.assert_awaited_once()


async def test_uncertain_transport_failure_is_not_retried():
    store = AsyncMock(spec=EntityClient)
    store.update.side_effect = NemoTransportError(httpx.ReadError("response lost"))
    with pytest.raises(HTTPException) as caught:
        await advance_fileset_revision(_fileset(), store)
    assert caught.value.status_code == 503
    store.update.assert_awaited_once()
    store.get_by_id.assert_not_awaited()
