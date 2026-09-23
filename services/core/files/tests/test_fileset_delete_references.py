# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from nemo_helix_plugin.client.errors import InternalServerError, NemoTransportError
from nhx.common.entities import EntityStoreError
from nhx.core.files.api.v2.filesets.endpoints import (
    _count_entity_fileset_references,
    _count_fileset_references,
    _ModelFilesetReference,
    delete_fileset,
)
from nhx.core.files.entities import Fileset


def _internal_server_error() -> InternalServerError:
    request = httpx.Request("GET", "http://entities.test/list")
    return InternalServerError(httpx.Response(500, request=request))


def _transport_error() -> NemoTransportError:
    return NemoTransportError(httpx.ConnectError("connection refused"))


async def test_count_fileset_references_uses_total_results() -> None:
    entity_store = MagicMock()
    entity_store.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[object()],
            pagination=SimpleNamespace(total_results=203),
        )
    )

    count = await _count_fileset_references(entity_store, _ModelFilesetReference, "default/weights")

    assert count == 203
    await_args = entity_store.list.await_args
    assert await_args is not None
    assert await_args.kwargs == {
        "workspace": "-",
        "filter_obj": {"fileset": "default/weights"},
        "page_size": 1,
    }


async def test_count_entity_fileset_references_includes_supported_ref_formats() -> None:
    with patch(
        "nhx.core.files.api.v2.filesets.endpoints._count_fileset_references",
        new=AsyncMock(side_effect=[2, 1, 3, 4]),
    ) as count_references:
        count = await _count_entity_fileset_references(
            MagicMock(),
            _ModelFilesetReference,
            "default",
            "weights",
        )

    assert count == 10
    assert [call.args[2] for call in count_references.await_args_list] == [
        "default/weights",
        "fileset://default/weights",
        "weights",
        "fileset://weights",
    ]
    assert count_references.await_args_list[2].kwargs == {"workspace": "default"}
    assert count_references.await_args_list[3].kwargs == {"workspace": "default"}


async def test_delete_fileset_rejects_references_before_deleting_storage() -> None:
    fileset = MagicMock(name="weights")
    fileset.name = "weights"
    entity_store = MagicMock()
    entity_store.delete = AsyncMock()
    entity_store.as_service.return_value = MagicMock()

    with (
        patch(
            "nhx.core.files.api.v2.filesets.endpoints.get_fileset",
            new=AsyncMock(return_value=fileset),
        ),
        patch(
            "nhx.core.files.api.v2.filesets.endpoints._count_entity_fileset_references",
            new=AsyncMock(side_effect=[1, 1]),
        ),
        patch("nhx.core.files.api.v2.filesets.endpoints.storage_impl_factory") as storage_factory,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_fileset(
                workspace="default",
                name="weights",
                entity_store=entity_store,
                sdk=MagicMock(),
                auth_client=MagicMock(),
            )

    assert exc_info.value.status_code == 409
    assert "1 model entity reference(s)" in exc_info.value.detail
    assert "1 adapter entity reference(s)" in exc_info.value.detail
    entity_store.as_service.assert_called_once_with("files", internal=True)
    storage_factory.assert_not_called()
    entity_store.delete.assert_not_awaited()


@pytest.mark.parametrize(
    "reference_results",
    [
        [EntityStoreError("model reference lookup failed")],
        [_internal_server_error()],
        [_transport_error()],
        [0, EntityStoreError("adapter reference lookup failed")],
        [0, _internal_server_error()],
        [0, _transport_error()],
    ],
)
async def test_delete_fileset_fails_closed_when_references_cannot_be_checked(
    reference_results: list[int | Exception],
) -> None:
    entity_store = MagicMock()
    entity_store.delete = AsyncMock()
    entity_store.as_service.return_value = MagicMock()

    with (
        patch(
            "nhx.core.files.api.v2.filesets.endpoints.get_fileset",
            new=AsyncMock(return_value=MagicMock(name="weights")),
        ),
        patch(
            "nhx.core.files.api.v2.filesets.endpoints._count_entity_fileset_references",
            new=AsyncMock(side_effect=reference_results),
        ),
        patch("nhx.core.files.api.v2.filesets.endpoints.storage_impl_factory") as storage_factory,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_fileset(
                workspace="default",
                name="weights",
                entity_store=entity_store,
                sdk=MagicMock(),
                auth_client=MagicMock(),
            )

    assert exc_info.value.status_code == 503
    storage_factory.assert_not_called()
    entity_store.delete.assert_not_awaited()


async def test_delete_unreferenced_fileset_deletes_storage_and_entity() -> None:
    fileset = MagicMock()
    fileset.name = "weights"
    fileset.storage = MagicMock()
    storage = SimpleNamespace(delete_all=AsyncMock())
    entity_store = MagicMock()
    entity_store.delete = AsyncMock()
    entity_store.as_service.return_value = MagicMock()
    deleted_output = object()

    with (
        patch(
            "nhx.core.files.api.v2.filesets.endpoints.get_fileset",
            new=AsyncMock(return_value=fileset),
        ),
        patch(
            "nhx.core.files.api.v2.filesets.endpoints._count_entity_fileset_references",
            new=AsyncMock(side_effect=[0, 0]),
        ),
        patch(
            "nhx.core.files.api.v2.filesets.endpoints.resolve_storage_secrets_for_user",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "nhx.core.files.api.v2.filesets.endpoints.storage_impl_factory",
            return_value=storage,
        ),
        patch(
            "nhx.core.files.api.v2.filesets.endpoints.fileset_output_from_entity",
            return_value=deleted_output,
        ),
    ):
        result = await delete_fileset(
            workspace="default",
            name="weights",
            entity_store=entity_store,
            sdk=MagicMock(),
            auth_client=MagicMock(),
        )

    assert result is deleted_output
    storage.delete_all.assert_awaited_once_with()
    entity_store.delete.assert_awaited_once_with(Fileset, "weights", workspace="default")
