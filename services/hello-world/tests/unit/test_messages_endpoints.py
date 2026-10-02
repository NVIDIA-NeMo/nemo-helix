# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HelloWorld message endpoints."""

import pytest
from fastapi import HTTPException
from nemo_helix_plugin.entities.types import DeleteResponse
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.hello_world.api.v1.messages.endpoints import delete_message


class DeleteNotFoundEntityClient(EntityClient):
    def __init__(self) -> None:
        pass

    async def delete(
        self,
        entity_type: object,
        name: str,
        *,
        workspace: str | None = None,
        parent: str | None = None,
        expected_db_version: int | None = None,
    ) -> DeleteResponse:
        raise EntityNotFoundError("not found")


@pytest.mark.asyncio
async def test_delete_message_maps_delete_not_found_to_404():
    entity_store = DeleteNotFoundEntityClient()

    with pytest.raises(HTTPException) as exc_info:
        await delete_message(workspace="default", name="missing-message", entity_store=entity_store)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Message 'missing-message' not found in workspace 'default'"
