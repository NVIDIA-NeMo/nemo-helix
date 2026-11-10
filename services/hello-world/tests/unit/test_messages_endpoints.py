# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HelloWorld message endpoints."""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.hello_world.api.v1.messages.endpoints import delete_message
from nhx.hello_world.entities import HelloWorldMessage


@pytest.mark.asyncio
async def test_delete_message_maps_delete_not_found_to_404():
    message = HelloWorldMessage(name="missing-message", workspace="default", message="hello")
    message._id = "message-id"
    entity_store = AsyncMock(spec=EntityClient)
    entity_store.get.return_value = message
    entity_store.delete_by_id.side_effect = EntityNotFoundError("not found")

    with pytest.raises(HTTPException) as exc_info:
        await delete_message(workspace="default", name="missing-message", entity_store=entity_store)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Message 'missing-message' not found in workspace 'default'"

    entity_store.delete_by_id.assert_awaited_once_with(
        HelloWorldMessage, "message-id", expected_db_version=message.db_version
    )
