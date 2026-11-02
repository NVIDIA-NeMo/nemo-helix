# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model provider can be marked as a router that calls models back through the gateway as its caller."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.core.models.api.service.model_provider_service import ModelProviderService
from nhx.core.models.entities import ModelProvider as ModelProviderEntity
from nhx.core.models.schemas import CreateModelProviderRequest, UpsertModelProviderRequest


@pytest.mark.parametrize("request_type", [CreateModelProviderRequest, UpsertModelProviderRequest])
def test_a_router_keeps_the_stored_key_it_calls_back_with(request_type: type) -> None:
    fields = {"host_url": "http://switchyard:4002", "calls_back_as_caller": True, "api_key_secret_name": "router-key"}
    if request_type is CreateModelProviderRequest:
        fields["name"] = "switchyard"

    request = request_type(**fields)

    assert request.calls_back_as_caller is True
    assert request.api_key_secret_name == "router-key"


@pytest.mark.asyncio
async def test_the_setting_is_stored_and_returned() -> None:
    entity_client = AsyncMock(spec=EntityClient)
    entity_client.get.side_effect = EntityNotFoundError("not found")

    async def created(entity: ModelProviderEntity) -> ModelProviderEntity:
        entity._id = "provider-1"
        entity._created_at = entity._updated_at = datetime.now(timezone.utc)
        return entity

    entity_client.create.side_effect = created

    provider = await ModelProviderService(entity_client).create_model_provider(
        CreateModelProviderRequest(name="switchyard", host_url="http://switchyard:4002", calls_back_as_caller=True),
        "default",
    )

    assert entity_client.create.await_args.args[0].calls_back_as_caller is True
    assert provider.calls_back_as_caller is True
