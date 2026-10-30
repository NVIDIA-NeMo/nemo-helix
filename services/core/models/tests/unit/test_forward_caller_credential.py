# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model provider can be sent each call's caller credential instead of a stored key."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from nhx.common.entities.client import EntityClient, EntityNotFoundError
from nhx.core.models.api.service.model_provider_service import ModelProviderService
from nhx.core.models.entities import ModelProvider as ModelProviderEntity
from nhx.core.models.schemas import CreateModelProviderRequest, UpsertModelProviderRequest
from pydantic import ValidationError


@pytest.mark.parametrize("request_type", [CreateModelProviderRequest, UpsertModelProviderRequest])
def test_a_provider_sent_its_callers_credential_has_no_stored_key(request_type: type) -> None:
    fields = {"host_url": "http://switchyard:4002", "forward_caller_credential": True, "api_key_secret_name": "k"}
    if request_type is CreateModelProviderRequest:
        fields["name"] = "switchyard"

    with pytest.raises(ValidationError, match="leave api_key_secret_name unset"):
        request_type(**fields)


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
        CreateModelProviderRequest(
            name="switchyard", host_url="http://switchyard:4002", forward_caller_credential=True
        ),
        "default",
    )

    assert entity_client.create.await_args.args[0].forward_caller_credential is True
    assert provider.forward_caller_credential is True
