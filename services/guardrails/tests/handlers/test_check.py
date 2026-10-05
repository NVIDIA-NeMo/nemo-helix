# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from nemoguardrails import LLMRails
from nhx.guardrails.app.handlers.checks import CheckRequestHandler
from nhx.guardrails.app.services.rails.service import RailsService
from nhx.guardrails.entities.enums import StatusEnum
from nhx.guardrails.entities.values._private import Model, RailsConfig
from nhx.guardrails.entities.values.check import GuardrailCheckRequest, GuardrailCheckResponse
from nhx.guardrails.entities.values.common import GuardrailsDataInput, GuardrailsDataOutput


@pytest.fixture
def mock_rails_service():
    service = RailsService(rails_registry=MagicMock(), config_registry=MagicMock())
    service.get_config = AsyncMock(return_value=AsyncMock())
    service.get_rails = AsyncMock(return_value=MagicMock())
    return service


@pytest.fixture
def mock_request():
    request = Request(scope={"type": "http", "headers": []})
    request.state.request_id = "test-request-id"
    return request


@pytest.fixture
def mock_request_body():
    return GuardrailCheckRequest(
        messages=[{"role": "user", "content": "test-message"}],
        model="test-model",
        guardrails=GuardrailsDataInput(config_ids=["test-workspace/test-config-id"], config=None),
    )


@pytest.fixture
def mock_response_model():
    return GuardrailCheckResponse


@pytest.fixture
def handler(mock_rails_service, mock_request, mock_request_body, mock_response_model):
    return CheckRequestHandler(
        rails_service=mock_rails_service,
        request=mock_request,
        request_body=mock_request_body,
        response_model=mock_response_model,
        workspace="test-workspace",
    )


@pytest.mark.asyncio
@patch("nhx.guardrails.app.handlers.checks.convert_check_request_to_guardrails")
@patch("nhx.guardrails.app.handlers.checks.create_guardrail_check_response_from_generation_response")
async def test_handle_request(
    mock_create_response,
    mock_convert_request,
    handler,
):
    mock_convert_request.return_value = MagicMock()
    guardrails_data = GuardrailsDataOutput(config_ids=[], output_data={}, log={})
    mock_create_response.return_value = GuardrailCheckResponse(
        status=StatusEnum.SUCCESS,
        rails_status={},
        guardrails_data=guardrails_data,
    )

    handler.instantiate_llm_rails = AsyncMock(return_value=MagicMock())
    handler._handle_non_streaming = AsyncMock(return_value=mock_create_response.return_value)
    handler.llm_rails = AsyncMock(spec=LLMRails)
    handler.llm_rails.config = MagicMock(
        spec=RailsConfig, models=[Model(model="test_model", type="main", engine="nim")]
    )

    response = await handler.handle_request()

    assert response is not None
    assert isinstance(response, GuardrailCheckResponse)
    assert response.status == StatusEnum.SUCCESS


def test_ensure_request_id(handler):
    handler.ensure_request_id()
    assert handler.request.state.request_id == "test-request-id"


def test_get_guardrails_config(handler):
    config_ids, config = handler.get_guardrails_config()
    assert config_ids == ["test-workspace/test-config-id"]
    assert config is None


@pytest.mark.asyncio
@patch("nhx.guardrails.app.handlers.checks.LLMRails")
@patch("nhx.guardrails.app.handlers.checks.set_main_model_into_context")
async def test_inline_config_sets_main_model_in_context(
    mock_set_main_model_into_context,
    _mock_llm_rails,
    mock_rails_service,
    mock_request,
):
    """
    Verify that inline configs set the main model into context.

    The main model is extracted at inference-time to determine the base URL for the model.
    This test ensures the model is set into context correctly when handling a check request given an inline config.
    """
    inline_config = {
        "models": [{"type": "main", "engine": "nim", "model": "default/my-model"}],
    }
    request_body = GuardrailCheckRequest(
        messages=[{"role": "user", "content": "test"}],
        model="default/my-model",
        guardrails=GuardrailsDataInput(config_ids=None, config=inline_config),
    )

    handler = CheckRequestHandler(
        rails_service=mock_rails_service,
        request=mock_request,
        request_body=request_body,
        response_model=GuardrailCheckResponse,
        workspace="test-workspace",
    )
    handler._handle_non_streaming = AsyncMock(return_value=MagicMock())

    await handler.handle_request()

    mock_set_main_model_into_context.assert_called_once()
    called_model = mock_set_main_model_into_context.call_args[0][0]
    assert called_model.type == "main"
