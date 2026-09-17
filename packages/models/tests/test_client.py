# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from models import parse_workspace_name_ref
from nemo_platform import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncNeMoPlatform,
    NeMoPlatform,
    NotFoundError,
)
from nemo_platform.types.inference import ModelDeployment, ModelProvider
from nemo_platform.types.inference.gateway.openai.v1 import OpenAIModelResp
from nemo_platform.types.models import ModelEntity

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sdk():
    """Create a real NeMoPlatform SDK instance for testing."""
    return NeMoPlatform(base_url="https://nmp.example.com/")


@pytest.fixture
def sdk_with_workspace():
    """Create SDK with client-level workspace set."""
    return NeMoPlatform(base_url="https://nmp.example.com/", workspace="client-ws")


@pytest.fixture
def sdk_no_trailing_slash():
    """Create SDK with base_url without trailing slash."""
    return NeMoPlatform(base_url="https://nmp.example.com")


@pytest.fixture
def async_sdk():
    """Create a real AsyncNeMoPlatform SDK instance for testing."""
    return AsyncNeMoPlatform(base_url="https://nmp.example.com/")


@pytest.fixture
def async_sdk_with_workspace():
    """Create async SDK with client-level workspace set."""
    return AsyncNeMoPlatform(base_url="https://nmp.example.com/", workspace="client-ws")


def _not_found_error() -> NotFoundError:
    request = httpx.Request("GET", "https://nmp.example.com/apis/inference-gateway/openai/v1/models/model-a")
    response = httpx.Response(404, request=request)
    return NotFoundError("not found", response=response, body={"detail": "not found"})


def _api_status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("GET", "https://nmp.example.com/apis/inference-gateway/openai/v1/models/model-a")
    response = httpx.Response(status_code, request=request)
    return APIStatusError("gateway not ready", response=response, body={"detail": "gateway not ready"})


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleep(seconds)


# ============================================================================
# ModelsResource Tests
# ============================================================================


# Tests for _get_base_url_str


def test_get_base_url_str_removes_trailing_slash(sdk):
    """Test that trailing slash is removed from base URL."""
    result = sdk.models._get_base_url_str()

    assert result == "https://nmp.example.com"
    assert not result.endswith("/")


def test_get_base_url_str_handles_no_trailing_slash(sdk_no_trailing_slash):
    """Test that URLs without trailing slash are unchanged."""
    result = sdk_no_trailing_slash.models._get_base_url_str()

    assert result == "https://nmp.example.com"


# Tests for get_openai_route_base_url


def test_get_openai_route_base_url_explicit_workspace(sdk):
    """Test URL generation with explicit workspace."""
    result = sdk.models.get_openai_route_base_url(workspace="default")

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/openai/-/v1"


def test_get_openai_route_base_url_custom_workspace(sdk):
    """Test URL generation with custom workspace name."""
    result = sdk.models.get_openai_route_base_url(workspace="my-workspace")

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1"


def test_get_openai_route_base_url_with_trailing_slash(sdk):
    """Test that trailing slash in base_url is handled correctly."""
    result = sdk.models.get_openai_route_base_url(workspace="default")

    assert "//v2" not in result
    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/openai/-/v1"


def test_get_openai_route_base_url_without_trailing_slash(sdk_no_trailing_slash):
    """Test URL generation when base_url has no trailing slash."""
    result = sdk_no_trailing_slash.models.get_openai_route_base_url(workspace="default")

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/openai/-/v1"


# Tests for get_provider_route_openai_url


def test_get_provider_route_openai_url_appends_v1(sdk):
    """Test that /v1 is appended when provider host_url doesn't end with /v1."""
    provider = MagicMock(spec=ModelProvider)
    provider.workspace = "default"
    provider.name = "openai-provider"
    provider.host_url = "https://api.openai.com"

    result = sdk.models.get_provider_route_openai_url(provider)

    assert (
        result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/provider/openai-provider/-/v1"
    )


def test_get_provider_route_openai_url_no_v1_when_host_ends_with_v1(sdk):
    """Test that /v1 is NOT appended when provider host_url already ends with /v1."""
    provider = MagicMock(spec=ModelProvider)
    provider.workspace = "default"
    provider.name = "nim-provider"
    provider.host_url = "https://nim.example.com/v1"

    result = sdk.models.get_provider_route_openai_url(provider)

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/provider/nim-provider/-"


def test_get_provider_route_openai_url_no_v1_when_host_ends_with_v1_slash(sdk):
    """Test that /v1 is NOT appended when provider host_url ends with /v1/."""
    provider = MagicMock(spec=ModelProvider)
    provider.workspace = "default"
    provider.name = "nim-provider"
    provider.host_url = "https://nim.example.com/v1/"

    result = sdk.models.get_provider_route_openai_url(provider)

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/provider/nim-provider/-"


def test_get_provider_route_openai_url_custom_workspace(sdk):
    """Test provider URL generation with custom workspace."""
    provider = MagicMock(spec=ModelProvider)
    provider.workspace = "production"
    provider.name = "my-provider"
    provider.host_url = "https://api.example.com"

    result = sdk.models.get_provider_route_openai_url(provider)

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/production/provider/my-provider/-/v1"


# Tests for get_model_entity_route_openai_url


def test_get_model_entity_route_openai_url_default_workspace(sdk):
    """Test URL generation for model entity."""
    model_entity = MagicMock(spec=ModelEntity)
    model_entity.workspace = "default"
    model_entity.name = "llama3-70b-instruct"

    result = sdk.models.get_model_entity_route_openai_url(model_entity)

    assert (
        result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/model/llama3-70b-instruct/-/v1"
    )


def test_get_model_entity_route_openai_url_custom_workspace(sdk):
    """Test URL generation for model entity with custom workspace."""
    model_entity = MagicMock(spec=ModelEntity)
    model_entity.workspace = "ml-team"
    model_entity.name = "custom-model"

    result = sdk.models.get_model_entity_route_openai_url(model_entity)

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/ml-team/model/custom-model/-/v1"


def test_get_model_entity_route_openai_url_always_appends_v1(sdk):
    """Test that /v1 is always appended for model entity routes."""
    model_entity = MagicMock(spec=ModelEntity)
    model_entity.workspace = "default"
    model_entity.name = "test-model"

    result = sdk.models.get_model_entity_route_openai_url(model_entity)

    assert result.endswith("/v1")


def test_resolve_model_reference_fetches_model_and_provider(sdk):
    """Test model reference resolution returns model route details."""
    models = sdk.models
    model_entity = MagicMock(spec=ModelEntity)
    model_entity.workspace = "default"
    model_entity.name = "judge"
    model_entity.model_providers = ["default/provider"]
    provider = MagicMock(spec=ModelProvider)
    provider.workspace = "default"
    provider.name = "provider"
    provider.host_url = "http://nim.example.test:8000"
    provider.served_models = [MagicMock(model_entity_id="default/judge", served_model_name="publisher/upstream-judge")]

    with (
        patch.object(models, "retrieve", return_value=model_entity) as mock_retrieve,
        patch.object(sdk.inference.providers, "retrieve", return_value=provider) as mock_provider_retrieve,
    ):
        result = models.resolve_model_reference("default/judge")

    mock_retrieve.assert_called_once_with("judge", workspace="default")
    mock_provider_retrieve.assert_called_once_with("provider", workspace="default")
    assert result.name == "judge"
    assert result.url == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1"
    assert result.host_url == "http://nim.example.test:8000"
    assert result.served_model_name == "publisher/upstream-judge"


def test_resolve_model_reference_ignores_missing_provider(sdk):
    """Provider lookup failures do not prevent model route resolution."""
    models = sdk.models
    model_entity = MagicMock(spec=ModelEntity)
    model_entity.workspace = "default"
    model_entity.name = "judge"
    model_entity.model_providers = ["default/missing-provider"]

    with (
        patch.object(models, "retrieve", return_value=model_entity),
        patch.object(sdk.inference.providers, "retrieve", side_effect=_not_found_error()),
    ):
        result = models.resolve_model_reference("default/judge")

    assert result.name == "judge"
    assert result.host_url is None


def test_parse_workspace_name_ref_rejects_extra_separator():
    with pytest.raises(ValueError, match="ModelRef must be in format 'workspace/model_name'"):
        parse_workspace_name_ref("default/judge/extra", label="ModelRef", expected_format="workspace/model_name")


# Tests for get_provider_route_openai_url_for_deployment


def test_get_provider_route_openai_url_for_deployment_fetches_provider(sdk):
    """Test that provider is fetched and URL is generated correctly."""
    mock_provider = MagicMock(spec=ModelProvider)
    mock_provider.workspace = "default"
    mock_provider.name = "my-provider"
    mock_provider.host_url = "https://api.example.com"

    deployment = MagicMock(spec=ModelDeployment)
    deployment.name = "my-deployment"
    deployment.model_provider_id = "default/my-provider"

    # Mock the provider retrieval
    with patch.object(sdk.inference.providers, "retrieve", return_value=mock_provider) as mock_retrieve:
        result = sdk.models.get_provider_route_openai_url_for_deployment(deployment)

        mock_retrieve.assert_called_once_with("my-provider", workspace="default")
        assert (
            result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/provider/my-provider/-/v1"
        )


def test_get_provider_route_openai_url_for_deployment_respects_v1_suffix(sdk):
    """Test that /v1 is not appended when provider host_url ends with /v1."""
    mock_provider = MagicMock(spec=ModelProvider)
    mock_provider.workspace = "production"
    mock_provider.name = "nim-provider"
    mock_provider.host_url = "https://nim.example.com/v1"

    deployment = MagicMock(spec=ModelDeployment)
    deployment.name = "nim-deployment"
    deployment.model_provider_id = "production/nim-provider"

    with patch.object(sdk.inference.providers, "retrieve", return_value=mock_provider):
        result = sdk.models.get_provider_route_openai_url_for_deployment(deployment)

        assert (
            result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/production/provider/nim-provider/-"
        )


def test_get_provider_route_openai_url_for_deployment_raises_when_no_provider_id(sdk):
    """Test that ValueError is raised when deployment has no model_provider_id."""
    deployment = MagicMock(spec=ModelDeployment)
    deployment.name = "orphan-deployment"
    deployment.model_provider_id = None

    with pytest.raises(ValueError) as exc_info:
        sdk.models.get_provider_route_openai_url_for_deployment(deployment)

    assert "orphan-deployment" in str(exc_info.value)
    assert "no associated model_provider_id" in str(exc_info.value)


def test_get_provider_route_openai_url_for_deployment_raises_when_empty_provider_id(sdk):
    """Test that ValueError is raised when deployment has empty model_provider_id."""
    deployment = MagicMock(spec=ModelDeployment)
    deployment.name = "empty-provider-deployment"
    deployment.model_provider_id = ""

    with pytest.raises(ValueError) as exc_info:
        sdk.models.get_provider_route_openai_url_for_deployment(deployment)

    assert "empty-provider-deployment" in str(exc_info.value)


# Tests for get_openai_client


def test_get_openai_client_returns_configured_client(sdk):
    """Test that get_openai_client returns an OpenAI client with correct base_url."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()

        sdk.models.get_openai_client(workspace="default")
        expected_headers = sdk.models.get_client_default_headers()

        mock_openai_cls.assert_called_once_with(
            base_url="https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            api_key="not-needed",
            default_headers=expected_headers,
        )


def test_get_openai_client_custom_workspace(sdk):
    """Test get_openai_client with custom workspace."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()

        sdk.models.get_openai_client(workspace="production")
        expected_headers = sdk.models.get_client_default_headers()

        mock_openai_cls.assert_called_once_with(
            base_url="https://nmp.example.com/apis/inference-gateway/v2/workspaces/production/openai/-/v1",
            api_key="not-needed",
            default_headers=expected_headers,
        )


def test_get_openai_client_includes_auth_headers():
    """Test that auth headers from the SDK are propagated to OpenAI client."""
    sdk_with_auth = NeMoPlatform(
        base_url="https://nmp.example.com/",
        default_headers={"Authorization": "Bearer token-123", "X-NMP-Principal-Id": "user@example.com"},
    )

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()

        sdk_with_auth.models.get_openai_client(workspace="default")
        default_headers = mock_openai_cls.call_args.kwargs["default_headers"]

        assert default_headers["Authorization"] == "Bearer token-123"
        assert default_headers["X-NMP-Principal-Id"] == "user@example.com"


# Tests for wait_for_openai_model


def test_wait_for_openai_model_uses_client_workspace_and_retries(sdk_with_workspace):
    """Test that OpenAI model route readiness retries until the model is visible."""
    model_response = OpenAIModelResp(id="client-ws/model-a", owned_by="test")

    with patch.object(
        sdk_with_workspace.inference.gateway.openai.v1.models,
        "get",
        side_effect=[_not_found_error(), model_response],
    ) as mock_get:
        result = sdk_with_workspace.models.wait_for_openai_model("model-a", timeout=1, poll_interval=0)

    assert result == model_response
    assert [call.kwargs for call in mock_get.call_args_list] == [
        {"name": "model-a", "workspace": "client-ws"},
        {"name": "model-a", "workspace": "client-ws"},
    ]


def test_wait_for_openai_model_accepts_qualified_name(sdk):
    """Test that workspace-qualified names are normalized for the route path."""
    model_response = OpenAIModelResp(id="ws/model-a", owned_by="test")

    with patch.object(
        sdk.inference.gateway.openai.v1.models,
        "get",
        return_value=model_response,
    ) as mock_get:
        result = sdk.models.wait_for_openai_model("ws/model-a", workspace="ws", timeout=1, poll_interval=0)

    assert result == model_response
    mock_get.assert_called_once_with(name="model-a", workspace="ws")


def test_wait_for_openai_model_retries_unexpected_model_id(sdk):
    """Test that a stale or mismatched route response is not treated as ready."""
    stale_response = OpenAIModelResp(id="ws/other-model", owned_by="test")
    model_response = OpenAIModelResp(id="ws/model-a", owned_by="test")

    with patch.object(
        sdk.inference.gateway.openai.v1.models,
        "get",
        side_effect=[stale_response, model_response],
    ) as mock_get:
        result = sdk.models.wait_for_openai_model("model-a", workspace="ws", timeout=1, poll_interval=0)

    assert result == model_response
    assert mock_get.call_count == 2


@pytest.mark.parametrize("status_code", [429, 502, 503, 504])
def test_wait_for_openai_model_retries_transient_gateway_errors(sdk, status_code):
    """Test that transient gateway and connection failures keep polling."""
    model_response = OpenAIModelResp(id="ws/model-a", owned_by="test")
    request = httpx.Request("GET", "https://nmp.example.com/apis/inference-gateway/openai/v1/models/model-a")

    with patch.object(
        sdk.inference.gateway.openai.v1.models,
        "get",
        side_effect=[
            APIConnectionError(request=request),
            _api_status_error(status_code),
            model_response,
        ],
    ) as mock_get:
        result = sdk.models.wait_for_openai_model("model-a", workspace="ws", timeout=1, poll_interval=0)

    assert result == model_response
    assert mock_get.call_count == 3


def test_wait_for_openai_model_reraises_non_transient_status_error(sdk):
    """Test that non-readiness HTTP errors are not hidden until timeout."""
    with patch.object(
        sdk.inference.gateway.openai.v1.models,
        "get",
        side_effect=_api_status_error(401),
    ):
        with pytest.raises(APIStatusError):
            sdk.models.wait_for_openai_model("model-a", workspace="ws", timeout=1, poll_interval=0)


def test_wait_for_openai_model_times_out(sdk):
    """Test that timeout includes the expected model id in the error."""
    with patch.object(
        sdk.inference.gateway.openai.v1.models,
        "get",
        side_effect=_not_found_error(),
    ):
        with pytest.raises(TimeoutError, match="OpenAI model ws/model-a not available"):
            sdk.models.wait_for_openai_model("model-a", workspace="ws", timeout=0.01, poll_interval=0)


def test_wait_for_openai_model_bounds_sleep_to_remaining_timeout(sdk):
    """Test that sync polling does not sleep beyond the configured timeout."""
    clock = _FakeClock()

    with (
        patch.object(
            sdk.inference.gateway.openai.v1.models,
            "get",
            side_effect=_not_found_error(),
        ),
        patch("models.resources.time.time", side_effect=clock.time),
        patch("models.resources.time.sleep", side_effect=clock.sleep),
    ):
        with pytest.raises(TimeoutError, match="OpenAI model ws/model-a not available"):
            sdk.models.wait_for_openai_model("model-a", workspace="ws", timeout=1.25, poll_interval=5)

    assert clock.sleep_calls == [1.25]


# ============================================================================
# AsyncModelsResource Tests
# ============================================================================


# Tests for AsyncModelsResource URL builders (sync methods, no I/O)


def test_async_get_base_url_str_removes_trailing_slash(async_sdk):
    """Test that trailing slash is removed from base URL (async resource)."""
    result = async_sdk.models._get_base_url_str()

    assert result == "https://nmp.example.com"


def test_async_get_openai_route_base_url(async_sdk):
    """Test URL generation with async resource (sync method, no I/O)."""
    result = async_sdk.models.get_openai_route_base_url(workspace="default")

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/openai/-/v1"


def test_async_get_provider_route_openai_url(async_sdk):
    """Test provider URL generation with async resource (sync method, no I/O)."""
    provider = MagicMock(spec=ModelProvider)
    provider.workspace = "default"
    provider.name = "my-provider"
    provider.host_url = "https://api.example.com"

    result = async_sdk.models.get_provider_route_openai_url(provider)

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/provider/my-provider/-/v1"


def test_async_get_model_entity_route_openai_url(async_sdk):
    """Test model entity URL generation with async resource (sync method, no I/O)."""
    model_entity = MagicMock(spec=ModelEntity)
    model_entity.workspace = "default"
    model_entity.name = "my-model"

    result = async_sdk.models.get_model_entity_route_openai_url(model_entity)

    assert result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/model/my-model/-/v1"


@pytest.mark.asyncio
async def test_async_resolve_model_reference_fetches_model_and_provider(async_sdk):
    """Test async model reference resolution returns model route details."""
    models = async_sdk.models
    model_entity = MagicMock(spec=ModelEntity)
    model_entity.workspace = "default"
    model_entity.name = "judge"
    model_entity.model_providers = ["default/provider"]
    provider = MagicMock(spec=ModelProvider)
    provider.workspace = "default"
    provider.name = "provider"
    provider.host_url = "http://nim.example.test:8000"
    mock_retrieve = AsyncMock(return_value=model_entity)
    mock_provider_retrieve = AsyncMock(return_value=provider)

    with (
        patch.object(models, "retrieve", mock_retrieve),
        patch.object(async_sdk.inference.providers, "retrieve", mock_provider_retrieve),
    ):
        result = await models.resolve_model_reference("default/judge")

    mock_retrieve.assert_awaited_once_with("judge", workspace="default")
    mock_provider_retrieve.assert_awaited_once_with("provider", workspace="default")
    assert result.name == "judge"
    assert result.url == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/model/judge/-/v1"
    assert result.host_url == "http://nim.example.test:8000"


# Tests for get_async_openai_client


def test_get_async_openai_client_returns_async_client(async_sdk):
    """Test that get_async_openai_client returns an AsyncOpenAI client."""
    with patch("openai.AsyncOpenAI") as mock_async_openai_cls:
        mock_async_openai_cls.return_value = MagicMock()

        async_sdk.models.get_async_openai_client(workspace="default")
        expected_headers = async_sdk.models.get_client_default_headers()

        mock_async_openai_cls.assert_called_once_with(
            base_url="https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/openai/-/v1",
            api_key="not-needed",
            default_headers=expected_headers,
        )


def test_get_async_openai_client_custom_workspace(async_sdk):
    """Test get_async_openai_client with custom workspace."""
    with patch("openai.AsyncOpenAI") as mock_async_openai_cls:
        mock_async_openai_cls.return_value = MagicMock()

        async_sdk.models.get_async_openai_client(workspace="production")
        expected_headers = async_sdk.models.get_client_default_headers()

        mock_async_openai_cls.assert_called_once_with(
            base_url="https://nmp.example.com/apis/inference-gateway/v2/workspaces/production/openai/-/v1",
            api_key="not-needed",
            default_headers=expected_headers,
        )


def test_get_async_openai_client_includes_auth_headers():
    """Test that auth headers from the async SDK are propagated to AsyncOpenAI client."""
    sdk_with_auth = AsyncNeMoPlatform(
        base_url="https://nmp.example.com/",
        default_headers={"Authorization": "Bearer token-abc", "X-NMP-Principal-Id": "async-user@example.com"},
    )

    with patch("openai.AsyncOpenAI") as mock_async_openai_cls:
        mock_async_openai_cls.return_value = MagicMock()

        sdk_with_auth.models.get_async_openai_client(workspace="default")
        default_headers = mock_async_openai_cls.call_args.kwargs["default_headers"]

        assert default_headers["Authorization"] == "Bearer token-abc"
        assert default_headers["X-NMP-Principal-Id"] == "async-user@example.com"


# Tests for async wait_for_openai_model


@pytest.mark.asyncio
async def test_async_wait_for_openai_model_retries_until_visible(async_sdk_with_workspace):
    """Test async OpenAI model route readiness polling."""
    model_response = OpenAIModelResp(id="client-ws/model-a", owned_by="test")
    mock_get = AsyncMock(side_effect=[_not_found_error(), model_response])

    with patch.object(async_sdk_with_workspace.inference.gateway.openai.v1.models, "get", mock_get):
        result = await async_sdk_with_workspace.models.wait_for_openai_model("model-a", timeout=1, poll_interval=0)

    assert result == model_response
    assert [call.kwargs for call in mock_get.call_args_list] == [
        {"name": "model-a", "workspace": "client-ws"},
        {"name": "model-a", "workspace": "client-ws"},
    ]


@pytest.mark.asyncio
async def test_async_wait_for_openai_model_accepts_qualified_name_and_retries(async_sdk):
    """Test that async workspace-qualified names are normalized while polling."""
    model_response = OpenAIModelResp(id="ws/model-a", owned_by="test")
    mock_get = AsyncMock(side_effect=[_not_found_error(), model_response])

    with patch.object(async_sdk.inference.gateway.openai.v1.models, "get", mock_get):
        result = await async_sdk.models.wait_for_openai_model("ws/model-a", workspace="ws", timeout=1, poll_interval=0)

    assert result == model_response
    assert [call.kwargs for call in mock_get.call_args_list] == [
        {"name": "model-a", "workspace": "ws"},
        {"name": "model-a", "workspace": "ws"},
    ]


@pytest.mark.asyncio
async def test_async_wait_for_openai_model_retries_unexpected_model_id(async_sdk):
    """Test that an async stale or mismatched route response is not treated as ready."""
    stale_response = OpenAIModelResp(id="ws/other-model", owned_by="test")
    model_response = OpenAIModelResp(id="ws/model-a", owned_by="test")
    mock_get = AsyncMock(side_effect=[stale_response, model_response])

    with patch.object(async_sdk.inference.gateway.openai.v1.models, "get", mock_get):
        result = await async_sdk.models.wait_for_openai_model("model-a", workspace="ws", timeout=1, poll_interval=0)

    assert result == model_response
    assert mock_get.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [429, 502, 503, 504])
async def test_async_wait_for_openai_model_retries_transient_gateway_errors(async_sdk, status_code):
    """Test that async polling retries transient gateway and timeout failures."""
    model_response = OpenAIModelResp(id="ws/model-a", owned_by="test")
    request = httpx.Request("GET", "https://nmp.example.com/apis/inference-gateway/openai/v1/models/model-a")
    mock_get = AsyncMock(
        side_effect=[
            APITimeoutError(request=request),
            _api_status_error(status_code),
            model_response,
        ]
    )

    with patch.object(async_sdk.inference.gateway.openai.v1.models, "get", mock_get):
        result = await async_sdk.models.wait_for_openai_model("model-a", workspace="ws", timeout=1, poll_interval=0)

    assert result == model_response
    assert mock_get.await_count == 3


@pytest.mark.asyncio
async def test_async_wait_for_openai_model_bounds_sleep_to_remaining_timeout(async_sdk):
    """Test that async polling does not sleep beyond the configured timeout."""
    clock = _FakeClock()
    mock_get = AsyncMock(side_effect=_not_found_error())

    with (
        patch.object(async_sdk.inference.gateway.openai.v1.models, "get", mock_get),
        patch("models.resources.time.time", side_effect=clock.time),
        patch("models.resources.asyncio.sleep", side_effect=clock.async_sleep),
    ):
        with pytest.raises(TimeoutError, match="OpenAI model ws/model-a not available"):
            await async_sdk.models.wait_for_openai_model("model-a", workspace="ws", timeout=1.25, poll_interval=5)

    assert clock.sleep_calls == [1.25]


# Tests for async get_provider_route_openai_url_for_deployment


@pytest.mark.asyncio
async def test_async_get_provider_route_openai_url_for_deployment(async_sdk):
    """Test async version fetches provider and generates URL correctly."""
    mock_provider = MagicMock(spec=ModelProvider)
    mock_provider.workspace = "default"
    mock_provider.name = "my-provider"
    mock_provider.host_url = "https://api.example.com"

    deployment = MagicMock(spec=ModelDeployment)
    deployment.name = "my-deployment"
    deployment.model_provider_id = "default/my-provider"

    with patch.object(
        async_sdk.inference.providers, "retrieve", AsyncMock(return_value=mock_provider)
    ) as mock_retrieve:
        result = await async_sdk.models.get_provider_route_openai_url_for_deployment(deployment)

        mock_retrieve.assert_called_once_with("my-provider", workspace="default")
        assert (
            result == "https://nmp.example.com/apis/inference-gateway/v2/workspaces/default/provider/my-provider/-/v1"
        )


@pytest.mark.asyncio
async def test_async_get_provider_route_openai_url_for_deployment_raises_when_no_provider_id(async_sdk):
    """Test that ValueError is raised when deployment has no model_provider_id (async)."""
    deployment = MagicMock(spec=ModelDeployment)
    deployment.name = "orphan-deployment"
    deployment.model_provider_id = None

    with pytest.raises(ValueError) as exc_info:
        await async_sdk.models.get_provider_route_openai_url_for_deployment(deployment)

    assert "orphan-deployment" in str(exc_info.value)
    assert "no associated model_provider_id" in str(exc_info.value)


# ============================================================================
# Workspace Resolution Tests (client-level fallback)
# ============================================================================


# Tests for get_openai_route_base_url workspace resolution


def test_get_openai_route_base_url_uses_client_workspace(sdk_with_workspace):
    """Test that client-level workspace is used when not explicitly provided."""
    result = sdk_with_workspace.models.get_openai_route_base_url()

    assert "/workspaces/client-ws/" in result


def test_get_openai_route_base_url_explicit_overrides_client(sdk_with_workspace):
    """Test that explicit workspace overrides client-level workspace."""
    result = sdk_with_workspace.models.get_openai_route_base_url(workspace="override")

    assert "/workspaces/override/" in result
    assert "/workspaces/client-ws/" not in result


def test_get_openai_route_base_url_raises_without_workspace(sdk):
    """Test that ValueError is raised when no workspace is available."""
    with pytest.raises(ValueError, match="Missing workspace"):
        sdk.models.get_openai_route_base_url()


# Tests for get_openai_client workspace resolution


def test_get_openai_client_uses_client_workspace(sdk_with_workspace):
    """Test that client-level workspace is used when not explicitly provided."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()

        sdk_with_workspace.models.get_openai_client()

        call_args = mock_openai_cls.call_args
        assert "/workspaces/client-ws/" in call_args.kwargs["base_url"]


def test_get_openai_client_explicit_overrides_client(sdk_with_workspace):
    """Test that explicit workspace overrides client-level workspace."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()

        sdk_with_workspace.models.get_openai_client(workspace="override")

        call_args = mock_openai_cls.call_args
        assert "/workspaces/override/" in call_args.kwargs["base_url"]


def test_get_openai_client_raises_without_workspace(sdk):
    """Test that ValueError is raised when no workspace is available."""
    with pytest.raises(ValueError, match="Missing workspace"):
        sdk.models.get_openai_client()


# Tests for async get_openai_route_base_url workspace resolution


def test_async_get_openai_route_base_url_uses_client_workspace(async_sdk_with_workspace):
    """Test that client-level workspace is used when not explicitly provided."""
    result = async_sdk_with_workspace.models.get_openai_route_base_url()

    assert "/workspaces/client-ws/" in result


def test_async_get_openai_route_base_url_explicit_overrides_client(async_sdk_with_workspace):
    """Test that explicit workspace overrides client-level workspace."""
    result = async_sdk_with_workspace.models.get_openai_route_base_url(workspace="override")

    assert "/workspaces/override/" in result


def test_async_get_openai_route_base_url_raises_without_workspace(async_sdk):
    """Test that ValueError is raised when no workspace is available."""
    with pytest.raises(ValueError, match="Missing workspace"):
        async_sdk.models.get_openai_route_base_url()


# Tests for get_async_openai_client workspace resolution


def test_get_async_openai_client_uses_client_workspace(async_sdk_with_workspace):
    """Test that client-level workspace is used when not explicitly provided."""
    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()

        async_sdk_with_workspace.models.get_async_openai_client()

        call_args = mock_openai_cls.call_args
        assert "/workspaces/client-ws/" in call_args.kwargs["base_url"]


def test_get_async_openai_client_explicit_overrides_client(async_sdk_with_workspace):
    """Test that explicit workspace overrides client-level workspace."""
    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()

        async_sdk_with_workspace.models.get_async_openai_client(workspace="override")

        call_args = mock_openai_cls.call_args
        assert "/workspaces/override/" in call_args.kwargs["base_url"]


def test_get_async_openai_client_raises_without_workspace(async_sdk):
    """Test that ValueError is raised when no workspace is available."""
    with pytest.raises(ValueError, match="Missing workspace"):
        async_sdk.models.get_async_openai_client()
