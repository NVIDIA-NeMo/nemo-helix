# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from contextvars import ContextVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from nhx.guardrails.app.handlers.utils import (
    get_merged_custom_headers_token,
    get_rail_types_from_config,
    get_rails_name_from_config,
    model_with_req_scoped_custom_headers,
    run_generate_async,
    set_main_model_merged_custom_headers_into_context,
    update_models_in_config,
)
from nhx.guardrails.entities.values._private import Model, ModelParameters, RailsConfig
from pytest_mock import MockerFixture

default_llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", "nim")


@pytest.fixture
def set_headers_into_context_mock(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("nhx.guardrails.app.handlers.utils.set_request_default_headers_into_context")


@pytest.fixture
def config():
    return RailsConfig(
        models=[
            Model(type="type1", model="model1", engine="engine1").model_dump(),
            Model(type="type2", model="model2", engine="engine2").model_dump(),
        ]
    )


class TestUpdateModelsInConfig:
    def test_update_existing_model(self, config):
        new_model = Model(type="type1", model="new_model1", engine="new_engine1")
        updated_config = update_models_in_config(config, new_model)
        assert len(updated_config.models) == 2
        assert updated_config.models[0].model == "new_model1"
        assert updated_config.models[0].engine == "new_engine1"
        assert updated_config.models[0].parameters == {"default_headers": {}}

    def test_add_new_model(self, config):
        new_model = Model(type="type3", model="model3", engine="engine3")
        updated_config = update_models_in_config(config, new_model)
        assert len(updated_config.models) == 3
        assert updated_config.models[2].model == "model3"
        assert updated_config.models[2].parameters == {}
        assert updated_config.models[2].engine == "engine3"

    def test_add_main_model_when_config_has_none(self, mocker: MockerFixture):
        """Test that a main model can be dynamically added to a config that has no main model."""
        # Config without a main model
        config_without_main = RailsConfig(
            models=[
                Model(type="content_safety", model="nvidia/nemoguard", engine="nim").model_dump(),
            ]
        )
        mocker.patch(
            "nhx.guardrails.app.handlers.utils.get_request_default_headers_from_context",
            return_value={},
        )

        # Main model that would be populated from a request at runtime
        main_model = Model(type="main", model="meta/llama-3.1-8b-instruct", engine="nim")
        updated_config = update_models_in_config(config_without_main, main_model)

        # Verify config has 2 models: the original content_safety and the new main model
        assert len(updated_config.models) == 2
        main_models = [m for m in updated_config.models if m.type == "main"]
        assert len(main_models) == 1
        assert main_models[0].model == "meta/llama-3.1-8b-instruct"
        assert main_models[0].engine == "nim"

    def test_merge_main_model_parameters_from_config_and_request(self, mocker: MockerFixture):
        """Test that a config's main model parameters are preserved and merged with the model from incoming request."""
        # Config has a main model with parameters but no model name
        config_with_main_params = RailsConfig(
            models=[
                Model(
                    type="main",
                    engine="nim",
                    parameters={
                        "base_url": "https://mock-nim/v1",
                        "temperature": 0.7,
                    },
                ).model_dump(),
            ]
        )
        mocker.patch(
            "nhx.guardrails.app.handlers.utils.get_request_default_headers_from_context",
            return_value={},
        )

        # Main model that would be populated from a request at runtime
        request_model = Model(type="main", model="meta/llama-3.1-8b-instruct", engine="nim")
        updated_config = update_models_in_config(config_with_main_params, request_model)

        # Should still have 1 main model
        assert len(updated_config.models) == 1
        main_model = updated_config.models[0]

        # Model name should come from request
        assert main_model.model == "meta/llama-3.1-8b-instruct"

        # Parameters from config should be preserved
        assert main_model.parameters["base_url"] == "https://mock-nim/v1"
        assert main_model.parameters["temperature"] == 0.7

    def test_updated_models_keep_model_parameters_type(self, mocker: MockerFixture):
        """`model_copy` skips validation, so parameters must not be left as plain dicts."""
        config_with_base_url = RailsConfig(
            models=[
                Model(type="main", engine="nim", parameters={"base_url": "https://mock-nim/v1"}).model_dump(),
                Model(
                    type="content_safety",
                    engine="nim",
                    model="nvidia/nemoguard",
                    parameters={"base_url": "https://mock-nim/v1"},
                ).model_dump(),
            ]
        )
        mocker.patch(
            "nhx.guardrails.app.handlers.utils.get_request_default_headers_from_context",
            return_value={"x-custom": "value"},
        )

        updated_config = update_models_in_config(
            config_with_base_url, Model(type="main", model="meta/llama-3.1-8b-instruct", engine="nim")
        )
        added_config = update_models_in_config(config_with_base_url, Model(type="new", model="model3", engine="nim"))

        for model in [*updated_config.models, *added_config.models]:
            assert isinstance(model.parameters, ModelParameters)
        assert updated_config.models[0].parameters.base_url == "https://mock-nim/v1"
        assert updated_config.models[1].parameters.default_headers == {}

    def test_request_headers_are_not_built_into_models(self, config: RailsConfig, mocker: MockerFixture):
        """Only the configured headers (lower-cased) go into the models, not the request's headers.

        The engine is shared across requests, so building in something like a proxy's unique
        ``x-request-id`` would be wrong for every other request.
        """
        mocker.patch(
            "nhx.guardrails.app.handlers.utils.get_request_default_headers_from_context",
            return_value={"x-request-id": "unique-per-request", "x-custom-header1": "from-request"},
        )
        config.models[1].parameters = {"default_headers": {"X-Custom-Header1": "configured"}}

        updated_config = update_models_in_config(config, Model(type="type3", model="model3", engine="engine3"))

        assert updated_config.models[0].parameters["default_headers"] == {}
        assert updated_config.models[1].parameters["default_headers"] == {"x-custom-header1": "configured"}


def test_get_rail_types_from_config():
    config = MagicMock()
    config.rails.input.flows = ["flow1"]
    config.rails.output.flows = ["flow2"]
    rail_types = get_rail_types_from_config(config)
    assert rail_types == {"rails": ["input", "output"]}


def test_get_rails_name_from_config():
    config = MagicMock()
    config.rails.input.flows = ["flow1"]
    config.rails.output.flows = ["flow2"]
    rail_names = get_rails_name_from_config(config)
    assert rail_names == ["flow1", "flow2"]


class TestModelWithReqScopedCustomHeaders:
    @pytest.mark.parametrize(
        "model_params, req_custom_headers, expected_params, test_id",
        [
            # Add custom headers to model without existing headers
            (
                {},
                {"x-custom-header1": "value1", "x-custom-header2": "value2"},
                {"default_headers": {"x-custom-header1": "value1", "x-custom-header2": "value2"}},
                "add_headers_to_empty_model",
            ),
            # Merge request headers with existing static headers
            (
                {"default_headers": {"X-Static-Header": "static_value"}},
                {"x-request-header": "request_value"},
                {"default_headers": {"x-static-header": "static_value", "x-request-header": "request_value"}},
                "merge_with_existing_headers",
            ),
            # Override existing header
            (
                {"default_headers": {"x-custom-header": "old_value"}},
                {"x-custom-header": "new_value"},
                {"default_headers": {"x-custom-header": "new_value"}},
                "override_existing_header",
            ),
            # Preserve other parameters
            (
                {"temperature": 0.7, "max_tokens": 100, "default_headers": {"X-Existing": "value"}},
                {"x-new-header": "new_value"},
                {
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "default_headers": {"x-existing": "value", "x-new-header": "new_value"},
                },
                "preserve_other_parameters",
            ),
            # Case-insensitive header merging
            (
                {"default_headers": {"X-Header-One": "value1", "X-HEADER-TWO": "value2", "x-header-three": "value3"}},
                {"x-header-one": "override1", "x-new-header": "new_value"},
                {
                    "default_headers": {
                        "x-header-one": "override1",
                        "x-header-two": "value2",
                        "x-header-three": "value3",
                        "x-new-header": "new_value",
                    }
                },
                "case_insensitive_merging",
            ),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_custom_header_merging(self, model_params, req_custom_headers, expected_params, test_id):
        """Test various scenarios of custom header merging."""
        model = Model(type="content_safety", model="test_model", engine="test_engine", parameters=model_params)

        updated_model = model_with_req_scoped_custom_headers(model, req_custom_headers)

        expected_model = model.model_copy(update={"parameters": expected_params})
        assert updated_model == expected_model

    @pytest.mark.parametrize(
        "model_params,req_custom_headers,test_id",
        [
            # Empty request headers with existing static headers
            ({"default_headers": {"X-Static": "static_value"}}, {}, "empty_request_headers"),
            # None request headers with existing static headers
            ({"default_headers": {"X-Static": "static_value"}}, None, "none_request_headers_with_static"),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_no_changes_when_no_request_headers(self, model_params, req_custom_headers, test_id):
        """Test that model is unchanged when there are no request headers to add."""
        model = Model(type="content_safety", model="test_model", engine="test_engine", parameters=model_params)

        updated_model = model_with_req_scoped_custom_headers(model, req_custom_headers)

        expected_model = model.model_copy(update={"parameters": {"default_headers": {"x-static": "static_value"}}})
        assert updated_model.parameters == {"default_headers": {"x-static": "static_value"}}
        assert updated_model == expected_model

    def test_error_handling(self, mocker: MockerFixture):
        """Test that errors are propagated when merging fails."""
        model = Model(type="content_safety", model="test_model", engine="test_engine")
        req_custom_headers = {"x-header": "value"}

        mocker.patch("nhx.guardrails.app.handlers.utils.get_merged_custom_headers", side_effect=Exception("Test error"))

        with pytest.raises(Exception, match="Test error"):
            model_with_req_scoped_custom_headers(model, req_custom_headers)


class TestEngineCacheKeyHeaders:
    def test_request_headers_do_not_change_the_key(self, mocker: MockerFixture):
        """A unique ``x-request-id`` per request (added by proxies like Envoy) mustn't change the key."""
        mocker.patch(
            "nhx.guardrails.app.handlers.utils.get_platform_auth_headers_from_context",
            return_value={"X-NHX-Principal-Id": "service:guardrails"},
        )
        keys = set()
        for request_id in ("a", "b"):
            mocker.patch(
                "nhx.guardrails.app.handlers.utils.get_request_default_headers_from_context",
                return_value={"x-request-id": request_id},
            )
            keys.add(get_merged_custom_headers_token("caller-token"))

        assert len(keys) == 1

    def test_credentials_built_into_the_engine_change_the_key(self, mocker: MockerFixture):
        """The caller's model credential and platform auth headers are built into the engine, so
        callers with different values need different engines."""
        auth = mocker.patch("nhx.guardrails.app.handlers.utils.get_platform_auth_headers_from_context")
        auth.return_value = {"X-NHX-Principal-On-Behalf-Of": "user:a"}
        user_a = get_merged_custom_headers_token("token")
        auth.return_value = {"X-NHX-Principal-On-Behalf-Of": "user:b"}
        user_b = get_merged_custom_headers_token("token")
        other_token = get_merged_custom_headers_token("other-token")

        assert len({user_a, user_b, other_token}) == 3


class TestSetMainModelMergedCustomHeadersIntoContext:
    @pytest.mark.parametrize(
        "main_model_params, test_id",
        [
            # Model with custom headers and other parameters
            (
                {"temperature": 0.7, "max_tokens": 100, "default_headers": {"X-Existing": "value"}},
                "preserve_other_model_parameters",
            ),
            # Model with multiple custom headers
            (
                {
                    "default_headers": {
                        "X-Header-1": "value1",
                        "X-Header-2": "value2",
                        "X-Header-3": "value3",
                    }
                },
                "multiple_custom_headers",
            ),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_context_updated_correctly(
        self, main_model_params, test_id, set_headers_into_context_mock: MagicMock, mocker: MockerFixture
    ):
        """Test that context is updated correctly with model's custom headers."""
        mocker.patch("nhx.guardrails.app.handlers.utils.get_request_default_headers_from_context", return_value={})
        main_model = Model(type="main", model="main_model", engine="test_engine", parameters=main_model_params)

        set_main_model_merged_custom_headers_into_context(main_model)
        assert set_headers_into_context_mock.call_count == 1

        called_headers = set_headers_into_context_mock.call_args[0][0]
        expected_headers = {k.lower(): v for k, v in main_model_params.get("default_headers", {}).items()}
        assert called_headers == expected_headers

    def test_request_headers_override_configured_headers(
        self, set_headers_into_context_mock: MagicMock, mocker: MockerFixture
    ):
        """The main model sends its configured headers with this request's headers on top. They come
        from the request context, since the model itself is shared across requests."""
        mocker.patch(
            "nhx.guardrails.app.handlers.utils.get_request_default_headers_from_context",
            return_value={"x-header": "from-request", "x-request-id": "this-request"},
        )
        main_model = Model(
            type="main",
            model="main_model",
            engine="test_engine",
            parameters={"default_headers": {"X-Header": "configured", "X-Static": "kept"}},
        )

        set_main_model_merged_custom_headers_into_context(main_model)

        set_headers_into_context_mock.assert_called_once_with(
            {"x-header": "from-request", "x-static": "kept", "x-request-id": "this-request"}
        )

    def test_none_model_raises_error(self, set_headers_into_context_mock: MagicMock):
        """Test that no headers are reset into context when model is None."""
        set_main_model_merged_custom_headers_into_context(None)
        assert set_headers_into_context_mock.call_count == 0

    def test_wrong_model_type_raises_error(self):
        """Test that ValueError is raised when model type validation fails."""
        # Create a non-main model
        non_main_model = Model(type="content_safety", model="safety_model", engine="test_engine")

        with pytest.raises(ValueError, match="Expected main model. Got: content_safety"):
            set_main_model_merged_custom_headers_into_context(non_main_model)

    def test_empty_custom_headers(self, set_headers_into_context_mock: MagicMock):
        """Test that empty dict is set when model has no custom headers."""
        main_model = Model(type="main", model="main_model", engine="test_engine")

        set_main_model_merged_custom_headers_into_context(main_model)

        set_headers_into_context_mock.assert_not_called()

    def test_error_in_set_context_propagates(self, mocker: MockerFixture):
        """Test that errors in set_request_default_headers_into_context are propagated."""
        main_model = Model(
            type="main", model="main_model", engine="test_engine", parameters={"default_headers": {"X-Header": "value"}}
        )

        mocker.patch(
            "nhx.guardrails.app.handlers.utils.set_request_default_headers_into_context",
            side_effect=Exception("Context error"),
        )

        with pytest.raises(Exception, match="Context error"):
            set_main_model_merged_custom_headers_into_context(main_model)


class TestRunGenerateAsync:
    """Tests for run_generate_async, which runs the rails on the server event loop."""

    @pytest.mark.asyncio
    async def test_returns_generate_async_result(self):
        """generate_async's result is returned to the caller."""
        expected = {"response": "hello"}
        mock_llm_rails = MagicMock()
        mock_llm_rails.generate_async = AsyncMock(return_value=expected)

        result = await run_generate_async(
            mock_llm_rails,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result == expected

    @pytest.mark.asyncio
    async def test_passes_all_kwargs_to_generate_async(self):
        """All keyword arguments are forwarded to generate_async unchanged."""
        mock_llm_rails = MagicMock()
        mock_llm_rails.generate_async = AsyncMock(return_value={})

        messages = [{"role": "user", "content": "hi"}]
        options = {"temperature": 0.5, "max_tokens": 100}
        state = {"key": "value"}
        prompt = "test prompt"

        await run_generate_async(
            mock_llm_rails,
            messages=messages,
            prompt=prompt,
            options=options,
            state=state,
        )

        mock_llm_rails.generate_async.assert_called_once_with(
            messages=messages,
            prompt=prompt,
            options=options,
            state=state,
        )

    @pytest.mark.asyncio
    async def test_exceptions_from_generate_async_propagate(self):
        """Exceptions raised inside generate_async propagate back to the caller."""
        mock_llm_rails = MagicMock()
        mock_llm_rails.generate_async = AsyncMock(side_effect=ValueError("rails error"))

        with pytest.raises(ValueError, match="rails error"):
            await run_generate_async(mock_llm_rails, messages=[])

    @pytest.mark.asyncio
    async def test_every_request_runs_on_the_server_loop(self):
        """Model clients are tied to the loop that first used them, so every request has to run on
        the server loop. With a new loop per request, later requests found the client on a closed
        loop and had to wait and retry."""
        server_loop = asyncio.get_running_loop()
        loops: list[asyncio.AbstractEventLoop] = []

        async def capture_loop(**kwargs):
            loops.append(asyncio.get_running_loop())
            return {}

        mock_llm_rails = MagicMock()
        mock_llm_rails.generate_async = AsyncMock(side_effect=capture_loop)

        await run_generate_async(mock_llm_rails, messages=[])
        await run_generate_async(mock_llm_rails, messages=[])

        assert loops == [server_loop, server_loop]

    @pytest.mark.asyncio
    async def test_request_contextvars_are_visible_to_the_rails(self):
        """Request-scoped variables (auth token, headers, main model) reach the rails pipeline."""
        _test_var: ContextVar[str] = ContextVar("_test_var")
        _test_var.set("expected_value")

        captured: str | None = None

        async def capture_contextvar(**kwargs):
            nonlocal captured
            captured = _test_var.get(None)
            return {}

        mock_llm_rails = MagicMock()
        mock_llm_rails.generate_async = AsyncMock(side_effect=capture_contextvar)

        await run_generate_async(mock_llm_rails, messages=[])

        assert captured == "expected_value"
