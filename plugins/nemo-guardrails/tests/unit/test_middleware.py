# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`GuardrailsMiddleware`.

Covers:

- The resolver methods (``get_middleware_config`` / ``validate_middleware_config``)
  returning a :class:`GuardrailSource` that the request/response hooks
  consume directly.
- Cache lookups keyed by :class:`StableRailsConfig.content_hash`, with
  warming reusing the same resolver methods IGW calls per request.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
import pytest_asyncio
from nemo_guardrails_plugin.constants import GUARDRAILS_PLUGIN_CONFIG_TYPE
from nemo_guardrails_plugin.llm_clients import get_request_headers
from nemo_guardrails_plugin.llmrails_cache import (
    EntityGuardrailConfigSource,
    InlineGuardrailConfigSource,
    Provenance,
    StableRailsConfig,
)
from nemo_guardrails_plugin.middleware import (
    GUARDRAILS_LIBRARY_LOGGER_NAME,
    PLUGIN_NAME,
    STATE_KEY_GUARDRAILS_REQUEST_BODY,
    STATE_KEY_INPUT_GENERATION_RESPONSE,
    GuardrailsMiddleware,
    handle_streaming_output_check,
    is_streaming_response_result,
)
from nemo_guardrails_plugin.requests import parse_guardrails_request
from nemo_guardrails_plugin.streaming import close_async_iterator
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import NotFoundError
from nemo_platform_plugin.guardrail.client import AsyncGuardrailClient
from nemo_platform_plugin.guardrail.types import GuardrailConfig
from nemo_platform_plugin.guardrail.types import RailsConfig as SDKRailsConfig
from nemo_platform_plugin.inference_middleware import (
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceMiddlewareUnavailableError,
    InferenceRequest,
    InferenceResponse,
    ResponseResult,
)
from nemoguardrails.exceptions import LLMCallException
from nemoguardrails.rails.llm.options import ActivatedRail, GenerationLog, GenerationResponse, GenerationStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sdk_rails(
    *,
    input_flows: list[str] | None = None,
    output_flows: list[str] | None = None,
    output_streaming: dict[str, Any] | None = None,
    models: list[dict[str, Any]] | None = None,
) -> SDKRailsConfig:
    """Build a wire-format :class:`SDKRailsConfig` matching ``source.rails``.

    Defaults to "input flows configured" because most of the request-side
    tests need to exercise the rails path; pass ``input_flows=[]`` (or
    ``None``) to skip it. Uses a non-special flow name so the library-side
    :func:`stabilize` validator doesn't reject the config for missing built-in
    prompt templates (``self check input`` etc. require a registered prompt).
    """
    rails: dict[str, Any] = {}
    if input_flows is None:
        input_flows = ["custom check"]
    rails["input"] = {"flows": input_flows}
    if output_flows is not None:
        output_rails: dict[str, Any] = {"flows": output_flows}
        if output_streaming is not None:
            output_rails["streaming"] = output_streaming
        rails["output"] = output_rails
    payload: dict[str, Any] = {"rails": rails, "models": models if models is not None else []}
    return SDKRailsConfig.model_validate(payload)


def _entity_source(
    *,
    workspace: str = "my-workspace",
    name: str = "my-config",
    input_flows: list[str] | None = None,
    output_flows: list[str] | None = None,
    output_streaming: dict[str, Any] | None = None,
    models: list[dict[str, Any]] | None = None,
    updated_at: str = "2026-01-01T00:00:00Z",
) -> EntityGuardrailConfigSource:
    return EntityGuardrailConfigSource(
        workspace=workspace,
        name=name,
        updated_at=updated_at,
        rails=_sdk_rails(
            input_flows=input_flows,
            output_flows=output_flows,
            output_streaming=output_streaming,
            models=models,
        ),
    )


def _inline_source(
    *,
    label: str | None = "ad-hoc",
    input_flows: list[str] | None = None,
    output_flows: list[str] | None = None,
    output_streaming: dict[str, Any] | None = None,
    models: list[dict[str, Any]] | None = None,
) -> InlineGuardrailConfigSource:
    return InlineGuardrailConfigSource(
        label=label,
        rails=_sdk_rails(
            input_flows=input_flows,
            output_flows=output_flows,
            output_streaming=output_streaming,
            models=models,
        ),
    )


def _make_entity(
    *,
    workspace: str = "my-workspace",
    name: str = "my-config",
    input_flows: list[str] | None = None,
    output_flows: list[str] | None = None,
    output_streaming: dict[str, Any] | None = None,
    updated_at: str = "2026-01-01T00:00:00Z",
) -> GuardrailConfig:
    """Build the SDK :class:`GuardrailConfig` entity that ``get_middleware_config``'s
    SDK call returns. Distinct from :func:`_entity_source` because the
    resolver method itself converts entity → source."""
    rails = _sdk_rails(
        input_flows=input_flows,
        output_flows=output_flows,
        output_streaming=output_streaming,
    )
    return GuardrailConfig.model_validate(
        {
            "id": f"guardrail-config-{name}",
            "entity_id": f"guardrail-config-{name}",
            "parent": "",
            "workspace": workspace,
            "name": name,
            "db_version": 1,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": updated_at,
            "data": rails.model_dump(exclude_none=True),
        }
    )


def _client_response(data: Any) -> MagicMock:
    response = MagicMock()
    response.data.return_value = data
    return response


def _patch_get_guardrail_config(
    *,
    return_value: Any = None,
    side_effect: Any = None,
) -> Any:
    if return_value is not None:
        mock = AsyncMock(return_value=_client_response(return_value))
    elif callable(side_effect):

        async def _wrapped_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
            result = side_effect(*args, **kwargs)
            if hasattr(result, "__await__"):
                result = await result
            return _client_response(result)

        mock = AsyncMock(side_effect=_wrapped_side_effect)
    elif side_effect is not None:
        mock = AsyncMock(side_effect=side_effect)
    else:
        mock = AsyncMock()
    return patch.object(AsyncGuardrailClient, "get_guardrail_config", new=mock)


def _not_found_error() -> NotFoundError:
    return NotFoundError(
        httpx.Response(
            404,
            request=httpx.Request("GET", "http://test:8000/apis/guardrails/v2/workspaces/ws/configs/missing"),
            json={"detail": "not found"},
        )
    )


def _make_generation_response(
    *,
    is_blocked: bool = False,
    content: str = "I'm sorry, I can't help with that.",
) -> GenerationResponse:
    return GenerationResponse(
        response=[{"role": "assistant", "content": content}],
        log=GenerationLog(
            activated_rails=[ActivatedRail(type="input", name="self check input", stop=is_blocked)],
            stats=GenerationStats(input_rails_duration=0.1, total_duration=0.1),
        ),
    )


def _make_request(body: dict[str, Any], headers: dict[str, str] | None = None) -> InferenceRequest:
    return InferenceRequest(body=body, headers=headers or {}, path="v1/chat/completions")


def _make_response(result: Any, headers: dict[str, str] | None = None) -> InferenceResponse:
    return InferenceResponse(result=result, headers=headers or {})


def _make_ctx(
    original_body: dict[str, Any] | None = None, original_headers: dict[str, str] | None = None
) -> InferenceMiddlewareContext:
    """Build a minimal :class:`InferenceMiddlewareContext` for tests.

    ``original_request`` is the snapshot ``process_response`` reads when it
    needs the request body and headers — wire the same ``request_body`` the
    test threaded through ``process_request`` so output rails see consistent
    state.
    """
    return InferenceMiddlewareContext(
        request_id="test-req",
        virtual_model_name="test-vm",
        workspace="test-ws",
        original_request=_make_request(original_body or {}, original_headers),
    )


def _request_scoped_client(mw: GuardrailsMiddleware, headers: dict[str, str]) -> AsyncNemoClient:
    client = mw._client
    assert client is not None
    return client.with_options(headers=headers)


async def _process_request(
    mw: GuardrailsMiddleware,
    request_body: dict[str, Any],
    request_headers: dict[str, str],
    source: Any,
    *,
    ctx: InferenceMiddlewareContext | None = None,
) -> dict[str, Any] | ImmediateResponse:
    """Test wrapper preserving the pre-PR#243 shape.

    Returns the (possibly sanitized) request body dict on the pass-through
    path or :class:`ImmediateResponse` on the blocked path, mirroring how
    callers used the legacy positional signature. Constructing the
    :class:`InferenceMiddlewareContext` with the same body the test passes
    keeps ``ctx.original_request`` stable for output rails downstream.

    Pass ``ctx`` explicitly when the test needs to inspect cross-hook state
    that ``process_request`` writes into ``ctx.state(PLUGIN_NAME)`` — the
    same context object is then visible to the assertion.
    """
    request = _make_request(request_body, request_headers)
    if ctx is None:
        ctx = _make_ctx(request_body, request_headers)
    result = await mw.process_request(ctx, request, source)
    if isinstance(result, ImmediateResponse):
        return result
    return result.body


async def _process_response(
    mw: GuardrailsMiddleware,
    response_result: Any,
    request_body: dict[str, Any],
    request_headers: dict[str, str],
    response_headers: dict[str, str],
    source: Any,
    *,
    ctx: InferenceMiddlewareContext | None = None,
) -> ResponseResult:
    """Test wrapper preserving the pre-PR#243 shape.

    Returns the inner :data:`ResponseResult` (dict or ``AsyncIterator``)
    instead of the wrapping :class:`InferenceResponse` so existing
    assertions like ``assert is_streaming_response_result(result)`` keep
    working unchanged.

    Pass ``ctx`` explicitly when the test needs to seed cross-hook state
    (e.g. an input-rail :class:`GenerationResponse` from ``process_request``)
    via ``ctx.state(PLUGIN_NAME).set(...)`` before calling.
    """
    if ctx is None:
        ctx = _make_ctx(request_body, request_headers)
    response = _make_response(response_result, response_headers)
    result = await mw.process_response(ctx, response, source)
    return result.result


def _patch_prepare_lease(rails: Any | None = None):
    """Stub ``GuardrailsMiddleware._prepare_lease`` whose deferred
    ``cache.lease(...)`` yields the given mock ``LLMRails``.

    The streaming branch of ``process_response`` runs ``_prepare_lease``
    eagerly and opens ``cache.lease(...)`` lazily inside the returned async
    generator (deferring the lease closes the never-started-generator leak
    surface). Stubbing at this boundary lets streaming tests exercise the
    real generator path without standing up an ``LLMRailsCache``.
    """
    rails = rails if rails is not None else MagicMock()

    @asynccontextmanager
    async def _lease(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
        yield rails

    cache = MagicMock()
    cache.lease = _lease

    stable = StableRailsConfig(rails=MagicMock(), content_hash="hash-stub", embedding_model_id=None)

    async def _prepare(*_args: Any, **_kwargs: Any) -> tuple[Any, Any, Any, Any]:
        return cache, stable, Provenance("ws/name@ts"), MagicMock()

    return _prepare


@pytest_asyncio.fixture
async def middleware() -> AsyncIterator[GuardrailsMiddleware]:
    """A started middleware with a mocked platform client and a real cache.

    Owns startup AND shutdown so any pool-owned resources are released
    between tests.
    """
    instance = GuardrailsMiddleware()
    instance._inject_cache(MagicMock())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    ) as http_client:
        client = AsyncNemoClient(base_url="http://test:8000", http_client=http_client)
        instance._inject_platform_client(client)
        await instance.on_startup()
        try:
            yield instance
        finally:
            await instance.on_shutdown()


# ---------------------------------------------------------------------------
# get_middleware_config — returns an EntityGuardrailConfigSource (sets the discriminator)
# ---------------------------------------------------------------------------


class TestGetMiddlewareConfig:
    async def test_unsupported_config_type_raises_value_error(self, middleware: GuardrailsMiddleware) -> None:
        with pytest.raises(ValueError, match="unknown_type"):
            await middleware.get_middleware_config("unknown_type", "ws/name")

    async def test_returns_entity_source(self, middleware: GuardrailsMiddleware) -> None:
        """The resolver sets the discriminator: returns :class:`EntityGuardrailConfigSource`
        carrying provenance fields plus the SDK rails payload."""
        assert middleware._client is not None
        entity = _make_entity()

        with _patch_get_guardrail_config(return_value=entity):
            result = await middleware.get_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, "my-workspace/my-config")

        assert isinstance(result, EntityGuardrailConfigSource)
        assert result.workspace == "my-workspace"
        assert result.name == "my-config"
        assert result.updated_at == entity.updated_at.isoformat()
        assert result.rails is entity.data

    async def test_splits_config_id_correctly(self, middleware: GuardrailsMiddleware) -> None:
        assert middleware._client is not None
        entity = _make_entity()
        with _patch_get_guardrail_config(return_value=entity) as retrieve_mock:
            await middleware.get_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, "my-workspace/my-config")

        retrieve_mock.assert_awaited_once_with(name="my-config", workspace="my-workspace")

    async def test_not_found_raises_middleware_config_not_found(self, middleware: GuardrailsMiddleware) -> None:
        """A 404 from the SDK becomes :class:`MiddlewareConfigNotFoundError` so IGW
        can distinguish "config was deleted" from a transient SDK / network blip
        and evict the cached middleware accordingly. The original
        ``NotFoundError`` is chained as ``__cause__`` so debug traces preserve the
        request ID / status info the SDK exception carries.
        """
        from nemo_platform_plugin.inference_middleware import MiddlewareConfigNotFoundError

        assert middleware._client is not None
        not_found = _not_found_error()

        with _patch_get_guardrail_config(side_effect=not_found):
            with pytest.raises(MiddlewareConfigNotFoundError) as exc_info:
                await middleware.get_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, "ws/missing")

        assert exc_info.value.config_id == "ws/missing"
        assert exc_info.value.status_code == 404
        assert exc_info.value.__cause__ is not_found

    @pytest.mark.parametrize(
        "config_id",
        [
            "no-slash",  # missing workspace prefix
            "/missing-workspace",  # empty workspace
            "missing-name/",  # empty name
            "a/b/c",  # too many slashes
            "",  # empty
        ],
    )
    async def test_malformed_config_id_raises_value_error(
        self, middleware: GuardrailsMiddleware, config_id: str
    ) -> None:
        """Malformed ``config_id`` must be rejected before the SDK call.

        Bare ``str.split("/", 1)`` would silently accept several of
        these (``"no-slash"`` would unpack-raise with a confusing
        message, ``"/missing-workspace"`` would call the SDK with an
        empty workspace, ``"a/b/c"`` would silently take ``b/c`` as
        name). ``parse_entity_ref`` rejects all of them with a clear
        ValueError → 400 at the plugin boundary. Pinning so a future
        "let's just split here, it's simpler" regression can't ship."""
        assert middleware._client is not None

        with _patch_get_guardrail_config() as retrieve_mock:
            with pytest.raises(ValueError):
                await middleware.get_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, config_id)

        retrieve_mock.assert_not_awaited()

    async def test_missing_data_raises_value_error(self, middleware: GuardrailsMiddleware) -> None:
        """An entity whose ``data`` is ``None`` cannot be turned into a source —
        the structural check moves up here so the per-request path never has
        to defend against it."""
        assert middleware._client is not None
        entity = _make_entity()
        # Deliberately violates the model's non-null data contract to exercise
        # the resolver guard against a malformed SDK/entity-store response.
        object.__setattr__(entity, "data", None)

        with _patch_get_guardrail_config(return_value=entity):
            with pytest.raises(ValueError, match="no data"):
                await middleware.get_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, "ws/my-config")

    async def test_empty_updated_at_raises_value_error(self, middleware: GuardrailsMiddleware) -> None:
        """An entity carrying ``""`` for ``updated_at`` would seed a
        :class:`StabilizedRailsConfigCache` slot that can never collide cleanly
        with a real entity revision; reject at the resolver boundary."""
        assert middleware._client is not None
        entity = _make_entity()
        # Deliberately violates the model's ``datetime`` type to exercise the
        # "empty updated_at" guard against a malformed entity.
        object.__setattr__(entity, "updated_at", "")

        with _patch_get_guardrail_config(return_value=entity):
            with pytest.raises(ValueError, match="empty updated_at"):
                await middleware.get_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, "ws/my-config")

    async def test_empty_name_raises_value_error(self, middleware: GuardrailsMiddleware) -> None:
        assert middleware._client is not None
        entity = _make_entity()
        entity.name = ""

        with _patch_get_guardrail_config(return_value=entity):
            with pytest.raises(ValueError, match="no name"):
                await middleware.get_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, "ws/my-config")

    async def test_empty_workspace_raises_value_error(self, middleware: GuardrailsMiddleware) -> None:
        """Empty ``workspace`` would let two genuinely-distinct entities
        (different workspaces, same name + ``updated_at``) share a
        :class:`StabilizedRailsConfigCache` slot. Fail closed at the
        resolver boundary so the upstream never observes the collision."""
        assert middleware._client is not None
        entity = _make_entity()
        entity.workspace = ""

        with _patch_get_guardrail_config(return_value=entity):
            with pytest.raises(ValueError, match="empty workspace"):
                await middleware.get_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, "ws/my-config")


# ---------------------------------------------------------------------------
# validate_middleware_config — repaired inline path
# ---------------------------------------------------------------------------


class TestValidateMiddlewareConfig:
    async def test_unsupported_config_type_raises_value_error(self, middleware: GuardrailsMiddleware) -> None:
        with pytest.raises(ValueError, match="unknown_type"):
            await middleware.validate_middleware_config("unknown_type", {"rails": {}})

    async def test_inline_dict_returns_inline_source(self, middleware: GuardrailsMiddleware) -> None:
        """The repaired inline path: a raw :class:`SDKRailsConfig`-shaped
        dict (the natural shape that ``MiddlewareCall.config`` advertises)
        is accepted and wrapped as an :class:`InlineGuardrailConfigSource`. Previously
        ``validate_middleware_config`` required the entity wrapper shape
        and ``MiddlewareCall.config`` was therefore unusable."""
        config_dict = {
            "rails": {
                "input": {"flows": ["self check input"]},
                "output": {"flows": ["self check output"]},
            },
        }

        result = await middleware.validate_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, config_dict)

        assert isinstance(result, InlineGuardrailConfigSource)
        assert result.label is None
        assert result.rails.rails is not None
        assert result.rails.rails.input is not None
        assert result.rails.rails.input.flows == ["self check input"]

    async def test_inline_dict_picks_up_optional_label(self, middleware: GuardrailsMiddleware) -> None:
        """A caller-supplied ``name`` field on the inline dict surfaces as
        the inline source's diagnostic label so logs say ``<inline:my-test>``
        instead of ``<inline:unnamed>``. Never affects identity."""
        config_dict = {
            "name": "my-test",
            "rails": {"input": {"flows": []}},
        }

        result = await middleware.validate_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, config_dict)

        assert isinstance(result, InlineGuardrailConfigSource)
        assert result.label == "my-test"

    async def test_idempotent_for_existing_source(self, middleware: GuardrailsMiddleware) -> None:
        """Useful in tests and when ``validate_middleware_config`` is invoked
        twice (e.g. once via the request path and once via warming)."""
        original = _entity_source()
        result = await middleware.validate_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, original)
        assert result is original

    async def test_sdk_rails_instance_is_wrapped(self, middleware: GuardrailsMiddleware) -> None:
        rails = _sdk_rails(input_flows=["self check input"])
        result = await middleware.validate_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, rails)
        assert isinstance(result, InlineGuardrailConfigSource)
        assert result.rails is rails

    async def test_invalid_dict_raises_value_error(self, middleware: GuardrailsMiddleware) -> None:
        """Garbage in → ValueError out, with the underlying validation
        error chained as ``__cause__``."""
        with pytest.raises(ValueError, match="failed validation"):
            # ``rails`` must be a dict, not a list — SDK schema rejects this.
            await middleware.validate_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, {"rails": []})

    async def test_non_dict_raises_value_error(self, middleware: GuardrailsMiddleware) -> None:
        with pytest.raises(ValueError, match="must be a dict"):
            await middleware.validate_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, "not-a-dict")

    async def test_inline_dict_with_envelope_name_strips_name_for_validation(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        """Back-compat for envelope-style inline payloads: a dict that includes
        a top-level ``name`` (e.g. legacy config-as-envelope shape) must not
        fail :class:`PlatformRailsConfig` validation. ``name`` is consumed as
        the diagnostic label and the remaining payload validates as rails.

        Pinning this prevents a future ``payload.pop("name", None)`` removal
        from silently flipping every envelope-shaped inline call to a 4xx
        ``ValueError`` from the validator.
        """
        config_dict = {
            "name": "envelope-style",
            "rails": {"input": {"flows": ["self check input"]}},
        }

        result = await middleware.validate_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, config_dict)

        assert isinstance(result, InlineGuardrailConfigSource)
        assert result.label == "envelope-style"
        assert result.rails.rails is not None
        assert result.rails.rails.input is not None
        assert result.rails.rails.input.flows == ["self check input"]

    async def test_inline_dict_with_non_string_name_discards_label(self, middleware: GuardrailsMiddleware) -> None:
        """A non-string ``name`` is dropped silently (logged at DEBUG)
        rather than coerced — coercing would let an upstream typo present
        as a misleading label like ``<inline:42>``. The dict still validates
        as rails because ``name`` is popped before validation."""
        config_dict = {
            "name": 42,
            "rails": {"input": {"flows": ["self check input"]}},
        }

        result = await middleware.validate_middleware_config(GUARDRAILS_PLUGIN_CONFIG_TYPE, config_dict)

        assert isinstance(result, InlineGuardrailConfigSource)
        assert result.label is None


# ---------------------------------------------------------------------------
# process_request
# ---------------------------------------------------------------------------


class TestProcessRequest:
    """Tests for ``process_request``.

    All paths that should reach the rails layer mock ``_run_rails`` directly:
    it is the unified entry point for stabilize → ``build_main_llm`` →
    ``cache.lease`` → ``run_generate_in_new_loop`` → parse + error-mapping.
    Mocking it lets these tests focus on what ``process_request`` *does with*
    the rails outcome (sanitize, block, propagate options) without dragging
    in cache / LLM construction. The actual ``_run_rails`` pipeline —
    including the error-mapping contract — is covered separately by
    :class:`TestProcessRequestErrorSurfacing` and the integration suite.
    """

    async def test_no_input_rails_short_circuits(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}

        with patch.object(middleware, "_run_rails") as mock_run:
            result = await _process_request(middleware, request_body, {}, _entity_source(input_flows=[]))

        assert result == request_body
        assert result is not request_body
        mock_run.assert_not_called()

    async def test_no_input_rails_returns_sanitized_request_body(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"options": {"log": {"activated_rails": True}}},
        }
        ctx = _make_ctx(request_body)

        with patch.object(middleware, "_run_rails") as mock_run:
            result = await _process_request(middleware, request_body, {}, _entity_source(input_flows=[]), ctx=ctx)

        assert isinstance(result, dict)
        assert result == {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        assert "guardrails" not in result
        assert "guardrails" in request_body
        # ``process_request`` must stash the user's ``guardrails`` field on
        # ``ctx.state`` so ``process_response`` of the same request can read
        # it back when re-applying user-requested log options.
        assert ctx.state(PLUGIN_NAME).get(STATE_KEY_GUARDRAILS_REQUEST_BODY) == parse_guardrails_request(
            {"options": {"log": {"activated_rails": True}}}
        )
        mock_run.assert_not_called()

    @pytest.mark.parametrize(
        "request_body",
        [
            {"model": "ws/llama"},  # no "messages" key
            {"model": "ws/llama", "messages": "not a list"},
        ],
    )
    async def test_invalid_messages_short_circuits(
        self, middleware: GuardrailsMiddleware, request_body: dict[str, Any]
    ) -> None:
        with patch.object(middleware, "_run_rails") as mock_run:
            result = await _process_request(middleware, request_body, {}, _entity_source())

        assert result == request_body
        assert result is not request_body
        mock_run.assert_not_called()

    async def test_successful_generation_returns_request_body(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        generation_response = _make_generation_response(is_blocked=False, content="Hello")
        ctx = _make_ctx(request_body)

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=generation_response)):
            result = await _process_request(middleware, request_body, {}, _entity_source(), ctx=ctx)

        assert result == request_body
        assert result is not request_body
        # Successful input rails must hand off the GenerationResponse via
        # ``ctx.state`` so ``process_response`` can fold it into the final
        # ``guardrails_data`` payload.
        assert ctx.state(PLUGIN_NAME).get(STATE_KEY_INPUT_GENERATION_RESPONSE) is generation_response
        assert ctx.response_body_annotations["guardrails_data"]["config_ids"] == ["my-workspace/my-config"]

    async def test_input_masking_writes_back_last_user_message(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [
                {"role": "user", "content": "earlier turn"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "Hi! I am Mr. John!"},
            ],
        }
        generation_response = _make_generation_response(
            is_blocked=False,
            content="Hi! I am <TITLE> <PERSON>!",
        )

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=generation_response)):
            result = await _process_request(middleware, request_body, {}, _entity_source())

        assert not isinstance(result, ImmediateResponse)
        assert result["messages"] == [
            {"role": "user", "content": "earlier turn"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Hi! I am <TITLE> <PERSON>!"},
        ]
        # Original request body must not be mutated (IGW aliases nested values).
        assert request_body["messages"][-1]["content"] == "Hi! I am Mr. John!"

    async def test_input_rails_receive_only_current_turn_after_tool_history(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        current_user = {"role": "user", "content": "What about tomorrow?"}
        request_body = {
            "model": "ws/llama",
            "messages": [
                {"role": "user", "content": "Look up the weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
                current_user,
            ],
        }
        generation_response = _make_generation_response(is_blocked=False, content="What about tomorrow?")
        run_rails = AsyncMock(return_value=generation_response)

        with patch.object(middleware, "_run_rails", new=run_rails):
            result = await _process_request(middleware, request_body, {}, _entity_source())

        assert not isinstance(result, ImmediateResponse)
        assert run_rails.call_args.kwargs["messages"] == [current_user]
        assert result["messages"] == request_body["messages"]

    async def test_input_rails_skip_request_ending_with_tool_result(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [
                {"role": "user", "content": "Look up the weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
            ],
        }

        with patch.object(middleware, "_run_rails") as run_rails:
            result = await _process_request(middleware, request_body, {}, _entity_source())

        assert result == request_body
        assert result is not request_body
        run_rails.assert_not_called()

    async def test_input_masking_skips_non_string_user_content(self, middleware: GuardrailsMiddleware) -> None:
        # Multimodal content is unsupported for PII write-back; leave the request alone
        # even when response.content looks like a redacted string (or a stringified list).
        multimodal_content = [
            {"type": "text", "text": "Hi! I am Mr. John!"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": multimodal_content}],
        }
        generation_response = _make_generation_response(
            is_blocked=False,
            content=str(multimodal_content),
        )

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=generation_response)):
            result = await _process_request(middleware, request_body, {}, _entity_source())

        assert not isinstance(result, ImmediateResponse)
        assert result["messages"][0]["content"] == multimodal_content

    async def test_user_log_options_forwarded_to_run_rails(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"options": {"log": {"activated_rails": True, "internal_events": True}}},
        }
        generation_response = _make_generation_response(is_blocked=False)
        run_rails_mock = AsyncMock(return_value=generation_response)

        with patch.object(middleware, "_run_rails", new=run_rails_mock):
            await _process_request(middleware, request_body, {}, _entity_source())

        kwargs = run_rails_mock.call_args.kwargs
        assert kwargs["user_log_options"] == {"activated_rails": True, "internal_events": True}

    @pytest.mark.parametrize(
        "guardrails",
        [
            {"config": {"rails": {}}},
            {"config_id": "default/my-config"},
            {"options": {"rails": {"input": False}}},
            {"options": {"log": {"unknown": True}}},
        ],
    )
    async def test_rejects_unsupported_guardrails_request_fields(
        self, middleware: GuardrailsMiddleware, guardrails: dict[str, Any]
    ) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": guardrails,
        }

        with patch.object(middleware, "_run_rails") as mock_run:
            with pytest.raises(InferenceMiddlewareError) as exc_info:
                await _process_request(middleware, request_body, {}, _entity_source())

        assert exc_info.value.status_code == 422
        mock_run.assert_not_called()

    async def test_blocked_rail_returns_immediate_response(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Do something bad"}]}
        generation_response = _make_generation_response(is_blocked=True)
        ctx = _make_ctx(request_body)

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=generation_response)):
            result = await _process_request(
                middleware,
                request_body,
                {},
                _entity_source(workspace="ws", name="my-config"),
                ctx=ctx,
            )

        assert isinstance(result, ImmediateResponse)
        assert not isinstance(result.data, AsyncIterator)

        data: dict[str, Any] = result.data  # type: ignore[assignment]
        assert data["id"].startswith("chatcmpl-")
        assert data["model"] == "ws/llama"
        assert data["choices"] == [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "I'm sorry, I can't help with that."},
                "finish_reason": "content_filter",
            }
        ]
        assert "guardrails_data" not in data
        # Wire identifier is ``workspace/name`` (stable across config
        # revisions). ``provenance.label`` would also embed ``updated_at``
        # for diagnostics, but the wire format keeps the legacy shape so
        # downstream consumers don't have to track per-revision IDs.
        assert result.response_body_annotations["guardrails_data"]["config_ids"] == ["ws/my-config"]
        # Must stash even on the blocked path — response middleware still runs
        # after ImmediateResponse and rebuilds guardrails_data from this state.
        assert ctx.state(PLUGIN_NAME).get(STATE_KEY_INPUT_GENERATION_RESPONSE) is generation_response

    async def test_blocked_input_preserves_requested_log_through_response_middleware(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        """Input block + response middleware must still surface requested log fields.

        IGW always runs response middleware after an ImmediateResponse. That
        handler rebuilds ``guardrails_data`` from plugin state, so a blocked
        input rail that skipped the stash would drop ``activated_rails`` (and
        other requested log fields) from the final response.
        """
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Do something bad"}],
            "guardrails": {"options": {"log": {"activated_rails": True}}},
        }
        input_blocked = _make_generation_response(is_blocked=True)
        output_ok = GenerationResponse(
            response=[{"role": "assistant", "content": "I'm sorry, I can't help with that."}],
            log=GenerationLog(
                activated_rails=[ActivatedRail(type="output", name="custom check output", stop=False)],
                stats=GenerationStats(output_rails_duration=0.05, total_duration=0.05),
            ),
        )
        source = _entity_source(
            workspace="ws",
            name="my-config",
            output_flows=["custom check output"],
        )
        ctx = _make_ctx(request_body)

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=input_blocked)):
            blocked = await _process_request(middleware, request_body, {}, source, ctx=ctx)

        assert isinstance(blocked, ImmediateResponse)
        # Mimic IGW wrapping ImmediateResponse before the response chain.
        inference_response = InferenceResponse(
            result=blocked.data,
            headers={},
            response_body_annotations={
                **ctx.response_body_annotations,
                **blocked.response_body_annotations,
            },
        )

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=output_ok)):
            final = await middleware.process_response(ctx, inference_response, source)

        guardrails_data = final.response_body_annotations["guardrails_data"]
        activated_rails = (guardrails_data.get("log") or {}).get("activated_rails") or []
        activated_by_name = {rail["name"]: rail for rail in activated_rails}

        # Verify the input rail's log fields are surfaced in the response.
        assert "self check input" in activated_by_name

    async def test_inline_source_blocked_rail_uses_inline_label(self, middleware: GuardrailsMiddleware) -> None:
        """An inline source's diagnostic label flows through into the
        ``guardrails_data.config_ids`` field of the blocked response so
        operators can correlate the block with the inline call."""
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "bad"}]}
        generation_response = _make_generation_response(is_blocked=True)

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=generation_response)):
            result = await _process_request(
                middleware,
                request_body,
                {},
                _inline_source(label="my-test"),
            )

        assert isinstance(result, ImmediateResponse)
        assert not isinstance(result.data, AsyncIterator)
        data = cast(dict[str, Any], result.data)
        assert "guardrails_data" not in data
        assert result.response_body_annotations["guardrails_data"]["config_ids"] == ["<inline:my-test>"]

    async def test_run_rails_failure_propagates(self, middleware: GuardrailsMiddleware) -> None:
        """``_run_rails`` is the layer that wraps lease-setup and generation
        failures into 503s. ``process_request`` must let those propagate
        unchanged — the IGW maps the resulting :class:`InferenceMiddlewareUnavailableError`
        to the wire format. The 503 wrapping itself is exercised in
        :class:`TestProcessRequestErrorSurfacing` against the real
        ``_run_rails``; here we just confirm propagation."""
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        with patch.object(
            middleware,
            "_run_rails",
            new=AsyncMock(side_effect=InferenceMiddlewareUnavailableError("Failed to run input rails")),
        ):
            with pytest.raises(InferenceMiddlewareUnavailableError, match="Failed to run input rails"):
                await _process_request(middleware, request_body, {}, _entity_source())

    async def test_missing_activation_log_raises_500(self, middleware: GuardrailsMiddleware) -> None:
        """When NeMo Guardrails returns a ``GenerationResponse`` without a log,
        ``process_request`` must surface a 500 rather than silently treating
        the request as blocked (the old fail-closed behaviour that produced a
        200 ``content_filter`` response indistinguishable from a genuine block).
        """
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        no_log_response = GenerationResponse(response="ok", log=None)

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=no_log_response)):
            with pytest.raises(InferenceMiddlewareError) as exc_info:
                await _process_request(middleware, request_body, {}, _entity_source())

        assert exc_info.value.status_code == 500
        assert "Unable to determine guardrail result" in exc_info.value.detail


# ---------------------------------------------------------------------------
# process_response
# ---------------------------------------------------------------------------


class TestHandleStreamingOutputCheck:
    async def test_runs_stream_async_and_wraps_stream_async_response(self) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}

        async def stream_response() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            }

        rails = MagicMock()

        async def stream_async_response(
            generator: AsyncIterator[str],
            messages: list[dict[str, Any]],
        ) -> AsyncIterator[str]:
            assert messages == request_body["messages"]
            async for token in generator:
                yield token.upper()

        rails.stream_async.side_effect = stream_async_response

        result = handle_streaming_output_check(
            rails,
            stream_response(),
            request_body,
            request_body["messages"],
        )

        chunks = [chunk async for chunk in result]

        assert rails.stream_async.call_count == 1
        assert chunks == [
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "HI"}}],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

    async def test_iteration_error_yields_error_payload(self) -> None:
        """When ``stream_async`` raises mid-iteration the ``except Exception``
        fallback in :func:`handle_streaming_output_check` must surface a
        terminal error chunk to the client instead of letting the
        exception propagate out of the async generator (which would
        bubble through IGW and break SSE framing).
        """
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}

        async def stream_response() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            }

        rails = MagicMock()

        async def stream_async_response(
            generator: AsyncIterator[str],
            messages: list[dict[str, Any]],
        ) -> AsyncIterator[str]:
            async for token in generator:
                if token:
                    raise RuntimeError("stream failed")
                yield token

        rails.stream_async.side_effect = stream_async_response

        result = handle_streaming_output_check(
            rails,
            stream_response(),
            request_body,
            request_body["messages"],
        )

        assert [chunk async for chunk in result] == [
            {
                "error": {
                    "message": "stream failed",
                    "type": "RuntimeError",
                    "param": "",
                    "code": "500",
                }
            }
        ]


class TestProcessResponse:
    async def test_streaming_runs_output_rails(self, middleware: GuardrailsMiddleware) -> None:
        latest_user = {"role": "user", "content": "What about tomorrow?"}
        request_body = {
            "model": "ws/llama",
            "messages": [
                {"role": "user", "content": "Look up the weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
                latest_user,
            ],
            "stream": True,
        }

        async def stream_response() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            }

        stream = stream_response()
        rails = MagicMock()

        async def stream_async_response(
            generator: AsyncIterator[str],
            messages: list[dict[str, Any]],
        ) -> AsyncIterator[str]:
            assert messages == [latest_user]
            async for token in generator:
                yield token.upper()

        rails.stream_async.side_effect = stream_async_response

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease(rails)):
            result = await _process_response(
                middleware,
                stream,
                request_body,
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )

        assert result is not stream
        assert is_streaming_response_result(result)
        chunks = [chunk async for chunk in result]

        assert rails.stream_async.call_count == 1
        assert chunks == [
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "HI"}}],
            },
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

    async def test_streaming_blocked_response_is_returned_as_error_chunk(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        """End-to-end: when ``stream_async`` emits a guardrail-violation
        JSON error string (after legitimate tokens), the resulting wire
        stream contains the successfully-streamed tokens followed by a
        structured error chunk. Pinned because ``parse_streaming_error_token``
        is the only path that converts rail-emitted errors into IGW's
        chunk shape, and this is the only test that exercises it through
        ``process_response``.
        """
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        async def stream_response() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"content": "unsafe"}, "finish_reason": None}],
            }

        rails = MagicMock()

        async def stream_async_response(
            generator: AsyncIterator[str],
            messages: list[dict[str, Any]],
        ) -> AsyncIterator[str]:
            async for token in generator:
                yield token
            yield (
                '{"error":{"message":"Blocked by self check output rails.",'
                '"type":"guardrails_violation","param":"self check output","code":"content_blocked"}}'
            )

        rails.stream_async.side_effect = stream_async_response

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease(rails)):
            result = await _process_response(
                middleware,
                stream_response(),
                request_body,
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )

        assert is_streaming_response_result(result)
        assert [chunk async for chunk in result] == [
            {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "unsafe"}}],
            },
            {
                "error": {
                    "message": "Blocked by self check output rails.",
                    "type": "guardrails_violation",
                    "param": "self check output",
                    "code": "content_blocked",
                }
            },
        ]

    async def test_streaming_disabled_rejects_streaming_output_rails(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        async def stream_response() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            }

        stream = stream_response()

        with patch.object(middleware, "_prepare_lease") as mock_prepare:
            with pytest.raises(
                InferenceMiddlewareError,
                match="Streaming output rails are disabled",
            ) as exc_info:
                await _process_response(
                    middleware,
                    stream,
                    request_body,
                    {},
                    {},
                    _entity_source(
                        output_flows=["self check output"],
                        output_streaming={"enabled": False},
                    ),
                )

        assert exc_info.value.status_code == 400
        mock_prepare.assert_not_called()

    async def test_streaming_no_output_flows_returns_original_stream(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

        async def stream_response() -> AsyncIterator[dict[str, Any]]:
            yield {"choices": [{"delta": {"content": "Hi"}}]}

        stream = stream_response()

        with patch.object(middleware, "_prepare_lease") as mock_prepare:
            result = await _process_response(middleware, stream, request_body, {}, {}, _entity_source())

        assert result is stream
        mock_prepare.assert_not_called()

    @pytest.mark.parametrize(
        "request_body",
        [
            {"model": "ws/llama"},  # no "messages" key
            {"model": "ws/llama", "messages": "not a list"},
        ],
    )
    async def test_returns_early_for_invalid_messages(
        self, middleware: GuardrailsMiddleware, request_body: dict[str, Any]
    ) -> None:
        response_result = {"id": "chatcmpl-123", "choices": []}

        result = await _process_response(middleware, response_result, request_body, {}, {}, _entity_source())

        assert result is response_result

    async def test_no_output_flows_calls_build_output_response_body(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}], "n": 2}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }

        input_response = _make_generation_response()
        ctx = _make_ctx(request_body)
        # Mimic ``process_request`` having stashed its GenerationResponse
        # so ``process_response`` of the same request can fold it into the
        # final ``guardrails_data`` payload.
        ctx.state(PLUGIN_NAME).set(STATE_KEY_INPUT_GENERATION_RESPONSE, input_response)
        mock_middleware_response: dict[str, Any] = {
            **response_result,
            "guardrails_data": {"config_ids": ["my-workspace/my-config"]},
        }

        with patch(
            "nemo_guardrails_plugin.middleware.build_output_response_body", return_value=mock_middleware_response
        ) as mock_build:
            result = await _process_response(
                middleware,
                response_result,
                request_body,
                {},
                {},
                _entity_source(),  # no output_flows → source_has_output_flows=False
                ctx=ctx,
            )

        assert result == response_result
        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["generation_response"] is None
        assert call_kwargs["input_generation_response"] is input_response

    async def test_response_middleware_clears_request_guardrails_annotation_for_return_choice(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        response = _make_response(response_result)
        response.response_body_annotations["guardrails_data"] = {"config_ids": ["request/fallback"]}
        ctx = _make_ctx(request_body)
        ctx.state(PLUGIN_NAME).set(STATE_KEY_INPUT_GENERATION_RESPONSE, _make_generation_response())
        ctx.state(PLUGIN_NAME).set(STATE_KEY_GUARDRAILS_REQUEST_BODY, parse_guardrails_request({"return_choice": True}))

        result = await middleware.process_response(ctx, response, _entity_source())

        assert "guardrails_data" not in result.response_body_annotations
        assert result.result == {
            "id": "chatcmpl-123",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"},
                {
                    "index": 1,
                    "message": {
                        "role": "guardrails_data",
                        "content": '{"config_ids":["my-workspace/my-config"]}',
                    },
                },
            ],
        }

    async def test_response_uses_guardrails_options_from_context(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        ctx = _make_ctx(request_body)
        ctx.state(PLUGIN_NAME).set(
            STATE_KEY_GUARDRAILS_REQUEST_BODY,
            parse_guardrails_request(
                {"options": {"log": {"activated_rails": True, "llm_calls": True}}, "return_choice": True}
            ),
        )
        mock_result: dict[str, Any] = {
            **response_result,
            "guardrails_data": {"config_ids": ["my-workspace/my-config"]},
        }

        with patch(
            "nemo_guardrails_plugin.middleware.build_output_response_body", return_value=mock_result
        ) as mock_build:
            result = await _process_response(
                middleware,
                response_result,
                request_body,
                {},
                {},
                _entity_source(),
                ctx=ctx,
            )

        assert result == response_result
        assert mock_build.call_args.kwargs["user_log_options"] == {"activated_rails": True, "llm_calls": True}
        assert mock_build.call_args.kwargs["return_guardrails_data_as_choice"] is True

    async def test_response_does_not_reparse_when_request_hook_stashed_no_guardrails(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        ctx = _make_ctx(request_body)
        ctx.state(PLUGIN_NAME).set(STATE_KEY_GUARDRAILS_REQUEST_BODY, None)

        with patch("nemo_guardrails_plugin.middleware.parse_guardrails_request") as mock_parse:
            await _process_response(
                middleware,
                response_result,
                request_body,
                {},
                {},
                _entity_source(),
                ctx=ctx,
            )

        mock_parse.assert_not_called()

    async def test_response_only_path_rejects_unsupported_guardrails_request_fields(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"config_id": "default/my-config"},
        }
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }

        with pytest.raises(InferenceMiddlewareError) as exc_info:
            await _process_response(middleware, response_result, request_body, {}, {}, _entity_source())

        assert exc_info.value.status_code == 422

    async def test_output_flows_reject_multiple_choices(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}], "n": 2}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"},
            ],
        }

        with patch.object(middleware, "_run_rails") as mock_run:
            with pytest.raises(
                InferenceMiddlewareError,
                match="Output rails do not support multiple completion choices",
            ) as exc_info:
                await _process_response(
                    middleware,
                    response_result,
                    request_body,
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                )

        assert exc_info.value.status_code == 400
        mock_run.assert_not_called()

    async def test_run_rails_failure_propagates(self, middleware: GuardrailsMiddleware) -> None:
        """Companion to ``TestProcessRequest.test_run_rails_failure_propagates``:
        ``process_response`` must surface ``_run_rails``'s 503 unchanged."""
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }

        with patch.object(
            middleware,
            "_run_rails",
            new=AsyncMock(side_effect=InferenceMiddlewareUnavailableError("Failed to run output rails")),
        ):
            with pytest.raises(InferenceMiddlewareUnavailableError, match="output rails"):
                await _process_response(
                    middleware,
                    response_result,
                    request_body,
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                )

    async def test_missing_activation_log_raises_500(self, middleware: GuardrailsMiddleware) -> None:
        """Companion to ``TestProcessRequest.test_missing_activation_log_raises_500``:
        output rails must also surface a 500 when NeMo Guardrails returns a
        ``GenerationResponse`` with no log, rather than silently treating the
        response as blocked.
        """
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        no_log_response = GenerationResponse(response="ok", log=None)

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=no_log_response)):
            with pytest.raises(InferenceMiddlewareError) as exc_info:
                await _process_response(
                    middleware,
                    response_result,
                    request_body,
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                )

        assert exc_info.value.status_code == 500
        assert "Unable to determine guardrail result" in exc_info.value.detail

    async def test_blocked_output_rail(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        blocked_response = _make_generation_response(is_blocked=True)
        mock_result: dict[str, Any] = {
            **response_result,
            "guardrails_data": {"config_ids": ["my-workspace/my-config"]},
        }

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=blocked_response)):
            with patch(
                "nemo_guardrails_plugin.middleware.build_blocked_output_response_body", return_value=mock_result
            ) as mock_build:
                result = await _process_response(
                    middleware,
                    response_result,
                    request_body,
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                )

        assert result == response_result
        mock_build.assert_called_once()

    async def test_passed_output_rail(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        passed_response = _make_generation_response(is_blocked=False)
        mock_middleware_response: dict[str, Any] = {
            **response_result,
            "guardrails_data": {"config_ids": ["my-workspace/my-config"]},
        }

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=passed_response)):
            with patch(
                "nemo_guardrails_plugin.middleware.build_output_response_body",
                return_value=mock_middleware_response,
            ) as mock_build:
                result = await _process_response(
                    middleware,
                    response_result,
                    request_body,
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                )

        assert result == response_result
        mock_build.assert_called_once()

    async def test_output_rails_receive_latest_user_and_response_without_tool_history(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        latest_user = {"role": "user", "content": "What about tomorrow?"}
        assistant_response = {"role": "assistant", "content": "Tomorrow will be sunny."}
        request_body = {
            "model": "ws/llama",
            "messages": [
                {"role": "user", "content": "Look up the weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
                latest_user,
            ],
        }
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": assistant_response, "finish_reason": "stop"}],
        }
        passed_response = _make_generation_response(is_blocked=False, content=assistant_response["content"])
        run_rails = AsyncMock(return_value=passed_response)

        with patch.object(middleware, "_run_rails", new=run_rails):
            await _process_response(
                middleware,
                response_result,
                request_body,
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )

        assert run_rails.call_args.kwargs["messages"] == [latest_user, assistant_response]

    @pytest.mark.parametrize("is_blocked", [True, False])
    async def test_input_generation_response_propagated(
        self, middleware: GuardrailsMiddleware, is_blocked: bool
    ) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hello"}]}
        response_result = {
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        output_response = _make_generation_response(is_blocked=is_blocked)
        input_response = _make_generation_response()
        ctx = _make_ctx(request_body)
        ctx.state(PLUGIN_NAME).set(STATE_KEY_INPUT_GENERATION_RESPONSE, input_response)

        build_response_path = (
            "nemo_guardrails_plugin.middleware.build_blocked_output_response_body"
            if is_blocked
            else "nemo_guardrails_plugin.middleware.build_output_response_body"
        )

        mock_result: dict[str, Any] = {
            **response_result,
            "guardrails_data": {"config_ids": ["my-workspace/my-config"]},
        }

        with patch.object(middleware, "_run_rails", new=AsyncMock(return_value=output_response)):
            with patch(build_response_path, return_value=mock_result) as mock_build:
                await _process_response(
                    middleware,
                    response_result,
                    request_body,
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                    ctx=ctx,
                )

        assert mock_build.call_args.kwargs["input_generation_response"] is input_response

    # -------------------------------------------------------------------
    # The returned ``InferenceResponse`` must have ``typed_body=None``.
    #
    # IGW's wire serializer (``_active_response_result`` in
    # ``inference_gateway/api/proxy.py``) prefers ``typed_body`` over
    # ``result`` whenever it's not ``None``. Every guardrails-driven
    # mutation produces a non-OpenAI shape (``guardrails_data``, blocked
    # rewrite, streaming wrapper), so a typed view carried forward from
    # upstream would silently shadow the changes. ``process_response``
    # returns a freshly-constructed ``InferenceResponse`` to make this
    # property structural — these tests pin that contract by populating
    # ``typed_body`` upstream and asserting both that the input is
    # untouched and that the returned response carries ``typed_body=None``.
    # -------------------------------------------------------------------

    async def test_no_output_flows_returns_fresh_response_without_typed_body(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hi"}]}
        response_result = {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        ctx = _make_ctx(request_body)
        upstream_typed = MagicMock(name="upstream-typed-result")
        response = _make_response(response_result)
        response.typed_body = upstream_typed
        guardrails_shaped = {**response_result, "guardrails_data": {"config_ids": ["my-workspace/my-config"]}}

        with patch("nemo_guardrails_plugin.middleware.build_output_response_body", return_value=guardrails_shaped):
            returned = await middleware.process_response(ctx, response, _entity_source())

        assert returned is not response
        assert returned.result == response_result
        assert returned.response_body_annotations == {"guardrails_data": {"config_ids": ["my-workspace/my-config"]}}
        assert returned.typed_body is None
        # Input untouched — fresh construction is the structural guarantee
        # that the bug can't recur.
        assert response.result is response_result
        assert response.typed_body is upstream_typed

    @pytest.mark.parametrize("is_blocked", [False, True], ids=["passed", "blocked"])
    async def test_output_rails_returns_fresh_response_without_typed_body(
        self, middleware: GuardrailsMiddleware, is_blocked: bool
    ) -> None:
        request_body = {"model": "ws/llama", "messages": [{"role": "user", "content": "Hi"}]}
        response_result = {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        }
        ctx = _make_ctx(request_body)
        upstream_typed = MagicMock(name="upstream-typed-result")
        response = _make_response(response_result)
        response.typed_body = upstream_typed
        guardrails_shaped = {**response_result, "guardrails_data": {"config_ids": ["my-workspace/my-config"]}}

        build_path = (
            "nemo_guardrails_plugin.middleware.build_blocked_output_response_body"
            if is_blocked
            else "nemo_guardrails_plugin.middleware.build_output_response_body"
        )

        with (
            patch.object(
                middleware,
                "_run_rails",
                new=AsyncMock(return_value=_make_generation_response(is_blocked=is_blocked)),
            ),
            patch(build_path, return_value=guardrails_shaped),
        ):
            returned = await middleware.process_response(
                ctx, response, _entity_source(output_flows=["self check output"])
            )

        assert returned is not response
        assert returned.result == response_result
        assert returned.response_body_annotations == {"guardrails_data": {"config_ids": ["my-workspace/my-config"]}}
        assert returned.typed_body is None
        assert response.result is response_result
        assert response.typed_body is upstream_typed

    async def test_streaming_returns_fresh_response_without_typed_body(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        async def upstream_stream() -> AsyncIterator[dict[str, Any]]:
            return
            yield  # pragma: no cover - async generator marker

        original_result = upstream_stream()
        upstream_typed = MagicMock(name="upstream-typed-iter")
        response = _make_response(original_result)
        response.typed_body = upstream_typed
        ctx = _make_ctx(request_body)

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            returned = await middleware.process_response(
                ctx, response, _entity_source(output_flows=["self check output"])
            )

        wrapped = returned.result
        try:
            assert returned is not response
            assert returned.typed_body is None
            assert isinstance(wrapped, AsyncIterator)
            assert wrapped is not original_result
            assert response.result is original_result
            assert response.typed_body is upstream_typed
        finally:
            if isinstance(wrapped, AsyncIterator):
                await close_async_iterator(wrapped)
            await close_async_iterator(original_result)


# ---------------------------------------------------------------------------
# Streaming lease lifecycle: cancellation must reach Pool.acquire's discard
# ---------------------------------------------------------------------------


class TestStreamingLeaseLifecycle:
    """Pin the resource-lifetime contract for streaming responses.

    ``process_response``'s streaming branch enters the lease eagerly (so
    setup failures surface as a synchronous 503) and releases it from the
    returned generator's ``finally``. The release must propagate
    :func:`sys.exc_info` into ``lease_ctx.__aexit__`` so cancellation reaches
    :meth:`Pool.acquire`'s ``except asyncio.CancelledError: discard = True``
    branch — otherwise a cancelled stream silently re-pools an instance
    whose worker may still be mutating ``events_history_cache`` /
    ``explain_info``.
    """

    @staticmethod
    def _patch_with_real_pool(rails: Any) -> tuple[Any, Any]:
        """Build a real :class:`Pool` and a ``_prepare_lease`` patch that
        leases from it. Returns ``(pool, patched_prepare_lease)``.

        Tests use a real :class:`Pool` rather than the fully mocked
        :func:`_patch_prepare_lease` helper because the discard branch
        under test lives inside :meth:`Pool.acquire` — stubbing the lease
        with an ``@asynccontextmanager`` that just yields a mock would
        short-circuit exactly the behaviour the regression test must
        exercise.

        Patches at the ``_prepare_lease`` boundary because the streaming
        branch of :meth:`process_response` reaches the real ``cache.lease``
        from inside the returned async generator — see the docstring of
        :meth:`GuardrailsMiddleware._prepare_lease` for why deferring the
        lease is load-bearing for the never-iterated leak fix.
        """
        from nemo_guardrails_plugin.llmrails_cache import Pool
        from nemoguardrails import RailsConfig

        async def build(_config: RailsConfig) -> Any:
            return rails

        stable = StableRailsConfig(
            rails=MagicMock(spec=RailsConfig),
            content_hash="hash-stub",
            embedding_model_id=None,
        )
        pool = Pool(stable=stable)

        @asynccontextmanager
        async def cache_lease(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            async with pool.acquire(build) as r:
                yield r

        cache = MagicMock()
        cache.lease = cache_lease

        async def prepare(*_args: Any, **_kwargs: Any) -> tuple[Any, Any, Any, Any]:
            return cache, stable, Provenance("ws/name@ts"), MagicMock()

        return pool, prepare

    @staticmethod
    def _make_rails(stream_async_impl: Any) -> Any:
        """Build a fake :class:`LLMRails` whose ``stream_async`` runs ``stream_async_impl``."""
        rails = SimpleNamespace(
            events_history_cache={},
            explain_info=None,
            stream_async=MagicMock(side_effect=stream_async_impl),
            update_llm=lambda _llm: None,
        )
        return rails

    @staticmethod
    def _streaming_request() -> dict[str, Any]:
        return {
            "model": "ws/llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }

    @staticmethod
    def _stream_one_chunk() -> AsyncIterator[dict[str, Any]]:
        async def _gen() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": "chatcmpl-cancel",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}],
            }

        return _gen()

    async def test_streaming_keeps_platform_header_context_active_during_iteration(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        active = False
        observed_active: list[bool] = []

        @contextmanager
        def _platform_context(_sdk: Any):
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

        def _handle_streaming_output_check(*_args: Any, **_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            async def _inner() -> AsyncIterator[dict[str, Any]]:
                observed_active.append(active)
                yield {"choices": [{"delta": {"content": "checked"}}]}
                observed_active.append(active)

            return _inner()

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch("nemo_guardrails_plugin.middleware.platform_headers_context", new=_platform_context):
                with patch(
                    "nemo_guardrails_plugin.middleware.handle_streaming_output_check",
                    new=_handle_streaming_output_check,
                ):
                    result = await _process_response(
                        middleware,
                        self._stream_one_chunk(),
                        self._streaming_request(),
                        {},
                        {},
                        _entity_source(output_flows=["self check output"]),
                    )
                    assert is_streaming_response_result(result)
                    chunks = [chunk async for chunk in result]

        assert chunks == [{"choices": [{"delta": {"content": "checked"}}]}]
        assert observed_active == [True, True]
        assert active is False

    async def test_streaming_uses_request_scoped_client_headers(self, middleware: GuardrailsMiddleware) -> None:
        request_body = self._streaming_request()
        request_headers = {
            "traceparent": "00-stream-request-trace",
            "X-NMP-Principal-On-Behalf-Of": "user:bob",
        }
        ctx = _make_ctx(request_body)
        ctx.request_nemo_client = _request_scoped_client(middleware, request_headers)
        observed_headers: list[dict[str, str]] = []

        def _handle_streaming_output_check(*_args: Any, **_kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            async def _inner() -> AsyncIterator[dict[str, Any]]:
                observed_headers.append(dict(get_request_headers()))
                yield {"choices": [{"delta": {"content": "checked"}}]}

            return _inner()

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch(
                "nemo_guardrails_plugin.middleware.handle_streaming_output_check",
                new=_handle_streaming_output_check,
            ):
                result = await _process_response(
                    middleware,
                    self._stream_one_chunk(),
                    request_body,
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                    ctx=ctx,
                )
                assert is_streaming_response_result(result)
                chunks = [chunk async for chunk in result]

        assert chunks == [{"choices": [{"delta": {"content": "checked"}}]}]
        assert observed_headers == [request_headers]

    async def test_natural_completion_returns_rails_to_pool(self, middleware: GuardrailsMiddleware) -> None:
        async def stream_async_impl(generator: Any, messages: Any) -> AsyncIterator[str]:
            async for token in generator:
                yield token

        rails = self._make_rails(stream_async_impl)
        pool, patched_prepare = self._patch_with_real_pool(rails)

        with patch.object(middleware, "_prepare_lease", new=patched_prepare):
            result = await _process_response(
                middleware,
                self._stream_one_chunk(),
                self._streaming_request(),
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )
            assert is_streaming_response_result(result)
            async for _ in result:
                pass

        assert pool._leased == 0
        assert len(pool._idle) == 1, "naturally-completed stream must return the rails to the idle queue"

    async def test_cancellation_mid_stream_discards_rails(self, middleware: GuardrailsMiddleware) -> None:
        first_chunk_yielded = asyncio.Event()

        async def stream_async_impl(generator: Any, messages: Any) -> AsyncIterator[str]:
            async for token in generator:
                yield token
                first_chunk_yielded.set()
                await asyncio.sleep(3600)

        rails = self._make_rails(stream_async_impl)
        pool, patched_prepare = self._patch_with_real_pool(rails)

        with patch.object(middleware, "_prepare_lease", new=patched_prepare):
            result = await _process_response(
                middleware,
                self._stream_one_chunk(),
                self._streaming_request(),
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )
            assert is_streaming_response_result(result)

            chunks: list[dict[str, Any]] = []

            async def consume() -> None:
                async for chunk in result:
                    chunks.append(chunk)

            task = asyncio.create_task(consume())
            await asyncio.wait_for(first_chunk_yielded.wait(), timeout=2.0)
            assert pool._leased == 1
            assert len(pool._idle) == 0

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert pool._leased == 0, "_leased must decrement on cancellation"
        assert len(pool._idle) == 0, (
            "cancelled streaming lease must discard the rails; requeuing would let "
            "the next acquirer race the suspended generator's pending writes"
        )

    async def test_iteration_error_does_not_discard(self, middleware: GuardrailsMiddleware) -> None:
        async def stream_async_impl(generator: Any, messages: Any) -> AsyncIterator[str]:
            async for token in generator:
                yield token
            raise RuntimeError("boom")

        rails = self._make_rails(stream_async_impl)
        pool, patched_prepare = self._patch_with_real_pool(rails)

        with patch.object(middleware, "_prepare_lease", new=patched_prepare):
            result = await _process_response(
                middleware,
                self._stream_one_chunk(),
                self._streaming_request(),
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )
            assert is_streaming_response_result(result)
            chunks = [c async for c in result]
            assert any("error" in c for c in chunks)

        assert pool._leased == 0
        assert len(pool._idle) == 1

    async def test_returned_generator_dropped_without_iteration_does_not_leak_lease(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        rails = self._make_rails(lambda generator, messages: generator)
        pool, patched_prepare = self._patch_with_real_pool(rails)

        with patch.object(middleware, "_prepare_lease", new=patched_prepare):
            result = await _process_response(
                middleware,
                self._stream_one_chunk(),
                self._streaming_request(),
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )
            assert is_streaming_response_result(result)
            await close_async_iterator(result)

        assert pool._leased == 0
        assert len(pool._idle) == 0

    async def test_inner_close_failure_is_swallowed_and_lease_released(self, middleware: GuardrailsMiddleware) -> None:
        """Pin the cleanup-masking guard in :meth:`_streaming_with_lease`.

        A buggy ``aclose()`` on the inner iterator from
        ``handle_streaming_output_check`` must NOT propagate from
        :meth:`_streaming_with_lease`'s ``finally`` block — the consumer
        already received the legitimate stream content; the cleanup
        failure must not retroactively turn a successful response into
        an error. Logged so the failure still surfaces in operator
        dashboards.

        Patches ``handle_streaming_output_check`` at the middleware
        boundary to return a controlled inner iterator whose ``aclose``
        raises — this isolates the OUTER cleanup path
        (``close_async_iterator(inner)`` in ``_streaming_with_lease``)
        from the inner iterator's own internal cleanup, so the test
        exercises exactly the swallow-and-log branch the fix introduces.
        """

        chunks_yielded = [
            {
                "id": "chatcmpl-x",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "served-model",
                "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            }
        ]

        class _LlmErrorOnCloseStream:
            """Async iterator that yields normally then fails on aclose()."""

            def __init__(self) -> None:
                self._items = iter(chunks_yielded)

            def __aiter__(self) -> "_LlmErrorOnCloseStream":
                return self

            async def __anext__(self) -> dict[str, Any]:
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration

            async def aclose(self) -> None:
                raise RuntimeError("inner aclose blew up")

        rails = self._make_rails(lambda generator, messages: generator)
        pool, patched_prepare = self._patch_with_real_pool(rails)

        with patch.object(middleware, "_prepare_lease", new=patched_prepare):
            with patch(
                "nemo_guardrails_plugin.middleware.handle_streaming_output_check",
                return_value=_LlmErrorOnCloseStream(),
            ):
                result = await _process_response(
                    middleware,
                    self._stream_one_chunk(),
                    self._streaming_request(),
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                )
                assert is_streaming_response_result(result)
                received = [c async for c in result]

        # All legitimate chunks delivered to the consumer; the close
        # failure is logged but does not produce an error chunk.
        assert received == chunks_yielded, (
            f"consumer should see all yielded chunks even when inner aclose fails; got {received}"
        )
        # Lease still released cleanly despite the cleanup failure.
        assert pool._leased == 0

    async def test_lease_aenter_failure_yields_streaming_error_chunk(self, middleware: GuardrailsMiddleware) -> None:
        @asynccontextmanager
        async def failing_lease(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
            raise RuntimeError("KB build failed")
            yield  # pragma: no cover - unreachable, satisfies type checker

        cache = MagicMock()
        cache.lease = failing_lease
        stable = StableRailsConfig(rails=MagicMock(), content_hash="hash-stub", embedding_model_id=None)

        async def patched_prepare(*_args: Any, **_kwargs: Any) -> tuple[Any, Any, Any, Any]:
            return cache, stable, Provenance("ws/name@ts"), MagicMock()

        with patch.object(middleware, "_prepare_lease", new=patched_prepare):
            result = await _process_response(
                middleware,
                self._stream_one_chunk(),
                self._streaming_request(),
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )
            assert is_streaming_response_result(result)
            chunks: list[dict[str, Any]] = [c async for c in result]

        assert len(chunks) == 1
        assert chunks[0]["error"]["type"] == "RuntimeError"
        assert "KB build failed" in chunks[0]["error"]["message"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.parametrize(
        ("log_level", "expected_library_level"),
        [
            ("INFO", logging.WARNING),
            ("WARN", logging.WARNING),
            ("DEBUG", logging.DEBUG),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ],
    )
    async def test_on_startup_configures_guardrails_library_logging(
        self,
        log_level: str,
        expected_library_level: int,
    ) -> None:
        library_logger = logging.getLogger(GUARDRAILS_LIBRARY_LOGGER_NAME)
        original_level = library_logger.level
        instance = GuardrailsMiddleware()

        mock_client = AsyncMock(spec=AsyncNemoClient)
        try:
            instance._inject_platform_client(mock_client)
            with (
                patch(
                    "nemo_guardrails_plugin.middleware.get_common_service_config",
                    return_value=SimpleNamespace(log_level=log_level),
                ),
            ):
                await instance.on_startup()

            assert library_logger.level == expected_library_level
        finally:
            library_logger.setLevel(original_level)
            if instance._client is not None:
                await instance.on_shutdown()

    async def test_on_shutdown_clears_client_and_closes_cache(self) -> None:
        instance = GuardrailsMiddleware()
        instance._inject_cache(MagicMock())

        mock_client = AsyncMock(spec=AsyncNemoClient)
        instance._inject_platform_client(mock_client)
        await instance.on_startup()

        assert instance._rails_cache is not None
        assert instance._stable_cache is not None
        await instance.on_shutdown()

        assert instance._client is None
        assert instance._rails_cache is None
        assert instance._stable_cache is None

    async def test_on_shutdown_clears_client_when_cache_close_raises(self) -> None:
        instance = GuardrailsMiddleware()
        instance._inject_cache(MagicMock())

        mock_client = AsyncMock(spec=AsyncNemoClient)
        instance._inject_platform_client(mock_client)
        await instance.on_startup()

        assert instance._rails_cache is not None
        with patch.object(instance._rails_cache, "close", new=AsyncMock(side_effect=RuntimeError("cache boom"))):
            with pytest.raises(RuntimeError, match="cache boom"):
                await instance.on_shutdown()

        assert instance._client is None
        assert instance._rails_cache is None
        assert instance._stable_cache is None


# ---------------------------------------------------------------------------
# Lifecycle hooks: warming via the same resolver methods IGW uses
# ---------------------------------------------------------------------------


def _make_virtual_model(
    *,
    request_calls: list[dict[str, Any]] | None = None,
    response_calls: list[dict[str, Any]] | None = None,
    workspace: str = "ws",
    name: str = "vm-1",
) -> Any:
    from nemo_platform_plugin.inference_middleware_models import MiddlewareCall, VirtualModel

    return VirtualModel(
        workspace=workspace,
        name=name,
        request_middleware=[MiddlewareCall.model_validate(c) for c in (request_calls or [])],
        response_middleware=[MiddlewareCall.model_validate(c) for c in (response_calls or [])],
    )


def _guardrail_call(config_id: str) -> dict[str, Any]:
    return {
        "name": "nemo-guardrails",
        "config_type": GUARDRAILS_PLUGIN_CONFIG_TYPE,
        "config_id": config_id,
    }


def _inline_guardrail_call(rails_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "nemo-guardrails",
        "config_type": GUARDRAILS_PLUGIN_CONFIG_TYPE,
        "config": rails_dict,
    }


@pytest_asyncio.fixture
async def lifecycle_cache(middleware: GuardrailsMiddleware) -> Any:
    """Patch ``cache.warm`` for upsert lifecycle tests.

    Lifecycle tests assert *which calls* the middleware makes into the cache —
    they don't exercise the actual KB build. Tracks every task ``warm``
    schedules and awaits them on fixture teardown.

    Yields a ``SimpleNamespace`` with ``warm`` (the mock) and ``await_warms``
    (an async helper that drains in-flight warm tasks before assertions).
    """
    assert middleware._rails_cache is not None
    cache = middleware._rails_cache

    async def _noop() -> None:
        return None

    warm_tasks: list[asyncio.Task[None]] = []

    def _track_warm(*_a: Any, **_kw: Any) -> asyncio.Task[None]:
        task = asyncio.create_task(_noop())
        warm_tasks.append(task)
        return task

    async def await_warms() -> None:
        if warm_tasks:
            await asyncio.gather(*warm_tasks, return_exceptions=True)

    warm_mock = MagicMock(side_effect=_track_warm)
    with patch.object(cache, "warm", new=warm_mock):
        yield SimpleNamespace(warm=warm_mock, await_warms=await_warms)

    if warm_tasks:
        await asyncio.gather(*warm_tasks, return_exceptions=True)


class TestVirtualModelLifecycle:
    async def test_upsert_warms_cache_for_each_unique_config(
        self, middleware: GuardrailsMiddleware, lifecycle_cache: Any
    ) -> None:
        assert middleware._client is not None
        entity = _make_entity(workspace="ws", name="guard-A")

        with _patch_get_guardrail_config(return_value=entity):
            vm = _make_virtual_model(request_calls=[_guardrail_call("ws/guard-A")])
            await middleware.on_virtual_model_upserted(vm)
            await lifecycle_cache.await_warms()

        lifecycle_cache.warm.assert_called_once()
        warmed_stable = lifecycle_cache.warm.call_args.args[0]
        assert isinstance(warmed_stable, StableRailsConfig)
        # Provenance is forwarded to the cache as a kw-arg for diagnostics.
        provenance = lifecycle_cache.warm.call_args.kwargs["provenance"]
        assert isinstance(provenance, Provenance)
        assert provenance.label == f"ws/guard-A@{entity.updated_at.isoformat()}"

    async def test_upsert_dedupes_within_a_single_vm(
        self, middleware: GuardrailsMiddleware, lifecycle_cache: Any
    ) -> None:
        """The same entity referenced twice in one VM warms once — dedup
        is by :attr:`StableRailsConfig.content_hash` at the warming layer,
        which catches both same-entity-twice (this test) and inline-vs-
        entity collisions where the rails happen to be identical."""
        assert middleware._client is not None
        entity = _make_entity(workspace="ws", name="guard-A")

        with _patch_get_guardrail_config(return_value=entity):
            vm = _make_virtual_model(
                request_calls=[_guardrail_call("ws/guard-A")],
                response_calls=[_guardrail_call("ws/guard-A")],
            )
            await middleware.on_virtual_model_upserted(vm)
            await lifecycle_cache.await_warms()

        lifecycle_cache.warm.assert_called_once()

    async def test_upsert_warms_distinct_configs_separately(
        self, middleware: GuardrailsMiddleware, lifecycle_cache: Any
    ) -> None:
        """Two configs with different ``content_hash`` warm independently.

        Distinct ``input_flows`` per entity guarantees distinct content
        hashes; without that, the dedup-by-hash path correctly merges
        them to one warm (covered by
        :meth:`test_upsert_dedupes_inline_against_entity_with_same_content`).
        """
        assert middleware._client is not None
        entity_a = _make_entity(workspace="ws", name="guard-A", input_flows=["check-a"])
        entity_b = _make_entity(workspace="ws", name="guard-B", input_flows=["check-b"])

        async def _resolve(*, name: str, workspace: str) -> GuardrailConfig:
            return entity_a if name == "guard-A" else entity_b

        with _patch_get_guardrail_config(side_effect=_resolve):
            vm = _make_virtual_model(
                request_calls=[
                    _guardrail_call("ws/guard-A"),
                    _guardrail_call("ws/guard-B"),
                ],
            )
            await middleware.on_virtual_model_upserted(vm)
            await lifecycle_cache.await_warms()

        warmed_labels = {args.kwargs["provenance"].label for args in lifecycle_cache.warm.call_args_list}
        assert warmed_labels == {
            f"ws/guard-A@{entity_a.updated_at.isoformat()}",
            f"ws/guard-B@{entity_b.updated_at.isoformat()}",
        }

    async def test_upsert_ignores_other_plugins(self, middleware: GuardrailsMiddleware, lifecycle_cache: Any) -> None:
        assert middleware._client is not None
        with _patch_get_guardrail_config() as mock_get:
            vm = _make_virtual_model(
                request_calls=[
                    {
                        "name": "some-other-plugin",
                        "config_type": "other_config",
                        "config_id": "ws/other",
                    }
                ],
            )
            await middleware.on_virtual_model_upserted(vm)
            await lifecycle_cache.await_warms()

        mock_get.assert_not_awaited()
        lifecycle_cache.warm.assert_not_called()

    async def test_upsert_swallows_resolution_errors(
        self, middleware: GuardrailsMiddleware, lifecycle_cache: Any
    ) -> None:
        """A failing resolution must not break the IGW polling cycle."""
        assert middleware._client is not None
        with _patch_get_guardrail_config(side_effect=RuntimeError("client down")):
            vm = _make_virtual_model(request_calls=[_guardrail_call("ws/guard-X")])
            # Must not raise; the next polling cycle retries.
            await middleware.on_virtual_model_upserted(vm)
            await lifecycle_cache.await_warms()

        lifecycle_cache.warm.assert_not_called()

    async def test_upsert_skips_when_config_id_has_been_deleted(
        self, middleware: GuardrailsMiddleware, lifecycle_cache: Any
    ) -> None:
        """A 404 from the client during warming must not raise — the same
        :class:`MiddlewareConfigNotFoundError` that IGW uses as the eviction
        signal would otherwise bubble up and stall the upsert hook for any
        other configs the VM references. ``_resolve_call`` swallows it the
        same way it swallows :class:`ValueError`."""
        assert middleware._client is not None
        not_found = _not_found_error()
        with _patch_get_guardrail_config(side_effect=not_found):
            vm = _make_virtual_model(request_calls=[_guardrail_call("ws/guard-deleted")])
            await middleware.on_virtual_model_upserted(vm)
            await lifecycle_cache.await_warms()

        lifecycle_cache.warm.assert_not_called()

    async def test_upsert_warms_inline_configs(self, middleware: GuardrailsMiddleware, lifecycle_cache: Any) -> None:
        """Inline configs warm at upsert time: the rails payload lives in
        ``MiddlewareCall.config`` and is fully resolvable without an entity
        lookup. The provenance label uses the ``<inline:label>`` form (no
        ``updated_at`` — inline configs have no revision identity).

        The inline payload here deliberately omits ``models`` to pin the
        IGW-Plugin invariant: a user posting the minimum-viable config
        (no main, no task models — just rails flows that reference
        actions resolved elsewhere) must warm cleanly. ``stabilize``
        coerces the missing field to ``[]`` at the platform→library
        boundary."""
        vm = _make_virtual_model(
            request_calls=[
                _inline_guardrail_call(
                    {
                        "name": "ad-hoc-1",
                        "rails": {"input": {"flows": ["custom check"]}},
                    }
                )
            ],
        )
        await middleware.on_virtual_model_upserted(vm)
        await lifecycle_cache.await_warms()

        lifecycle_cache.warm.assert_called_once()
        warmed_stable = lifecycle_cache.warm.call_args.args[0]
        assert isinstance(warmed_stable, StableRailsConfig)
        provenance = lifecycle_cache.warm.call_args.kwargs["provenance"]
        assert provenance.label == "<inline:ad-hoc-1>"

    async def test_upsert_dedupes_inline_against_entity_with_same_content(
        self, middleware: GuardrailsMiddleware, lifecycle_cache: Any
    ) -> None:
        """An inline config and an entity config with structurally identical
        rails dedupe to a single warm — the dedup key is
        :attr:`StableRailsConfig.content_hash`, not source identity. Pinning
        this so the dedup-by-hash can't silently regress to dedup-by-(entity-
        identity OR inline-label) (which wouldn't catch the cross-arm
        collision and would warm two pool slots that share an LLMRails)."""
        assert middleware._client is not None
        rails_payload: dict[str, Any] = {"rails": {"input": {"flows": ["custom check"]}}}
        entity = _make_entity(workspace="ws", name="guard-A")
        # Make the entity's rails identical to the inline payload so they
        # produce the same ``content_hash``.
        entity.data = SDKRailsConfig.model_validate(rails_payload)

        with _patch_get_guardrail_config(return_value=entity):
            vm = _make_virtual_model(
                request_calls=[
                    _guardrail_call("ws/guard-A"),
                    _inline_guardrail_call(rails_payload),
                ],
            )
            await middleware.on_virtual_model_upserted(vm)
            await lifecycle_cache.await_warms()

        lifecycle_cache.warm.assert_called_once()


# ---------------------------------------------------------------------------
# Error surfacing: how _run_rails internal failures reach the API boundary
# ---------------------------------------------------------------------------


class TestProcessRequestErrorSurfacing:
    """Pin the contract for how lease-setup errors map to HTTP status codes.

    Three error policies, by source of fault:

    - **Caller-shape errors** (``ValueError`` raised during eager lease
      setup — e.g. :func:`build_main_llm` rejecting a missing ``model``,
      or :func:`stabilize` rejecting a malformed inline config) are
      converted to :class:`InferenceMiddlewareError` with
      ``status_code=400`` at the plugin boundary
      (:meth:`_prepare_lease_with_503`). Wrapping locally is load-bearing
      because IGW only catches :class:`InferenceMiddlewareError` — a bare
      ``ValueError`` would surface as a 500 from FastAPI's default
      handler, mis-attributing a malformed request to the plugin.
    - **Caller-explicit errors** (:class:`InferenceMiddlewareError`
      raised from anywhere in the pipeline) propagate unchanged so a
      caller-set ``status_code`` (429, 401, 408, etc.) survives.
    - **Internal failures** below the lease boundary — cache not
      initialized, ``cache.lease`` build, KB build,
      :func:`run_generate_in_new_loop` — wrap as a 503
      :class:`InferenceMiddlewareUnavailableError` because the plugin
      cannot tell a transient outage apart from a misconfigured config
      without leaking cache internals into ``_run_rails``.

    Structural source-shape errors (empty ``updated_at``, missing
    ``data``) are caught at the resolver boundary in
    :class:`TestGetMiddlewareConfig` — they never reach
    ``process_request`` because the discriminated source union already
    encodes the validity preconditions.
    """

    async def test_missing_request_model_surfaces_as_400(self, middleware: GuardrailsMiddleware) -> None:
        """Caller-shape error (request body missing ``model``) surfaces as
        a 400 :class:`InferenceMiddlewareError`, not a 500 or 503. The
        ``ValueError`` raised by :func:`build_main_llm` is converted at
        the plugin boundary in :meth:`_prepare_lease_with_503`; pinning
        the resulting ``status_code`` here so a future refactor of the
        wrapping policy can't silently flip a 400 to a 503 (which would
        falsely suggest the plugin or upstream is unhealthy)."""
        request_body = {"messages": [{"role": "user", "content": "Hello"}]}  # no "model"

        with pytest.raises(InferenceMiddlewareError) as exc_info:
            await _process_request(middleware, request_body, {}, _entity_source())

        assert exc_info.value.status_code == 400
        # Confirms the failure originated in build_main_llm (not in
        # cache/stabilize), so future refactors that move the validation
        # accidentally trip the test rather than silently change semantics.
        # build_main_llm's own wording is "non-empty"; it's chained as
        # __cause__ below the InferenceMiddlewareError wrapper.
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "non-empty" in str(exc_info.value.__cause__)

    async def test_streaming_missing_request_model_surfaces_as_400(self, middleware: GuardrailsMiddleware) -> None:
        """Same caller-shape contract on the streaming branch: the eager
        portion of :meth:`_prepare_lease` runs before the response
        generator is returned, so the 400 reaches the caller as a
        synchronous error rather than getting wrapped in 503 or buried
        in a streaming error chunk."""
        request_body = {"messages": [{"role": "user", "content": "Hello"}], "stream": True}  # no "model"

        async def _stream() -> AsyncIterator[dict[str, Any]]:
            yield {"choices": []}

        with pytest.raises(InferenceMiddlewareError) as exc_info:
            await _process_response(
                middleware,
                _stream(),
                request_body,
                {},
                {},
                _entity_source(output_flows=["self check output"]),
            )

        assert exc_info.value.status_code == 400
        assert isinstance(exc_info.value.__cause__, ValueError)

    async def test_inference_middleware_error_inside_lease_preserves_status_code(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        """An :class:`InferenceMiddlewareError` raised below the lease
        boundary (here: from inside ``run_generate_in_new_loop``) must
        propagate with its caller-set ``status_code`` intact.

        ``_run_rails`` wraps everything else into 503; the explicit
        ``except InferenceMiddlewareError: raise`` line is what keeps a
        429 / 401 / 408 from being flattened to 503. Without this test,
        the re-raise can be removed without any test failing.
        """
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }

        explicit = InferenceMiddlewareError("rate limited", status_code=429)

        def _raise(*_args: Any, **_kwargs: Any) -> Any:
            raise explicit

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch("nemo_guardrails_plugin.middleware.run_generate_in_new_loop", side_effect=_raise):
                with pytest.raises(InferenceMiddlewareError) as exc_info:
                    await _process_request(middleware, request_body, {}, _entity_source())

        assert exc_info.value is explicit
        assert exc_info.value.status_code == 429

    async def test_run_rails_keeps_platform_header_context_active_during_generation(
        self, middleware: GuardrailsMiddleware
    ) -> None:
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }
        active = False
        observed_active: list[bool] = []

        @contextmanager
        def _platform_context(_sdk: Any):
            nonlocal active
            active = True
            try:
                yield
            finally:
                active = False

        def _generate(*_args: Any, **_kwargs: Any) -> GenerationResponse:
            observed_active.append(active)
            return _make_generation_response(is_blocked=False)

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch("nemo_guardrails_plugin.middleware.platform_headers_context", new=_platform_context):
                with patch("nemo_guardrails_plugin.middleware.run_generate_in_new_loop", side_effect=_generate):
                    await _process_request(middleware, request_body, {}, _entity_source())

        assert observed_active == [True]
        assert active is False

    async def test_run_rails_uses_request_scoped_client_headers(self, middleware: GuardrailsMiddleware) -> None:
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }
        request_headers = {
            "traceparent": "00-request-trace",
            "X-NMP-Principal-On-Behalf-Of": "user:alice",
        }
        ctx = _make_ctx(request_body)
        ctx.request_nemo_client = _request_scoped_client(middleware, request_headers)
        observed_headers: list[dict[str, str]] = []

        def _generate(*_args: Any, **_kwargs: Any) -> GenerationResponse:
            observed_headers.append(dict(get_request_headers()))
            return _make_generation_response(is_blocked=False)

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch("nemo_guardrails_plugin.middleware.run_generate_in_new_loop", side_effect=_generate):
                await _process_request(middleware, request_body, {}, _entity_source(), ctx=ctx)

        assert observed_headers == [request_headers]

    async def test_runtime_error_from_prepare_lease_wraps_to_503(self, middleware: GuardrailsMiddleware) -> None:
        """A non-caller-shape failure during eager lease setup (here: a
        ``RuntimeError`` simulating "cache not initialized" or a
        ``stabilize`` failure) wraps to 503 in **both**
        :meth:`_run_rails` and the streaming branch.

        Pins the unification: before the ``_prepare_lease_with_503``
        helper, ``_run_rails`` let these escape as raw ``RuntimeError`` →
        IGW 500, while the streaming branch already wrapped them to 503.
        """
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }

        async def _llm_error(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("cache exploded")

        with patch.object(middleware, "_prepare_lease", new=_llm_error):
            with pytest.raises(InferenceMiddlewareUnavailableError) as exc_info:
                await _process_request(middleware, request_body, {}, _entity_source())

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "cache exploded" in str(exc_info.value.__cause__)

    async def test_client_not_initialized_wraps_to_503(self, middleware: GuardrailsMiddleware) -> None:
        """Client detached after ``on_shutdown`` must map to 503, not a raw
        ``RuntimeError``, on both the non-streaming and streaming paths.

        ``_ensure_client`` lives inside :meth:`_prepare_lease_with_503` so the
        same lifecycle boundary that wraps cache/stabilize failures also
        covers client validation — without this, a shutdown race escaped as
        IGW 500.
        """
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }

        middleware._client = None

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with pytest.raises(InferenceMiddlewareUnavailableError) as exc_info:
                await _process_request(middleware, request_body, {}, _entity_source())

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "client is not initialized" in str(exc_info.value.__cause__)

        async def _stream() -> AsyncIterator[dict[str, Any]]:
            yield {"choices": []}

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with pytest.raises(InferenceMiddlewareUnavailableError) as stream_exc:
                await _process_response(
                    middleware,
                    _stream(),
                    {**request_body, "stream": True},
                    {},
                    {},
                    _entity_source(output_flows=["self check output"]),
                )

        assert isinstance(stream_exc.value.__cause__, RuntimeError)
        assert "client is not initialized" in str(stream_exc.value.__cause__)

    async def test_bracketed_upstream_400_from_rail_task_llm_preserved(self, middleware: GuardrailsMiddleware) -> None:
        """A rail-task LLM call (e.g. a vision-safety judge, via
        ``langchain_nvidia_ai_endpoints``) that rejects the request shape
        with an upstream 400 must surface as a 400, not a 503.

        The outer ``LLMCallException`` nemoguardrails raises never carries
        the status prefix itself — only its ``inner_exception`` does — so
        this pins that the middleware unwraps to ``inner_exception`` rather
        than checking the outer exception alone.
        """
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }

        def _llm_error(*_args: Any, **_kwargs: Any) -> Any:
            inner = Exception(  # noqa: TRY002 - mirrors langchain_nvidia_ai_endpoints._format_error
                '[400] Unknown Error {"object":"error","message":'
                '"At most 1 image(s) may be provided in one request.",'
                '"type":"BadRequestError","param":null,"code":400}'
            )
            raise LLMCallException(inner, detail="Error invoking LLM (model=vision-judge)") from inner

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch("nemo_guardrails_plugin.middleware.run_generate_in_new_loop", side_effect=_llm_error):
                with pytest.raises(InferenceMiddlewareError) as exc_info:
                    await _process_request(middleware, request_body, {}, _entity_source())

        assert not isinstance(exc_info.value, InferenceMiddlewareUnavailableError)
        assert exc_info.value.status_code == 400
        # The outer LLMCallException's ``detail`` (which model/provider/endpoint
        # failed) is preserved alongside the sanitized upstream message, not
        # discarded in favor of the inner exception's message alone.
        assert exc_info.value.detail == (
            "Error invoking LLM (model=vision-judge): At most 1 image(s) may be provided in one request."
        )

    async def test_openai_status_error_status_code_preserved(self, middleware: GuardrailsMiddleware) -> None:
        """An ``openai.APIStatusError`` subclass (``BadRequestError`` here) has
        a genuine ``status_code`` attribute, so it's picked up directly with
        no message parsing needed."""
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }

        def _llm_error(*_args: Any, **_kwargs: Any) -> Any:
            response = httpx.Response(422, request=httpx.Request("POST", "http://example.test"))
            inner = openai.BadRequestError("Unsupported parameter: foo", response=response, body=None)
            raise LLMCallException(inner, detail="Error invoking LLM") from inner

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch("nemo_guardrails_plugin.middleware.run_generate_in_new_loop", side_effect=_llm_error):
                with pytest.raises(InferenceMiddlewareError) as exc_info:
                    await _process_request(middleware, request_body, {}, _entity_source())

        assert not isinstance(exc_info.value, InferenceMiddlewareUnavailableError)
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail == "Error invoking LLM: Unsupported parameter: foo"

    async def test_upstream_5xx_also_propagated_verbatim(self, middleware: GuardrailsMiddleware) -> None:
        """A ``[5xx]``-prefixed upstream failure is propagated as-is, same as
        a 4xx — the middleware's generic 503 fallback is now reserved for
        failures with no recoverable upstream status at all."""
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }

        def _llm_error(*_args: Any, **_kwargs: Any) -> Any:
            raise Exception("[503] Service temporarily overloaded")  # noqa: TRY002

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch("nemo_guardrails_plugin.middleware.run_generate_in_new_loop", side_effect=_llm_error):
                with pytest.raises(InferenceMiddlewareError) as exc_info:
                    await _process_request(middleware, request_body, {}, _entity_source())

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Service temporarily overloaded"

    async def test_no_recoverable_status_still_wraps_to_generic_503(self, middleware: GuardrailsMiddleware) -> None:
        """When the failure carries no recoverable upstream status at all
        (e.g. a plain library bug below the lease boundary), the middleware
        still falls back to its own generic 503 with the fixed ``error_msg``."""
        request_body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "ws/llama",
        }

        def _llm_error(*_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("something went wrong")

        with patch.object(middleware, "_prepare_lease", new=_patch_prepare_lease()):
            with patch("nemo_guardrails_plugin.middleware.run_generate_in_new_loop", side_effect=_llm_error):
                with pytest.raises(InferenceMiddlewareUnavailableError) as exc_info:
                    await _process_request(middleware, request_body, {}, _entity_source())

        assert exc_info.value.status_code == 503
        # Detail is the generic, fixed error_msg — not the raw upstream text.
        # extract_upstream_error() returned None here, so _run_rails's
        # fallback (not the status-preserving branch) ran.
        assert exc_info.value.detail == "Failed to run input rails"
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "something went wrong" in str(exc_info.value.__cause__)
