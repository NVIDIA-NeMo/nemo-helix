# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import typing
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any
from unittest.mock import MagicMock

import anthropic.types as anthropic_types
import anthropic.types.beta as anthropic_beta
import openai.types.chat as openai_chat
import openai.types.responses as openai_responses
import pytest
from nemo_evals.gateway.middleware.closed_book import (
    CONFIG_TYPE,
    MIDDLEWARE_NAME,
    REMOVED_TOOLS_STATE_KEY,
    ClosedBookConfig,
    ClosedBookMiddleware,
    RouterCredentialCheck,
    client_tool_label,
    is_self_searching_model,
    strip_hosted_tools,
)
from nemo_evals.gateway.policy import ModelAccess
from nemo_evals.gateway.tokens import RouterKeyClaims, TrialTokenReader, unsigned_token
from nemo_helix_plugin.client.oidc import decode_jwt_claims, generate_unsigned_jwt
from nemo_helix_plugin.inference_middleware import (
    InferenceMiddlewareCacheAccessor,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceRequest,
    ModelProviderInferenceTarget,
)


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "function", "name": "exec_command"},
        {"type": "custom", "name": "apply_patch"},
        {"type": "namespace", "name": "mcp__github", "tools": []},
        {"type": "local_shell"},
        {"type": "bash_20250124", "name": "bash"},
        {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
        {"type": "computer_use_preview", "display_width": 1024, "display_height": 768, "environment": "linux"},
        {"type": "computer_20250124", "name": "computer", "display_width_px": 1024, "display_height_px": 768},
        {"name": "lookup", "input_schema": {"type": "object"}},
    ],
)
def test_client_executed_tools_are_kept(tool: dict) -> None:
    assert client_tool_label(tool) is None


@pytest.mark.parametrize(
    ("tool", "label"),
    [
        ({"type": "web_search"}, "web_search"),
        ({"type": "web_search_preview_2025_03_11"}, "web_search_preview_2025_03_11"),
        ({"type": "web_search_20250305", "name": "web_search"}, "web_search_20250305"),
        ({"type": "web_fetch_20250910"}, "web_fetch_20250910"),
        ({"type": "mcp", "server_url": "https://example.com"}, "mcp"),
        ({"type": "mcp_toolset", "mcp_server_name": "x"}, "mcp_toolset"),
        ({"type": "x_search"}, "x_search"),
        ({"type": "file_search"}, "file_search"),
        ({"type": "code_interpreter"}, "code_interpreter"),
        ({"type": "image_generation"}, "image_generation"),
        ({"googleSearch": {}}, "googleSearch"),
        ({"enterpriseWebSearch": {}}, "enterpriseWebSearch"),
        ({"name": "web_search"}, "name"),
        ({}, "untyped"),
        ("web_search", "str"),
    ],
)
def test_unrecognised_tools_are_stripped_and_named(tool: object, label: str) -> None:
    """Anything not known to run on the client is treated as provider-executed (fail closed)."""
    assert client_tool_label(tool) == label


def test_codex_request_keeps_client_tools_and_drops_web_search() -> None:
    function_tool = {"type": "function", "name": "exec_command", "parameters": {}}
    namespace_tool = {"type": "namespace", "name": "mcp__x", "tools": []}
    body = {
        "model": "m",
        "tools": [function_tool, {"type": "web_search", "external_web_access": False}, namespace_tool],
        "tool_choice": "auto",
    }

    assert strip_hosted_tools(body) == ["web_search"]
    assert body["tools"] == [function_tool, namespace_tool]
    assert body["tool_choice"] == "auto"


def test_hosted_tools_nested_in_a_namespace_are_stripped() -> None:
    body = {"tools": [{"type": "namespace", "name": "ns", "tools": [{"type": "web_search"}, {"type": "function"}]}]}

    assert strip_hosted_tools(body) == ["web_search"]
    assert body["tools"][0]["tools"] == [{"type": "function"}]


def test_tools_carried_inside_responses_input_are_stripped() -> None:
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "additional_tools", "tools": [{"type": "web_search"}, {"type": "function"}]},
        ]
    }

    assert strip_hosted_tools(body) == ["web_search"]
    assert body["input"][1]["tools"] == [{"type": "function"}]


@pytest.mark.parametrize(
    "key", ["container", "mcp_servers", "plugins", "prompt", "search_parameters", "web_search_options"]
)
def test_provider_feature_keys_are_removed(key: str) -> None:
    body = {"model": "m", key: {"anything": True}}

    assert strip_hosted_tools(body) == [key]
    assert body == {"model": "m"}


@pytest.mark.parametrize("prompt", ["hello", ["hello", "world"], [[1, 2, 3]]])
def test_completions_prompt_is_kept(prompt: object) -> None:
    """A Completions ``prompt`` is the caller's input; only a Responses stored-prompt object can add tools."""
    body = {"model": "m", "prompt": prompt}

    assert strip_hosted_tools(body) == []
    assert body == {"model": "m", "prompt": prompt}


def test_forced_hosted_tool_choice_falls_back_to_auto_per_wire_format() -> None:
    function_tool = {"type": "function", "name": "f"}
    anthropic_tool = {"name": "f", "input_schema": {}}
    openai = {"tools": [{"type": "web_search_preview"}, function_tool], "tool_choice": {"type": "web_search_preview"}}
    anthropic = {
        "tools": [{"type": "web_search_20250305", "name": "web_search"}, anthropic_tool],
        "tool_choice": {"type": "tool", "name": "web_search"},
    }

    strip_hosted_tools(openai)
    strip_hosted_tools(anthropic, anthropic=True)

    assert openai == {"tools": [function_tool], "tool_choice": "auto"}
    assert anthropic == {"tools": [anthropic_tool], "tool_choice": {"type": "auto"}}


@pytest.mark.parametrize(
    "body",
    [
        {"tools": [{"type": "web_search"}], "tool_choice": "required", "parallel_tool_calls": True},
        {"tools": [{"type": "web_search_preview"}], "tool_choice": {"type": "web_search_preview"}},
        {"tools": [{"type": "web_search_20250305", "name": "web_search"}], "tool_choice": {"type": "any"}},
    ],
)
def test_removing_every_tool_drops_the_tool_settings(body: dict) -> None:
    """Providers reject an empty `tools` list, and a `tool_choice` or `parallel_tool_calls` without tools."""
    strip_hosted_tools(body)

    assert body == {}


def test_tool_choice_naming_a_kept_tool_or_a_mode_is_preserved() -> None:
    named = {
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }
    mode = {"tools": [], "tool_choice": {"type": "any"}}

    assert strip_hosted_tools(named) == []
    assert strip_hosted_tools(mode, anthropic=True) == []
    assert named["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert mode["tool_choice"] == {"type": "any"}


@pytest.mark.parametrize("tool", [{"type": "shell"}, {"type": "shell", "environment": {"type": "local"}}])
def test_shell_tool_running_locally_is_kept(tool: dict) -> None:
    assert client_tool_label(tool) is None


@pytest.mark.parametrize(
    ("environment", "label"),
    [
        ({"type": "container_auto", "network_policy": {"type": "allowlist"}}, "shell:container_auto"),
        ({"type": "container_reference", "container_id": "cntr_1"}, "shell:container_reference"),
        ("container_auto", "shell:str"),
    ],
)
def test_shell_tool_running_in_a_provider_container_is_stripped(environment: object, label: str) -> None:
    assert client_tool_label({"type": "shell", "environment": environment}) == label


def test_forced_choice_of_a_kept_client_tool_type_is_preserved() -> None:
    body = {"tools": [{"type": "apply_patch"}], "tool_choice": {"type": "apply_patch"}}

    assert strip_hosted_tools(body) == []
    assert body["tool_choice"] == {"type": "apply_patch"}


def test_forced_choice_of_a_stripped_shell_falls_back_to_auto() -> None:
    body = {
        "tools": [{"type": "shell", "environment": {"type": "container_auto"}}, {"type": "apply_patch"}],
        "tool_choice": {"type": "shell"},
    }

    assert strip_hosted_tools(body) == ["shell:container_auto", "tool_choice:shell"]
    assert body == {"tools": [{"type": "apply_patch"}], "tool_choice": "auto"}


def test_malformed_function_field_is_tolerated() -> None:
    body = {
        "tools": [{"type": "function", "function": "bash"}],
        "tool_choice": {"type": "function", "function": "bash"},
    }

    assert strip_hosted_tools(body) == ["tool_choice:function"]
    assert body == {"tools": [{"type": "function", "function": "bash"}], "tool_choice": "auto"}


def test_allowed_tools_choice_loses_hosted_entries() -> None:
    body = {
        "tool_choice": {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{"type": "function", "name": "lookup"}, {"type": "web_search"}],
        }
    }

    assert strip_hosted_tools(body) == ["web_search"]
    assert body["tool_choice"]["tools"] == [{"type": "function", "name": "lookup"}]


def test_allowed_tools_choice_left_empty_falls_back_to_auto() -> None:
    body = {
        "tools": [{"type": "function", "name": "lookup"}, {"type": "web_search"}],
        "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [{"type": "web_search"}]},
    }

    strip_hosted_tools(body)

    assert body == {"tools": [{"type": "function", "name": "lookup"}], "tool_choice": "auto"}


def test_request_without_tools_is_unchanged() -> None:
    body = {"model": "m", "input": "hello", "tool_choice": "none"}

    assert strip_hosted_tools(body) == []
    assert body == {"model": "m", "input": "hello", "tool_choice": "none"}


@pytest.mark.parametrize(
    "model",
    [
        "perplexity/perplexity/sonar",
        "perplexity/perplexity/sonar-deep-research",
        "azure/openai/gpt-4o-search-preview",
        "azure/openai/o4-mini-deep-research",
        "openai/gpt-5:online",
        "openai/gpt-5-search-api",
        "Perplexity/Llama-3-Sonar-Large",
    ],
)
def test_models_that_search_on_every_call_are_flagged(model: str) -> None:
    """These search without any tool in the request, so stripping tools cannot close them."""
    assert is_self_searching_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "azure/openai/gpt-5.6-luna",
        "azure/anthropic/claude-opus-5",
        "nvidia/google/gemma-4-31b-it",
        "my-research-model",
        "nvidia/searchless-chess",
        None,
    ],
)
def test_ordinary_models_are_not_flagged(model: object) -> None:
    assert not is_self_searching_model(model)


def _target(
    served_model_name: str,
    default_extra_body: dict[str, object] | None = None,
    required_extra_body: dict[str, object] | None = None,
    outbound_headers: dict[str, str] | None = None,
) -> ModelProviderInferenceTarget:
    return ModelProviderInferenceTarget(
        model_provider_gateway_url="http://igw/provider",
        served_model_name=served_model_name,
        default_extra_body=default_extra_body or {},
        required_extra_body=required_extra_body or {},
        outbound_headers=outbound_headers or {},
    )


def _middleware(targets: dict[str, ModelProviderInferenceTarget] | None = None) -> ClosedBookMiddleware:
    targets = targets or {}

    def resolve(model_entity_id: str, append_v1_suffix: bool = True) -> ModelProviderInferenceTarget:
        return targets[model_entity_id]

    cache = MagicMock(spec=InferenceMiddlewareCacheAccessor)
    cache.get_inference_url_and_model.side_effect = resolve
    middleware = ClosedBookMiddleware()
    middleware._inject_cache(cache)
    return middleware


def _request(path: str, body: dict) -> tuple[InferenceMiddlewareContext, InferenceRequest]:
    request = InferenceRequest(body=body, headers={}, path=path)
    ctx = InferenceMiddlewareContext(
        request_id="r1",
        virtual_model_name="closed-book",
        workspace="ws",
        original_request=InferenceRequest(body=dict(body), headers={}, path=path),
    )
    return ctx, request


async def test_responses_request_is_stripped_and_removed_tools_are_recorded() -> None:
    middleware = _middleware({"ws/luna": _target("azure/openai/gpt-5.6-luna")})
    ctx, request = _request(
        "v1/responses",
        {"model": "ws/luna", "stream": True, "tools": [{"type": "web_search"}, {"type": "function", "name": "f"}]},
    )

    result = await middleware.process_request(ctx, request, ClosedBookConfig())

    assert result.body["tools"] == [{"type": "function", "name": "f"}]
    assert result.body["stream"] is True
    assert ctx.state(MIDDLEWARE_NAME).get(REMOVED_TOOLS_STATE_KEY) == ["web_search"]


async def test_messages_request_falls_back_to_anthropic_auto_tool_choice() -> None:
    middleware = _middleware({"ws/opus": _target("azure/anthropic/claude-opus-5")})
    ctx, request = _request(
        "/v1/messages",
        {
            "model": "ws/opus",
            "tools": [{"type": "web_search_20250305", "name": "web_search"}, {"name": "f", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "web_search"},
        },
    )

    result = await middleware.process_request(ctx, request, ClosedBookConfig())

    assert result.body["tool_choice"] == {"type": "auto"}


async def test_request_with_nothing_to_strip_records_an_empty_list() -> None:
    middleware = _middleware({"ws/luna": _target("azure/openai/gpt-5.6-luna")})
    ctx, request = _request("v1/chat/completions", {"model": "ws/luna", "messages": []})

    await middleware.process_request(ctx, request, ClosedBookConfig())

    assert ctx.state(MIDDLEWARE_NAME).get(REMOVED_TOOLS_STATE_KEY) == []


@pytest.mark.parametrize(
    ("model", "served"),
    [
        ("ws/sonar", "azure/openai/gpt-5.6-luna"),
        ("ws/web-model", "perplexity/llama-3-large"),
    ],
)
async def test_self_searching_model_is_refused_by_entity_or_served_name(model: str, served: str) -> None:
    """A model entity's name need not reveal the provider model it serves, so both are checked."""
    middleware = _middleware({model: _target(served)})
    ctx, request = _request("v1/responses", {"model": model, "input": "hi"})

    with pytest.raises(InferenceMiddlewareError) as exc_info:
        await middleware.process_request(ctx, request, ClosedBookConfig())

    assert exc_info.value.status_code == 403


async def test_workspace_name_does_not_make_a_model_self_searching() -> None:
    """Only the model's own name says what it does; a workspace called `research` must not refuse its models."""
    middleware = _middleware({"research/luna": _target("azure/openai/gpt-5.6-luna")})
    ctx, request = _request("v1/responses", {"model": "research/luna", "input": "hi"})

    assert await middleware.process_request(ctx, request, ClosedBookConfig()) is request


@pytest.mark.parametrize(
    "history",
    [
        {"previous_response_id": "resp_1"},
        {"conversation": "conv_1"},
        {"conversation": {"id": "conv_1"}},
        {"input": [{"role": "user", "content": "hi"}, {"type": "item_reference", "id": "msg_1"}]},
    ],
)
async def test_provider_stored_history_is_refused(history: dict[str, object]) -> None:
    """Stored history may hold hosted search output from outside this route, and the middleware cannot read it."""
    middleware = _middleware({"ws/luna": _target("azure/openai/gpt-5.6-luna")})
    ctx, request = _request("v1/responses", {"model": "ws/luna", "input": "hi", **history})

    with pytest.raises(InferenceMiddlewareError) as exc_info:
        await middleware.process_request(ctx, request, ClosedBookConfig())

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize(
    ("default_extra_body", "required_extra_body"),
    [({"previous_response_id": "resp_1"}, None), (None, {"conversation": "conv_1"})],
)
async def test_provider_policy_adding_stored_history_is_refused(
    default_extra_body: dict[str, object] | None, required_extra_body: dict[str, object] | None
) -> None:
    """The gateway merges provider body policy after middleware runs, so the request alone is not enough."""
    target = _target("azure/openai/gpt-5.6-luna", default_extra_body, required_extra_body)
    middleware = _middleware({"ws/luna": target})
    ctx, request = _request("v1/responses", {"model": "ws/luna", "input": "hi"})

    with pytest.raises(InferenceMiddlewareError) as exc_info:
        await middleware.process_request(ctx, request, ClosedBookConfig())

    assert exc_info.value.status_code == 403


async def test_request_null_overriding_a_provider_default_history_is_allowed() -> None:
    """The gateway lets the request override `default_extra_body`, so a null there is what the provider gets."""
    target = _target("azure/openai/gpt-5.6-luna", default_extra_body={"previous_response_id": "resp_1"})
    middleware = _middleware({"ws/luna": target})
    ctx, request = _request("v1/responses", {"model": "ws/luna", "input": "hi", "previous_response_id": None})

    assert await middleware.process_request(ctx, request, ClosedBookConfig()) is request


async def test_null_provider_stored_history_is_allowed() -> None:
    middleware = _middleware({"ws/luna": _target("azure/openai/gpt-5.6-luna")})
    ctx, request = _request(
        "v1/responses", {"model": "ws/luna", "input": "hi", "previous_response_id": None, "conversation": None}
    )

    assert await middleware.process_request(ctx, request, ClosedBookConfig()) is request


@pytest.mark.parametrize("model", ["ws/unregistered", "ws/base&adapters/ws/adapter", None])
async def test_model_without_a_resolvable_provider_is_refused(model: object) -> None:
    """Closed-book cannot vouch for a model whose served name it cannot see, so it fails closed."""
    middleware = _middleware()
    ctx, request = _request("v1/responses", {"model": model, "input": "hi"})

    with pytest.raises(InferenceMiddlewareError) as exc_info:
        await middleware.process_request(ctx, request, ClosedBookConfig())

    assert exc_info.value.status_code == 403


@pytest.mark.parametrize(
    ("default_extra_body", "required_extra_body"),
    [
        ({"web_search_options": {}}, None),
        ({"tools": [{"type": "web_search"}]}, None),
        (None, {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
        (None, {"mcp_servers": [{"url": "https://example.com"}]}),
        (None, {"model": "perplexity/sonar"}),
    ],
)
async def test_provider_that_adds_provider_side_search_is_refused(
    default_extra_body: dict[str, object] | None, required_extra_body: dict[str, object] | None
) -> None:
    """The gateway merges provider body policy after middleware runs, so it could restore what was stripped."""
    target = _target("azure/openai/gpt-5.6-luna", default_extra_body, required_extra_body)
    middleware = _middleware({"ws/luna": target})
    ctx, request = _request("v1/responses", {"model": "ws/luna", "input": "hi"})

    with pytest.raises(InferenceMiddlewareError) as exc_info:
        await middleware.process_request(ctx, request, ClosedBookConfig())

    assert exc_info.value.status_code == 403


async def test_provider_body_policy_without_hosted_tools_is_allowed() -> None:
    target = _target(
        "azure/openai/gpt-5.6-luna",
        default_extra_body={
            "temperature": 0.2,
            "prompt": "Answer briefly.",
            "tools": [{"type": "function", "name": "f"}],
        },
        required_extra_body={"max_output_tokens": 128, "tool_choice": {"type": "function", "name": "lookup"}},
    )
    middleware = _middleware({"ws/luna": target})
    ctx, request = _request("v1/responses", {"model": "ws/luna", "input": "hi"})

    assert await middleware.process_request(ctx, request, ClosedBookConfig()) is request


@pytest.mark.parametrize(
    ("targets", "model", "reason"),
    [
        ({"ws/sonar": _target("perplexity/sonar")}, "ws/sonar", "self_searching_model"),
        (
            {"ws/luna": _target("azure/openai/gpt-5.6-luna", default_extra_body={"web_search_options": {}})},
            "ws/luna",
            "provider_adds_hosted_tools",
        ),
        ({}, "ws/unregistered", "unresolved_model"),
        ({"ws/luna": _target("azure/openai/gpt-5.6-luna")}, "ws/luna", "provider_stored_history"),
    ],
)
async def test_refusal_is_logged_with_its_reason(
    targets: dict[str, ModelProviderInferenceTarget], model: str, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Refusals are the security events an operator needs to find in the gateway logs."""
    body = {"model": model, "input": "hi", "previous_response_id": "resp_1"}
    ctx, request = _request("v1/responses", body)

    with (
        caplog.at_level(logging.WARNING, logger="nemo_evals.gateway.middleware.closed_book"),
        pytest.raises(InferenceMiddlewareError),
    ):
        await _middleware(targets).process_request(ctx, request, ClosedBookConfig())

    (record,) = [r for r in caplog.records if r.name == "nemo_evals.gateway.middleware.closed_book"]
    assert record.levelno == logging.WARNING
    assert (record.__dict__["reason"], record.__dict__["model"], record.__dict__["request_id"]) == (reason, model, "r1")


@pytest.mark.parametrize("config", [None, {}])
async def test_config_is_empty(config: object) -> None:
    assert await ClosedBookMiddleware().validate_middleware_config(CONFIG_TYPE, config) == ClosedBookConfig()


async def test_config_rejects_unknown_settings_and_config_types() -> None:
    middleware = ClosedBookMiddleware()

    with pytest.raises(ValueError):
        await middleware.validate_middleware_config(CONFIG_TYPE, {"allow": ["web_search"]})
    with pytest.raises(ValueError):
        await middleware.validate_middleware_config("open_book", {})


def test_runs_only_as_request_middleware() -> None:
    middleware = ClosedBookMiddleware()

    assert middleware.supports_middleware_phase("request")
    assert not middleware.supports_middleware_phase("response")
    assert not middleware.supports_middleware_phase("post_response")


def test_registered_as_inference_middleware() -> None:
    entries = [e for e in entry_points(group="nemo.inference_middleware") if e.name == MIDDLEWARE_NAME]

    assert entries
    assert all(entry.load() is ClosedBookMiddleware for entry in entries)


#: Tool families the provider runs itself. A new version of one of these is stripped as expected; a tool type in
#: neither this list nor the client allowlist fails the drift test below until someone classifies it.
_PROVIDER_EXECUTED_TOOL_FAMILY = re.compile(
    r"^(?:web_search|web_search_preview|web_fetch|code_execution|code_interpreter|programmatic_tool_calling|file_search"
    r"|image_generation|mcp|mcp_toolset|tool_search|tool_search_tool_(?:bm25|regex)|advisor)(?:_[0-9_]+)?$"
)


def _declared_tool_types(union: object) -> set[str]:
    types: set[str] = set()

    def literals(annotation: object) -> None:
        if typing.get_origin(annotation) is typing.Literal:
            types.update(str(value) for value in typing.get_args(annotation))
        for arg in typing.get_args(annotation):
            literals(arg)

    for member in typing.get_args(union) or (union,):
        literals(typing.get_type_hints(member, include_extras=True).get("type"))
    return types


@pytest.mark.parametrize(
    "union",
    [
        openai_responses.ToolParam,
        openai_chat.ChatCompletionToolUnionParam,
        anthropic_types.ToolUnionParam,
        anthropic_beta.BetaToolUnionParam,
    ],
    ids=["openai-responses", "openai-chat", "anthropic", "anthropic-beta"],
)
def test_every_sdk_tool_type_is_classified(union: object) -> None:
    """An SDK upgrade that adds a tool type fails here, so it is classified on purpose rather than by default.

    The middleware strips any type it doesn't recognise, so an unclassified client-executed tool would quietly
    vanish from closed-book runs.
    """
    declared = _declared_tool_types(union)

    unclassified = {
        tool_type
        for tool_type in declared
        if client_tool_label({"type": tool_type}) is not None and not _PROVIDER_EXECUTED_TOOL_FAMILY.match(tool_type)
    }

    assert declared
    assert not unclassified, f"classify these tool types as client- or provider-executed: {sorted(unclassified)}"


def test_no_tool_type_is_both_client_and_provider_executed() -> None:
    sdk_types = {
        tool_type
        for union in (openai_responses.ToolParam, anthropic_types.ToolUnionParam, anthropic_beta.BetaToolUnionParam)
        for tool_type in _declared_tool_types(union)
    }

    assert not {
        t for t in sdk_types if client_tool_label({"type": t}) is None and _PROVIDER_EXECUTED_TOOL_FAMILY.match(t)
    }


CLOSED_ROUTER_KEY = unsigned_token(RouterKeyClaims(model_access=ModelAccess.CLOSED, models=("ws/cb-strong",)))
OPEN_ROUTER_KEY = unsigned_token(RouterKeyClaims(model_access=ModelAccess.OPEN))
PLATFORM_KEY = generate_unsigned_jwt("user:alice")


class _Signed:
    """Stands in for the platform's signature check: trusts the claims of exactly the tokens it was given."""

    def __init__(self, *tokens: str) -> None:
        self._tokens = set(tokens)
        self.asked = 0

    async def __call__(self, token: str) -> Mapping[str, Any] | None:
        self.asked += 1
        return decode_jwt_claims(token) if token in self._tokens else None


def _router_check(signed: _Signed) -> RouterCredentialCheck:
    return RouterCredentialCheck(TrialTokenReader(signed), signed)


@pytest.mark.parametrize(
    ("headers", "signed", "refused"),
    [
        pytest.param({"Authorization": "Bearer nvapi-provider-key"}, (), False, id="a model provider's own key"),
        pytest.param({}, (), False, id="no credential"),
        pytest.param({"Authorization": f"Bearer {CLOSED_ROUTER_KEY}"}, (CLOSED_ROUTER_KEY,), False, id="a closed key"),
        pytest.param({"x-api-key": CLOSED_ROUTER_KEY}, (CLOSED_ROUTER_KEY,), False, id="a closed key as x-api-key"),
        pytest.param({"Authorization": f"Bearer {OPEN_ROUTER_KEY}"}, (OPEN_ROUTER_KEY,), True, id="an open key"),
        pytest.param({"Authorization": f"Bearer {CLOSED_ROUTER_KEY}"}, (), True, id="a closed key nobody signed"),
        pytest.param({"Authorization": f"Bearer {PLATFORM_KEY}"}, (PLATFORM_KEY,), True, id="a plain platform key"),
    ],
)
async def test_a_provider_calling_back_into_the_gateway_needs_a_closed_router_key(
    headers: dict[str, str], signed: tuple[str, ...], refused: bool
) -> None:
    """Closed-book can't see the calls a router makes back through the gateway, so the router's key must keep them closed."""
    middleware = _middleware({"ws/routed": _target("route", outbound_headers=headers)})
    middleware._router_check = _router_check(_Signed(*signed))
    ctx, request = _request("v1/responses", {"model": "ws/routed", "input": "hi"})

    if refused:
        with pytest.raises(InferenceMiddlewareError) as exc_info:
            await middleware.process_request(ctx, request, ClosedBookConfig())
        assert exc_info.value.status_code == 403
    else:
        assert await middleware.process_request(ctx, request, ClosedBookConfig()) is request


async def test_whether_a_provider_key_is_a_platform_key_is_remembered() -> None:
    signed = _Signed()
    check = _router_check(signed)

    for _ in range(3):
        assert await check.refusal({"Authorization": "Bearer nvapi-provider-key"}, "ws/model") is None

    assert signed.asked == 1
