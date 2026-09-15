# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`nemo_guardrails_plugin.rails` utilities.

Covers :func:`build_main_llm` (per-request main LLM construction reading
the pre-validated ``main_model_template`` from :class:`StableRailsConfig`),
:func:`run_generate_in_new_loop`, and the generation-log shaping helpers.

Tests for the discriminated source union, the stable-config transform, and
the LLMRails pool itself live in :mod:`test_llmrails_cache`.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from nemo_guardrails_plugin.llmrails_cache import (
    EntityGuardrailConfigSource,
    InlineGuardrailConfigSource,
    extract_output_rails_streaming_config,
    source_has_input_flows,
    source_has_output_flows,
    stabilize,
)
from nemo_guardrails_plugin.rails import (
    build_generate_async_options,
    build_guardrails_data,
    build_main_llm,
)
from nemo_guardrails_plugin.requests import (
    extract_log_options_from_request,
    extract_return_choice_from_request,
    parse_guardrails_request,
    sanitize_request_body_for_proxy,
)
from nemo_guardrails_plugin.responses import extract_response_content
from nemo_platform_plugin.guardrail.types import GenerationLogOptionsParam
from nemo_platform_plugin.guardrail.types import RailsConfig as SDKRailsConfig
from nemo_platform_plugin.inference_middleware import InferenceMiddlewareError, OpenAICompatibleInferenceTarget
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    GenerationLog,
    GenerationResponse,
    GenerationStats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rail(name: str = "self check input", rail_type: str = "input", stop: bool = False) -> ActivatedRail:
    return ActivatedRail(name=name, type=rail_type, stop=stop)


def _response(
    content: str | list = "Hello",
    activated_rails: list[ActivatedRail] | None = None,
    llm_calls: list | None = None,
    internal_events: list | None = None,
    colang_history: str | None = None,
    stats: GenerationStats | None = None,
    with_log: bool = True,
) -> GenerationResponse:
    if with_log:
        log_kwargs: dict[str, Any] = {
            "activated_rails": activated_rails if activated_rails is not None else [],
            "llm_calls": llm_calls,
            "internal_events": internal_events,
            "colang_history": colang_history,
        }
        if stats is not None:
            log_kwargs["stats"] = stats
        log = GenerationLog(**log_kwargs)
    else:
        log = None
    return GenerationResponse(response=content, log=log)


# ---------------------------------------------------------------------------
# parse_guardrails_request
# ---------------------------------------------------------------------------


class TestParseGuardrailsRequest:
    @pytest.mark.parametrize(
        ("guardrails", "expected_message"),
        [
            (
                {"config": {"rails": {}}},
                "Invalid guardrails request options: config: Extra inputs are not permitted",
            ),
            (
                {"config_id": "default/my-config"},
                "Invalid guardrails request options: config_id: Extra inputs are not permitted",
            ),
            (
                {"unknown": True},
                "Invalid guardrails request options: unknown: Extra inputs are not permitted",
            ),
            (
                {"options": {"rails": {"input": False}}},
                "Invalid guardrails request options: options.rails: Extra inputs are not permitted",
            ),
            (
                {"options": {"log": {"unknown": True}}},
                "Invalid guardrails request options: options.log.unknown: Extra inputs are not permitted",
            ),
            (
                {"options": {"log": {"unknown": True}}, "return_choice": 1},
                "Invalid guardrails request options: 2 validation errors. "
                "options.log.unknown: Extra inputs are not permitted; return_choice: Input should be a valid boolean",
            ),
            (
                {"return_choice": 1},
                "Invalid guardrails request options: return_choice: Input should be a valid boolean",
            ),
            (
                "not-a-dict",
                "Invalid guardrails request options: Input should be a valid dictionary or instance of GuardrailsRequest",
            ),
        ],
    )
    def test_rejects_unsupported_guardrails_fields(self, guardrails: Any, expected_message: str) -> None:
        with pytest.raises(InferenceMiddlewareError) as exc_info:
            parse_guardrails_request(guardrails)

        assert exc_info.value.status_code == 422
        assert str(exc_info.value) == expected_message

    def test_returns_none_for_missing_guardrails(self) -> None:
        assert parse_guardrails_request(None) is None


# ---------------------------------------------------------------------------
# extract_log_options_from_request
# ---------------------------------------------------------------------------


class TestExtractLogOptionsFromRequest:
    @pytest.mark.parametrize(
        "guardrails",
        [
            None,
            parse_guardrails_request({}),
            parse_guardrails_request({"options": {}}),
            parse_guardrails_request({"options": {"log": {}}}),
        ],
    )
    def test_returns_none_for_missing_or_empty_log_options(self, guardrails: Any) -> None:
        assert extract_log_options_from_request(guardrails) is None

    def test_returns_log_options_when_present(self) -> None:
        guardrails = parse_guardrails_request({"options": {"log": {"activated_rails": True, "llm_calls": False}}})

        result = extract_log_options_from_request(guardrails)

        assert result == {"activated_rails": True, "llm_calls": False}

    def test_allows_return_choice_with_log_options(self) -> None:
        guardrails = parse_guardrails_request({"return_choice": True, "options": {"log": {"activated_rails": True}}})

        result = extract_log_options_from_request(guardrails)

        assert result == {"activated_rails": True}

    def test_accepts_stats_flag(self) -> None:
        guardrails = parse_guardrails_request({"options": {"log": {"stats": True}}})

        result = extract_log_options_from_request(guardrails)

        assert result == {"stats": True}


# ---------------------------------------------------------------------------
# extract_return_choice_from_request
# ---------------------------------------------------------------------------


class TestExtractReturnChoiceFromRequest:
    @pytest.mark.parametrize(
        ("guardrails", "expected"),
        [
            (None, False),
            (parse_guardrails_request({}), False),
            (parse_guardrails_request({"return_choice": False}), False),
            (parse_guardrails_request({"return_choice": True}), True),
        ],
    )
    def test_variants(self, guardrails: Any, expected: bool) -> None:
        assert extract_return_choice_from_request(guardrails) is expected


# ---------------------------------------------------------------------------
# sanitize_request_body_for_proxy
# ---------------------------------------------------------------------------


class TestSanitizeRequestBodyForProxy:
    def test_removes_guardrails_without_mutating_original(self) -> None:
        request_body = {
            "model": "llama",
            "messages": [{"role": "user", "content": "Hello"}],
            "guardrails": {"options": {"log": {"activated_rails": True}}},
        }

        result = sanitize_request_body_for_proxy(request_body)

        assert result == {
            "model": "llama",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        assert "guardrails" in request_body

    def test_returns_copy_even_without_guardrails(self) -> None:
        request_body = {"model": "llama", "messages": []}

        result = sanitize_request_body_for_proxy(request_body)

        assert result == request_body
        assert result is not request_body


# ---------------------------------------------------------------------------
# build_generate_async_options
# ---------------------------------------------------------------------------


class TestBuildGenerateAsyncOptions:
    def test_forces_activated_rails_true_with_no_user_options(self) -> None:
        result = build_generate_async_options(["input"], None)
        assert result["log"]["activated_rails"] is True

    def test_forces_activated_rails_true_even_when_user_sets_false(self) -> None:
        result = build_generate_async_options(["input"], {"activated_rails": False})
        assert result["log"]["activated_rails"] is True

    def test_includes_user_log_flags(self) -> None:
        result = build_generate_async_options(["input"], {"llm_calls": True, "internal_events": True})
        assert result["log"]["llm_calls"] is True
        assert result["log"]["internal_events"] is True

    def test_rail_types_passed_through(self) -> None:
        assert build_generate_async_options(["input"], None)["rails"] == ["input"]
        assert build_generate_async_options(["output"], None)["rails"] == ["output"]
        assert build_generate_async_options(["input", "output"], None)["rails"] == ["input", "output"]

    def test_empty_user_options_dict(self) -> None:
        result = build_generate_async_options(["input"], {})
        assert result["log"] == {"activated_rails": True}

    def test_does_not_mutate_user_log_options(self) -> None:
        user_log_options: GenerationLogOptionsParam = {"activated_rails": False, "internal_events": True}

        result = build_generate_async_options(["input"], user_log_options)

        assert result["log"]["activated_rails"] is True
        assert user_log_options == {"activated_rails": False, "internal_events": True}


# ---------------------------------------------------------------------------
# build_guardrails_data
# ---------------------------------------------------------------------------


class TestBuildGuardrailsData:
    @pytest.mark.parametrize(
        (
            "include_input",
            "include_output",
            "expected_rails_count",
            "expected_colang_history",
        ),
        [
            (True, False, 1, None),
            (False, True, 1, "output-history"),
            (True, True, 2, "output-history"),
        ],
    )
    def test_build_guardrails_data_merges_requested_logs(
        self,
        include_input: bool,
        include_output: bool,
        expected_rails_count: int,
        expected_colang_history: str | None,
    ) -> None:
        input_response = _response(
            activated_rails=[_rail(stop=True)],
            llm_calls=[{"task": "self_check"}],
            internal_events=[{"event": "input_checked"}],
        )
        output_response = _response(
            activated_rails=[_rail(name="self check output", rail_type="output")],
            llm_calls=[{"task": "output_check"}],
            internal_events=[{"event": "output_checked"}],
            colang_history="output-history",
        )

        result = build_guardrails_data(
            config_id="ws/my-config",
            user_log_options=GenerationLogOptionsParam(
                activated_rails=True,
                llm_calls=True,
                internal_events=True,
                colang_history=True,
            ),
            input_generation_response=input_response if include_input else None,
            output_generation_response=output_response if include_output else None,
        )

        assert result.config_ids == ["ws/my-config"]
        assert result.log is not None
        assert result.log.activated_rails is not None
        assert len(result.log.activated_rails) == expected_rails_count

        expected_internal_events = []
        if include_input:
            expected_internal_events.append({"event": "input_checked"})
        if include_output:
            expected_internal_events.append({"event": "output_checked"})

        assert result.log.internal_events == expected_internal_events
        assert result.log.colang_history == expected_colang_history

    def test_input_only_generation_response_returns_guardrails_data_with_log(self) -> None:
        result = build_guardrails_data(
            config_id="ws/my-config",
            user_log_options={"activated_rails": True},
            input_generation_response=_response(),
            output_generation_response=None,
        )

        assert result.config_ids == ["ws/my-config"]
        assert result.log is not None

    def test_stats_flag_sums_input_and_output_stats(self) -> None:
        input_response = _response(
            stats=GenerationStats(
                input_rails_duration=0.1,
                total_duration=0.15,
                llm_calls_count=1,
                llm_calls_total_tokens=10,
            ),
        )
        output_response = _response(
            stats=GenerationStats(
                output_rails_duration=0.2,
                total_duration=0.25,
                llm_calls_count=2,
                llm_calls_total_tokens=20,
            ),
        )

        result = build_guardrails_data(
            config_id="ws/my-config",
            user_log_options={"stats": True},
            input_generation_response=input_response,
            output_generation_response=output_response,
        )

        assert result.log is not None
        assert result.log.stats is not None
        assert result.log.stats.input_rails_duration == pytest.approx(0.1)
        assert result.log.stats.output_rails_duration == pytest.approx(0.2)
        assert result.log.stats.total_duration == pytest.approx(0.4)
        assert result.log.stats.llm_calls_count == 3
        assert result.log.stats.llm_calls_total_tokens == 30


# ---------------------------------------------------------------------------
# extract_response_content
# ---------------------------------------------------------------------------


class TestExtractResponseContent:
    @pytest.mark.parametrize(
        ("generation_response", "expected_content"),
        [
            (_response(content=[{"role": "assistant", "content": "hi"}]), "hi"),
            (
                _response(
                    content=[
                        {"role": "assistant", "content": "first"},
                        {"role": "assistant", "content": "last"},
                    ]
                ),
                "last",
            ),
            (_response(content="direct string"), "direct string"),
            (_response(content=[]), ""),
            (_response(content=[{"role": "user", "content": "hello"}]), ""),
        ],
    )
    def test_extracts_expected_content(self, generation_response: GenerationResponse, expected_content: str) -> None:
        assert extract_response_content(generation_response) == expected_content

    def test_message_without_content_key_returns_empty_string(self) -> None:
        resp = _response(content=[{"role": "assistant"}])
        assert extract_response_content(resp) == ""


# ---------------------------------------------------------------------------
# Source-keyed rails introspection
# ---------------------------------------------------------------------------


def _rails_with_flows(
    *,
    input_flows: list[str] | None = None,
    output_flows: list[str] | None = None,
    output_streaming: dict[str, Any] | None = None,
) -> SDKRailsConfig:
    rails: dict[str, Any] = {}
    if input_flows is not None:
        rails["input"] = {"flows": input_flows}
    if output_flows is not None:
        output_rails: dict[str, Any] = {"flows": output_flows}
        if output_streaming is not None:
            output_rails["streaming"] = output_streaming
        rails["output"] = output_rails
    return SDKRailsConfig.model_validate({"rails": rails} if rails else {})


def _entity_source(rails: SDKRailsConfig, *, name: str = "test") -> EntityGuardrailConfigSource:
    return EntityGuardrailConfigSource(workspace="ws", name=name, updated_at="2026-01-01T00:00:00Z", rails=rails)


def _inline_source(rails: SDKRailsConfig, *, label: str | None = None) -> InlineGuardrailConfigSource:
    return InlineGuardrailConfigSource(rails=rails, label=label)


class TestSourceHasInputFlows:
    def test_returns_false_when_rails_section_missing(self) -> None:
        assert source_has_input_flows(_entity_source(_rails_with_flows())) is False

    def test_returns_false_when_only_output_flows(self) -> None:
        assert source_has_input_flows(_entity_source(_rails_with_flows(output_flows=["self check output"]))) is False

    def test_returns_false_when_input_flows_empty(self) -> None:
        assert source_has_input_flows(_entity_source(_rails_with_flows(input_flows=[]))) is False

    def test_returns_true_when_input_flows_present(self) -> None:
        assert source_has_input_flows(_entity_source(_rails_with_flows(input_flows=["self check input"]))) is True

    def test_works_for_inline_source(self) -> None:
        """The discriminator is set; source-keyed predicates work for both arms."""
        assert source_has_input_flows(_inline_source(_rails_with_flows(input_flows=["self check input"]))) is True


class TestSourceHasOutputFlows:
    def test_returns_false_when_rails_section_missing(self) -> None:
        assert source_has_output_flows(_entity_source(_rails_with_flows())) is False

    def test_returns_false_when_only_input_flows(self) -> None:
        assert source_has_output_flows(_entity_source(_rails_with_flows(input_flows=["self check input"]))) is False

    def test_returns_false_when_output_flows_empty(self) -> None:
        assert source_has_output_flows(_entity_source(_rails_with_flows(output_flows=[]))) is False

    def test_returns_true_when_output_flows_present(self) -> None:
        assert source_has_output_flows(_entity_source(_rails_with_flows(output_flows=["self check output"]))) is True

    def test_works_for_inline_source(self) -> None:
        assert source_has_output_flows(_inline_source(_rails_with_flows(output_flows=["self check output"]))) is True


class TestExtractOutputRailsStreamingConfig:
    def test_returns_none_when_output_section_missing(self) -> None:
        assert extract_output_rails_streaming_config(_entity_source(_rails_with_flows())) is None

    def test_returns_streaming_config_when_present(self) -> None:
        source = _entity_source(
            _rails_with_flows(
                output_flows=["self check output"],
                output_streaming={"enabled": False},
            )
        )
        cfg = extract_output_rails_streaming_config(source)
        assert cfg is not None
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# build_main_llm — consumes the pre-validated main_model_template
# ---------------------------------------------------------------------------


def _resolve_target(_model_id: str) -> OpenAICompatibleInferenceTarget:
    return OpenAICompatibleInferenceTarget(
        openai_base_url="http://igw.example/provider/v1",
        model="meta/llama-3.1-8b-instruct",
    )


def _sdk_rails(models: list[dict[str, Any]] | None = None) -> SDKRailsConfig:
    """Build an :class:`SDKRailsConfig` directly — this is what ``source.rails``
    is for both the entity and inline arms after the redesign."""
    return SDKRailsConfig.model_validate(
        {
            "rails": {"input": {"flows": []}, "output": {"flows": []}},
            "models": models if models is not None else [],
        }
    )


def _main_template(models: list[dict[str, Any]] | None) -> Model | None:
    """Run the same stabilize → main_model_template extraction the middleware
    uses, so build_main_llm here exercises the same input shape it sees in
    production. Skipping stabilize would test a path no caller takes.
    """
    rails = _sdk_rails(models)
    return stabilize(rails, _resolve_target).main_model_template


class TestBuildMainLlm:
    def test_uses_request_model_name(self) -> None:
        template = _main_template([{"type": "main", "engine": "nim", "model": "ws/placeholder"}])

        with pytest.MonkeyPatch.context() as monkey:
            captured: dict[str, Any] = {}

            def _fake_init(*, model_name: str, provider_name: str, mode: str, kwargs: dict) -> Any:
                captured.update(model_name=model_name, provider_name=provider_name, mode=mode, kwargs=kwargs)
                return MagicMock(name="main_llm")

            monkey.setattr("nemo_guardrails_plugin.rails.init_llm_model", _fake_init)

            llm = build_main_llm(
                {"model": "ws/llama-70b"},
                {},
                _resolve_target,
                template,
            )

        assert llm is not None
        assert captured["model_name"] == "ws/llama-70b"
        assert captured["provider_name"] == "nim"
        assert captured["mode"] == "chat"

    def test_warns_when_main_model_template_requests_non_chat_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        """IGW only routes chat-completions; surface the mismatch instead
        of silently using the user's ``mode='text'`` config so
        misconfigured deployments find out at runtime rather than after
        ``init_llm_model`` returns a text-mode client. ``init_llm_model``
        is still called with ``mode='chat'`` — the warning is purely
        observability.
        """
        template = _main_template(
            [
                {
                    "type": "main",
                    "engine": "nim",
                    "model": "ws/placeholder",
                    "mode": "text",
                }
            ]
        )
        assert template is not None
        assert template.mode == "text"

        with pytest.MonkeyPatch.context() as monkey:
            captured: dict[str, Any] = {}

            def _fake_init(**kw: Any) -> Any:
                captured.update(kw)
                return MagicMock()

            monkey.setattr("nemo_guardrails_plugin.rails.init_llm_model", _fake_init)

            with caplog.at_level("WARNING", logger="nemo_guardrails_plugin.rails"):
                build_main_llm({"model": "ws/llama"}, {}, _resolve_target, template)

        assert captured["mode"] == "chat"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "text" in warnings[0].getMessage()
        assert "chat" in warnings[0].getMessage()

    def test_no_warning_when_main_model_template_is_chat(self, caplog: pytest.LogCaptureFixture) -> None:
        # Negative case so the warning above can't degrade into "always fires".
        template = _main_template([{"type": "main", "engine": "nim", "model": "ws/placeholder", "mode": "chat"}])

        with pytest.MonkeyPatch.context() as monkey:
            monkey.setattr("nemo_guardrails_plugin.rails.init_llm_model", lambda **_: MagicMock())
            with caplog.at_level("WARNING", logger="nemo_guardrails_plugin.rails"):
                build_main_llm({"model": "ws/llama"}, {}, _resolve_target, template)

        assert [r for r in caplog.records if r.levelname == "WARNING"] == []

    def test_forwards_allowlisted_request_headers(self) -> None:
        """Forward ``x-*`` (NeMo Platform principal, ``x-otel-*``, custom) and W3C
        Trace Context (``traceparent``, ``tracestate``, ``baggage``) into
        the main LLM client; drop everything else (``Authorization`` is
        IGW's responsibility, not the plugin's).

        Pinning the W3C trace headers explicitly because they don't
        match the ``x-*`` prefix and were silently dropped before — a
        regression there would break trace propagation invisibly."""
        template = _main_template(
            [
                {
                    "type": "main",
                    "engine": "nim",
                    "model": "ws/placeholder",
                    "parameters": {"default_headers": {"X-Static": "yes"}},
                }
            ]
        )

        with pytest.MonkeyPatch.context() as monkey:
            captured: dict[str, Any] = {}

            def _fake_init(**kw: Any) -> Any:
                captured.update(kw)
                return MagicMock()

            monkey.setattr("nemo_guardrails_plugin.rails.init_llm_model", _fake_init)

            build_main_llm(
                {"model": "ws/llama"},
                {
                    "X-NMP-Principal": "user-1",
                    "x-otel-traceparent": "00-trace-001",
                    "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
                    "tracestate": "vendor=foo,other=bar",
                    "baggage": "userId=alice,session=42",
                    "Authorization": "Bearer should-be-dropped",
                    "Cookie": "should-also-be-dropped",
                },
                _resolve_target,
                template,
            )

        headers = captured["kwargs"]["default_headers"]
        assert headers["X-Static"] == "yes"
        assert headers["X-NMP-Principal"] == "user-1"
        assert headers["x-otel-traceparent"] == "00-trace-001"
        assert headers["traceparent"] == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        assert headers["tracestate"] == "vendor=foo,other=bar"
        assert headers["baggage"] == "userId=alice,session=42"
        assert "Authorization" not in headers
        assert "Cookie" not in headers

    def test_forwards_w3c_trace_headers_case_insensitively(self) -> None:
        """HTTP header names are case-insensitive per RFC 7230; the
        allowlist match must follow suit. Pinning this so a header
        arriving as ``Traceparent`` (e.g. from a Go-style upstream) or
        ``TRACEPARENT`` (some proxies normalize to upper) still
        propagates."""
        template = _main_template([])

        with pytest.MonkeyPatch.context() as monkey:
            captured: dict[str, Any] = {}

            def _fake_init(**kw: Any) -> Any:
                captured.update(kw)
                return MagicMock()

            monkey.setattr("nemo_guardrails_plugin.rails.init_llm_model", _fake_init)

            build_main_llm(
                {"model": "ws/llama"},
                {
                    "Traceparent": "00-mixed-case",
                    "TRACESTATE": "vendor=upper",
                    "Baggage": "k=v",
                },
                _resolve_target,
                template,
            )

        headers = captured["kwargs"]["default_headers"]
        assert headers["Traceparent"] == "00-mixed-case"
        assert headers["TRACESTATE"] == "vendor=upper"
        assert headers["Baggage"] == "k=v"

    def test_uses_resolved_base_url_when_unset_in_config(self) -> None:
        # No main entry → template is None → base_url is resolved per request.
        template = _main_template(None)

        with pytest.MonkeyPatch.context() as monkey:
            captured: dict[str, Any] = {}

            def _fake_init(**kw: Any) -> Any:
                captured.update(kw)
                return MagicMock()

            monkey.setattr("nemo_guardrails_plugin.rails.init_llm_model", _fake_init)

            build_main_llm({"model": "ws/llama"}, {}, _resolve_target, template)

        assert captured["kwargs"]["base_url"] == "http://igw.example/provider/v1"

    def test_falls_back_to_default_engine_when_no_main_in_config(self) -> None:
        # No main, no actions — template is None.
        template = _main_template([])

        with pytest.MonkeyPatch.context() as monkey:
            captured: dict[str, Any] = {}

            def _fake_init(**kw: Any) -> Any:
                captured.update(kw)
                return MagicMock()

            monkey.setattr("nemo_guardrails_plugin.rails.init_llm_model", _fake_init)

            build_main_llm({"model": "ws/llama"}, {}, _resolve_target, template)

        from nemo_guardrails_plugin.constants import DEFAULT_MAIN_ENGINE

        assert captured["provider_name"] == DEFAULT_MAIN_ENGINE

    def test_does_not_inject_api_key_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Authentication against the main LLM is IGW's responsibility, so
        ``Model.api_key_env_var`` is intentionally NOT consulted by
        :func:`build_main_llm`. Pinning this prevents a future "helpful"
        addition that would smuggle a per-tenant secret into the LangChain
        client and bypass IGW's auth path.

        ``_main_template`` runs the library validator, which rejects an
        ``api_key_env_var`` pointing at an unset env var — so the env var
        must still be populated to construct the template. That's a
        library invariant, not a plugin-side dependency.
        """
        monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
        template = _main_template(
            [
                {
                    "type": "main",
                    "engine": "nim",
                    "model": "ws/placeholder",
                    "api_key_env_var": "TEST_API_KEY",
                }
            ]
        )

        captured: dict[str, Any] = {}

        def _fake_init(**kw: Any) -> Any:
            captured.update(kw)
            return MagicMock()

        monkeypatch.setattr("nemo_guardrails_plugin.rails.init_llm_model", _fake_init)

        build_main_llm({"model": "ws/llama"}, {}, _resolve_target, template)

        assert "api_key" not in captured["kwargs"]

    def test_missing_request_model_raises(self) -> None:
        """The per-request main LLM is meaningless without ``request_body['model']``."""
        template = _main_template([])
        with pytest.raises(ValueError, match="model"):
            build_main_llm({}, {}, _resolve_target, template)
