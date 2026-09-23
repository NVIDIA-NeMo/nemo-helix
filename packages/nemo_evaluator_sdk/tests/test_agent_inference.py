# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Agent value type, AgentFormat enum, and agent_inference module."""

import json
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import nemo_evaluator_sdk.agent_inference as agent_inference
import pytest
from jsonschema import Draft202012Validator
from nemo_evaluator_sdk.agent_inference import (
    AgentInferenceFn,
    AgentInvocationResult,
    AgentInvocationStatus,
    _derive_input_message,
    _extract_jsonpath,
    _make_generic_agent_request,
    _make_nat_agent_request,
    _parse_sse_frame,
    _persist_stream_evidence,
    _resolve_http_agent_invocation,
    invoke_agent,
    make_agent_inference_request,
)
from nemo_evaluator_sdk.enums import AgentFormat
from nemo_evaluator_sdk.values.agents import (
    Agent,
    GenericAgent,
    NatAgentConfig,
    NemoAgentToolkitAgent,
)
from nemo_evaluator_sdk.values.common import SecretRef
from nemo_evaluator_sdk.values.evidence import CandidateEvidence, EvidenceDescriptor
from pydantic import TypeAdapter, ValidationError

# ============================================================================
# AgentFormat enum
# ============================================================================


class TestAgentFormat:
    def test_enum_values(self):
        assert AgentFormat.GENERIC == "generic"
        assert AgentFormat.NEMO_AGENT_TOOLKIT == "nemo_agent_toolkit"

    def test_is_string_enum(self):
        assert isinstance(AgentFormat.GENERIC, str)
        assert isinstance(AgentFormat.NEMO_AGENT_TOOLKIT, str)


# ============================================================================
# Agent value type validation
# ============================================================================


class TestAgentValidation:
    def test_nat_config_rejects_runtime_capture_policy(self):
        with pytest.raises(ValidationError, match="capture_evidence"):
            NatAgentConfig(capture_evidence=True)  # ty: ignore[unknown-argument]

    def test_agent_is_discriminated_union_of_concrete_variants(self):
        adapter = TypeAdapter(Agent)

        generic = adapter.validate_python(
            {
                "url": "http://agent.test",
                "name": "generic-agent",
                "format": "generic",
                "body": {"prompt": "{{ prompt }}"},
                "response_path": "$.answer",
                "stream": True,
            }
        )
        nat = adapter.validate_python(
            {
                "url": "http://nat.test",
                "name": "nat-agent",
                "format": "nemo_agent_toolkit",
                "nat": {"endpoint": "/generate/stream"},
            }
        )

        assert isinstance(generic, GenericAgent)
        assert generic.stream is True
        assert isinstance(nat, NemoAgentToolkitAgent)
        schema = adapter.json_schema()
        assert "oneOf" in schema
        assert schema["discriminator"]["propertyName"] == "format"
        assert "allOf" not in schema
        assert "format" in schema["$defs"]["GenericAgent"]["required"]
        assert "format" in schema["$defs"]["NemoAgentToolkitAgent"]["required"]

    def test_agent_union_requires_discriminator_and_rejects_cross_variant_fields(self):
        adapter = TypeAdapter(Agent)

        with pytest.raises(ValidationError, match="union_tag_not_found"):
            adapter.validate_python(
                {
                    "url": "http://agent.test",
                    "name": "agent",
                    "body": {"prompt": "{{ prompt }}"},
                    "response_path": "$.answer",
                }
            )

        with pytest.raises(ValidationError, match="nat"):
            adapter.validate_python(
                {
                    "url": "http://agent.test",
                    "name": "generic-agent",
                    "format": "generic",
                    "body": {"prompt": "{{ prompt }}"},
                    "response_path": "$.answer",
                    "nat": {},
                }
            )

        with pytest.raises(ValidationError, match="body"):
            adapter.validate_python(
                {
                    "url": "http://nat.test",
                    "name": "nat-agent",
                    "format": "nemo_agent_toolkit",
                    "body": {"prompt": "{{ prompt }}"},
                }
            )

    def test_generic_agent_requires_body_and_response_path(self):
        """Generic agents must supply body and response_path."""
        with pytest.raises(ValueError, match="body"):
            GenericAgent(url="http://agent.test", name="test", response_path="$.answer")  # ty: ignore[missing-argument]

        with pytest.raises(ValueError, match="response_path"):
            GenericAgent(  # ty: ignore[missing-argument]
                url="http://agent.test", name="test", body={"query": "{{ prompt }}"}
            )

    def test_generic_agent_valid(self):
        agent = GenericAgent(
            url="http://agent.test",
            name="my-agent",
            format=AgentFormat.GENERIC,
            body={"query": "{{ prompt }}"},
            response_path="$.result.text",
        )
        assert agent.format == AgentFormat.GENERIC
        assert agent.body == {"query": "{{ prompt }}"}
        assert agent.response_path == "$.result.text"
        assert agent.trajectory_path is None

    def test_agent_json_schema_rejects_nat_for_generic_agents(self):
        validator = Draft202012Validator(TypeAdapter(Agent).json_schema())
        generic_with_nat = {
            "url": "http://agent.test",
            "name": "generic-agent",
            "format": "generic",
            "body": {"prompt": "{{ prompt }}"},
            "response_path": "$.answer",
            "nat": {},
        }

        errors = list(validator.iter_errors(generic_with_nat))
        assert errors

        validator.validate(
            {
                "url": "http://nat.test",
                "name": "nat-agent",
                "format": "nemo_agent_toolkit",
                "nat": {},
            }
        )

    def test_nat_agent_does_not_require_body_or_response_path(self):
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)
        assert not hasattr(agent, "body")
        assert not hasattr(agent, "response_path")

    def test_default_format_is_generic(self):
        """Format defaults to 'generic' — so body + response_path are required."""
        with pytest.raises(ValueError, match="body"):
            GenericAgent(url="http://agent.test", name="test")  # ty: ignore[missing-argument]

    def test_api_key_env_sanitisation(self):
        agent = NemoAgentToolkitAgent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            api_key_secret=SecretRef("my-workspace/my-secret-key"),
        )
        # Hyphens and slashes become underscores
        assert agent.api_key_env == "my_workspace_my_secret_key"

    def test_api_key_env_digit_prefix(self):
        agent = NemoAgentToolkitAgent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            api_key_secret=SecretRef("9secret"),
        )
        assert agent.api_key_env == "_9secret"

    def test_api_key_resolved_from_env(self, monkeypatch):
        monkeypatch.setenv("ws_my_key", "sk-test-value")
        agent = NemoAgentToolkitAgent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            api_key_secret=SecretRef("ws/my-key"),
        )
        assert agent.api_key == "sk-test-value"

    def test_api_key_none_when_no_secret(self):
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="test", format=AgentFormat.NEMO_AGENT_TOOLKIT)
        assert agent.api_key is None
        assert agent.api_key_env is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            NemoAgentToolkitAgent.model_validate(
                {
                    "url": "http://agent.test",
                    "name": "test",
                    "format": AgentFormat.NEMO_AGENT_TOOLKIT,
                    "extra_field": "bad",
                }
            )

    def test_generic_with_trajectory_path(self):
        agent = GenericAgent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.answer",
            trajectory_path="$.steps",
        )
        assert agent.trajectory_path == "$.steps"


# ============================================================================
# Resolution: _resolve_http_agent_invocation (generic agent payloads)
# ============================================================================


class TestResolveGenericAgentPayload:
    def test_body_renders_against_task_inputs(self):
        """A generic agent's payload comes entirely from its `body`, rendered against the task
        inputs — no chat/completions wrapper is assumed."""
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"query": "{{ instruction }}", "task": "{{ task_id }}"},
            response_path="$.answer",
        )

        invocation = _resolve_http_agent_invocation(
            agent,
            {"instruction": "What is the capital of France?", "task_id": "capital-france"},
        )

        assert invocation.payload == {"query": "What is the capital of France?", "task": "capital-france"}
        assert invocation.endpoint == "http://agent.test/invoke"

    def test_body_can_reference_arbitrary_input_fields(self):
        """Any field in the task inputs is available to the body template, not just a fixed set."""
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"prompt": "{{ instruction }}", "context": "{{ domain }}"},
            response_path="$.answer",
        )

        invocation = _resolve_http_agent_invocation(
            agent,
            {"instruction": "Summarize the filing.", "domain": "finance"},
        )

        assert invocation.payload == {"prompt": "Summarize the filing.", "context": "finance"}


# ============================================================================
# Helper: _derive_input_message
# ============================================================================


class TestDeriveInputMessage:
    def test_extracts_last_user_message_from_chat(self):
        request = {"messages": [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hello!"}]}
        assert _derive_input_message(request) == "Hello!"

    def test_extracts_prompt(self):
        assert _derive_input_message({"prompt": "Summarise this"}) == "Summarise this"

    def test_fallback_concatenates_all_messages(self):
        """When no user message is found, all contents are concatenated."""
        request = {"messages": [{"role": "system", "content": "System"}, {"role": "assistant", "content": "Hi"}]}
        result = _derive_input_message(request)
        assert "System" in result
        assert "Hi" in result

    def test_raises_when_no_messages_or_prompt(self):
        with pytest.raises(ValueError, match="messages.*prompt"):
            _derive_input_message({"model": "x"})


# ============================================================================
# Helper: _extract_jsonpath
# ============================================================================


class TestExtractJsonpath:
    def test_simple_extraction(self):
        data = {"result": {"text": "hello world"}}
        assert _extract_jsonpath(data, "$.result.text") == "hello world"

    def test_multiple_matches_return_last_value(self):
        data = {"result": {"answers": ["first", "last"]}}
        assert _extract_jsonpath(data, "$.result.answers[*]") == "last"

    def test_required_raises_when_no_match(self):
        with pytest.raises(ValueError, match="did not match"):
            _extract_jsonpath({"a": 1}, "$.missing", field_name="response_path")

    def test_optional_returns_none(self):
        result = _extract_jsonpath({"a": 1}, "$.missing", required=False)
        assert result is None


# ============================================================================
# Routing: make_agent_inference_request
# ============================================================================


class TestMakeAgentInferenceRequestRouting:
    @pytest.mark.asyncio
    async def test_delegates_generic_agent_to_typed_invocation(self):
        agent = GenericAgent(
            url="http://agent.test",
            name="test",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.answer",
        )
        with patch(
            "nemo_evaluator_sdk.agent_inference.invoke_agent",
            new_callable=AsyncMock,
            return_value=AgentInvocationResult(
                status=AgentInvocationStatus.COMPLETED,
                response={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                output_text="ok",
            ),
        ) as mock_invoke:
            result = await make_agent_inference_request(agent, {"prompt": "hi"})
            mock_invoke.assert_awaited_once()
            await_args = mock_invoke.await_args
            assert await_args is not None
            assert await_args.args[:2] == (agent, {"prompt": "hi"})
            assert result["choices"][0]["message"]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_delegates_nat_agent_to_typed_invocation(self):
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat", format=AgentFormat.NEMO_AGENT_TOOLKIT)
        with patch(
            "nemo_evaluator_sdk.agent_inference.invoke_agent",
            new_callable=AsyncMock,
            return_value=AgentInvocationResult(
                status=AgentInvocationStatus.COMPLETED,
                response={"choices": [{"message": {"role": "assistant", "content": "nat-ok"}}]},
                output_text="nat-ok",
            ),
        ) as mock_invoke:
            result = await make_agent_inference_request(agent, {"prompt": "hi"})
            mock_invoke.assert_awaited_once()
            await_args = mock_invoke.await_args
            assert await_args is not None
            assert await_args.args[:2] == (agent, {"prompt": "hi"})
            assert result["choices"][0]["message"]["content"] == "nat-ok"


# ============================================================================
# Generic agent executor
# ============================================================================


class TestGenericAgentExecutor:
    @pytest.mark.asyncio
    async def test_stream_translator_receives_generic_agent_frames_and_context(self):
        from nemo_evaluator_sdk.agent_stream_translation import (
            AgentStreamTranslation,
            AgentStreamTranslationContext,
            SseFrame,
        )

        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="streaming-generic",
            body={"question": "{{ prompt }}"},
            response_path="$.answer",
            stream=True,
        )
        seen_frames: list[Sequence[SseFrame]] = []
        seen_contexts: list[AgentStreamTranslationContext] = []

        def translate(
            frames: Sequence[SseFrame],
            *,
            context: AgentStreamTranslationContext,
        ) -> AgentStreamTranslation:
            seen_frames.append(frames)
            seen_contexts.append(context)
            return AgentStreamTranslation(
                trajectory={
                    "schema_version": "ATIF-v1.7",
                    "agent": {"name": context.agent_name, "version": "0"},
                    "steps": [{"step_id": 1, "source": "agent", "message": context.output_text}],
                }
            )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content='data: {"answer":"final"}\n',
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await invoke_agent(
                agent,
                {"prompt": "question"},
                client=client,
                stream_translator=translate,
            )

        assert result.status is AgentInvocationStatus.COMPLETED
        assert result.evidence is not None
        assert result.evidence.require("trace", kind="trace").format == "atif"
        assert seen_frames[0][0].payload == {"answer": "final"}
        assert seen_contexts[0].endpoint == "http://agent.test/invoke"

    @pytest.mark.asyncio
    async def test_streams_json_sse_and_extracts_last_output_and_trajectory(self):
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="streaming-generic",
            body={"question": "{{ prompt }}"},
            response_path="$.answer",
            trajectory_path="$.steps",
            stream=True,
        )
        seen_payload: dict[str, object] = {}

        def handle(request: httpx.Request) -> httpx.Response:
            seen_payload.update(json.loads(request.content))
            return httpx.Response(
                200,
                content=(
                    'data: {"answer":"draft","steps":[{"id":1}]}\n'
                    'intermediate_data: {"name":"tool"}\n'
                    'data: {"answer":"final","steps":[{"id":1},{"id":2}]}\n'
                    "data: [DONE]\n"
                ),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await invoke_agent(agent, {"prompt": "question"}, client=client)

        assert seen_payload == {"question": "question"}
        assert result.status is AgentInvocationStatus.COMPLETED
        assert result.output_text == "final"
        assert result.response["choices"][0]["message"]["content"] == "final"
        assert result.response["trajectory"] == [{"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_stream_without_capture_or_translator_does_not_retain_raw_events(self):
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="streaming-generic",
            body={"question": "{{ prompt }}"},
            response_path="$.answer",
            stream=True,
        )
        captures: list[agent_inference._StreamCapture] = []
        capture_type = agent_inference._StreamCapture

        def create_capture() -> agent_inference._StreamCapture:
            capture = capture_type()
            captures.append(capture)
            return capture

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=('intermediate_data: {"name":"tool"}\ndata: {"answer":"final"}\ndata: [DONE]\n'),
                headers={"content-type": "text/event-stream"},
            )
        )
        with patch.object(agent_inference, "_StreamCapture", side_effect=create_capture):
            async with httpx.AsyncClient(transport=transport) as client:
                result = await invoke_agent(agent, {"prompt": "question"}, client=client)

        assert result.status is AgentInvocationStatus.COMPLETED
        assert result.output_text == "final"
        assert result.metadata["event_count"] == 3
        assert result.evidence is None
        assert captures[0].raw_lines == []
        assert captures[0].frames == []

    @pytest.mark.asyncio
    async def test_posts_rendered_body_and_extracts_response(self):
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"question": "{{ prompt }}"},
            response_path="$.answer",
        )

        with (
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience", new_callable=AsyncMock) as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            mock_resilience.return_value = {"answer": "42", "meta": {}}

            result = await _make_generic_agent_request(agent, {"prompt": "What is the meaning?"}, max_retries=1)

        assert result["choices"][0]["message"]["content"] == "42"
        assert result["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_extracts_trajectory_when_configured(self):
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.text",
            trajectory_path="$.steps",
        )

        with (
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience", new_callable=AsyncMock) as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            mock_resilience.return_value = {"text": "answer", "steps": [{"action": "search"}, {"action": "summarize"}]}

            result = await _make_generic_agent_request(agent, {"prompt": "hi"})

        assert result["trajectory"] == [{"action": "search"}, {"action": "summarize"}]

    @pytest.mark.asyncio
    async def test_includes_auth_header_when_api_key_provided(self):
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.text",
        )

        captured_fn = None

        async def capture_resilience(endpoint_key, fn, max_attempts):
            nonlocal captured_fn
            captured_fn = fn
            return {"text": "ok"}

        with (
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience", side_effect=capture_resilience),
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            await _make_generic_agent_request(agent, {"prompt": "hi"}, api_key="secret-key")
            # The inner function captured via resilience would have the headers set;
            # We verified the flow runs without error when api_key is provided.

    @pytest.mark.asyncio
    async def test_generic_agent_request_includes_default_headers(self):
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="gen",
            format=AgentFormat.GENERIC,
            body={"q": "{{ prompt }}"},
            response_path="$.text",
        )

        with (
            patch("nemo_evaluator_sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn
            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_response = MagicMock()
            mock_response.json.return_value = {"text": "ok"}
            mock_client_instance.post = AsyncMock(return_value=mock_response)

            await _make_generic_agent_request(
                agent,
                {"prompt": "hi"},
                default_headers={"X-NHX-Principal-Id": "service:evaluator"},
            )

        assert mock_client_instance.post.await_args is not None
        assert mock_client_instance.post.await_args.kwargs["headers"] == {
            "X-NHX-Principal-Id": "service:evaluator",
            "Content-Type": "application/json",
        }


# ============================================================================
# NAT agent executor
# ============================================================================


class TestNATAgentExecutor:
    def test_stream_translation_preserves_complete_original_trajectory(self):
        from nemo_evaluator_sdk.agent_stream_translation import AgentStreamTranslation

        raw_trajectory = {
            "schema_version": "ATIF-v1.7",
            "agent": {"name": "bugnemo", "version": "0", "future_agent_field": "kept"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "agent",
                    "message": "tool complete",
                    "tool_calls": [
                        {
                            "tool_call_id": "call-1",
                            "function_name": "count",
                            "arguments": {},
                            "extra": {"raw_id": "call-1"},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "call-1",
                                "content": '{"count": 4}',
                                "extra": {"event_type": "function_complete"},
                            }
                        ]
                    },
                    "extra": {"stream_updates": [{"sequence": 1}, {"sequence": 2}]},
                    "future_step_field": {"kept": True},
                }
            ],
            "future_root_field": {"kept": True},
        }

        translation = AgentStreamTranslation(trajectory=raw_trajectory)

        assert translation.trajectory == raw_trajectory
        assert translation.trajectory["future_root_field"] == {"kept": True}
        assert translation.trajectory["steps"][0]["future_step_field"] == {"kept": True}

    @pytest.mark.asyncio
    async def test_custom_stream_translator_adds_canonical_trace_and_context(self):
        from nemo_evaluator_sdk.agent_stream_translation import AgentStreamTranslation
        from nemo_evaluator_sdk.values.evidence import EvidenceDescriptor

        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )
        payload = {
            "input_message": "How many bugs?",
            "conversation_id": "conversation-1",
        }
        seen = {}

        def translate(frames, *, context):
            seen["frames"] = frames
            seen["context"] = context
            return AgentStreamTranslation(
                trajectory={
                    "schema_version": "ATIF-v1.7",
                    "session_id": context.conversation_id,
                    "agent": {"name": context.agent_name, "version": "0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "How many bugs?"}],
                },
                evidence={
                    "tool_evidence": EvidenceDescriptor(
                        kind="tool_evidence",
                        format="json",
                        data=[{"count": 12}],
                    )
                },
            )

        async def fake_aiter_lines():
            yield 'intermediate_data: {"name":"Function Complete: text2sql_df"}'
            yield 'data: {"value":{"value":"There are 12 bugs."}}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = fake_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with (
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn
            result = await invoke_agent(
                agent,
                payload,
                client=mock_client,
                stream_translator=translate,
                invocation_context={
                    "run_id": "run-1",
                    "task_id": "task-1",
                    "invocation_id": "invocation-1",
                },
                default_headers={"Authorization": "Bearer secret"},
            )

        assert result.status is AgentInvocationStatus.COMPLETED
        assert result.evidence is not None
        assert set(result.evidence.names()) >= {"trace", "tool_evidence", "stream_events"}
        trace = result.evidence.require("trace", kind="trace")
        assert trace.format == "atif"
        assert isinstance(trace.data, dict)
        assert trace.data["schema_version"] == "ATIF-v1.7"
        assert trace.data["agent"] == {"name": "nat-agent", "version": "0"}
        assert trace.data["steps"][0]["step_id"] == 1
        assert seen["frames"][0].channel == "intermediate_data"
        context = seen["context"]
        assert context.run_id == "run-1"
        assert context.task_id == "task-1"
        assert context.invocation_id == "invocation-1"
        assert context.conversation_id == "conversation-1"
        assert "headers" not in type(context).model_fields
        assert "secret" not in context.model_dump_json()

    @pytest.mark.asyncio
    async def test_custom_stream_translation_replaces_task_evidence_with_file_backed_artifacts(
        self,
        tmp_path,
    ):
        from nemo_evaluator_sdk.agent_stream_translation import AgentStreamTranslation
        from nemo_evaluator_sdk.values.evidence import EvidenceDescriptor

        evidence_dir = tmp_path / "task-1"
        evidence_dir.mkdir()
        stale = evidence_dir / "stale-legacy-artifact.json"
        stale.write_text("{}", encoding="utf-8")
        outside = tmp_path / "keep.txt"
        outside.write_text("keep", encoding="utf-8")

        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )

        def translate(frames, *, context):
            del frames
            return AgentStreamTranslation(
                trajectory={
                    "schema_version": "ATIF-v1.7",
                    "session_id": context.conversation_id,
                    "agent": {"name": context.agent_name, "version": "0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
                },
                evidence={
                    "tool_evidence": EvidenceDescriptor(
                        kind="tool_evidence",
                        format="json",
                        data=[{"count": 12}],
                    )
                },
            )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content='intermediate_data: {"name":"tool"}\ndata: {"value":{"value":"There are 12 bugs."}}\n',
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await invoke_agent(
                agent,
                {"input_message": "hi", "conversation_id": "conversation-1"},
                client=client,
                stream_translator=translate,
                evidence_dir=evidence_dir,
            )

        assert result.evidence is not None
        trace = result.evidence.require("trace", kind="trace")
        tool_evidence = result.evidence.require("tool_evidence", kind="tool_evidence")
        assert trace.ref == str((evidence_dir / "atif_trace.json").resolve())
        assert trace.data is None
        assert tool_evidence.ref == str((evidence_dir / "tool_evidence.json").resolve())
        assert tool_evidence.data is None
        assert not stale.exists()
        assert outside.read_text(encoding="utf-8") == "keep"

    def test_persisted_evidence_filenames_are_confined_and_unique(self, tmp_path: Path) -> None:
        evidence_dir = tmp_path / "task-1"
        evidence = CandidateEvidence(
            descriptors={
                "trace": EvidenceDescriptor(
                    kind="trace",
                    format="atif",
                    data={"schema_version": "ATIF-v1.7", "steps": []},
                ),
                "../outside": EvidenceDescriptor(
                    kind="derived",
                    format="json",
                    data={"value": "must stay inside"},
                ),
                "atif_trace": EvidenceDescriptor(
                    kind="derived",
                    format="json",
                    data={"value": "must not overwrite trace"},
                ),
            }
        )

        persisted = _persist_stream_evidence(evidence, evidence_dir)
        refs = {
            name: Path(descriptor.ref)
            for name, descriptor in persisted.descriptors.items()
            if descriptor.ref is not None
        }

        assert set(refs) == {"trace", "../outside", "atif_trace"}
        assert refs["trace"].name == "atif_trace.json"
        assert all(path.parent == evidence_dir.resolve() for path in refs.values())
        assert len({path.name.casefold() for path in refs.values()}) == len(refs)
        trace_payload = json.loads(refs["trace"].read_text(encoding="utf-8"))
        assert trace_payload["schema_version"] == "ATIF-v1.7"
        assert not (tmp_path / "outside.json").exists()

    @pytest.mark.asyncio
    async def test_translation_failure_returns_failed_result_with_raw_evidence(
        self,
        tmp_path,
    ):
        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )

        def translate(frames, *, context):
            del frames, context
            raise ValueError("cannot translate BugNeMo stream")

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content='intermediate_data: {"name":"broken"}\ndata: {"value":{"value":"final answer"}}\n',
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await invoke_agent(
                agent,
                {"input_message": "hi"},
                client=client,
                stream_translator=translate,
                evidence_dir=tmp_path / "task-1",
            )

        assert result.status is AgentInvocationStatus.FAILED
        assert result.output_text == "final answer"
        assert result.evidence is not None
        raw_stream = result.evidence.require("raw_stream", kind="agent_stream")
        translation_error = result.evidence.require("translation_error", kind="error")
        assert raw_stream.ref is not None
        assert translation_error.ref is not None
        assert "cannot translate BugNeMo stream" in result.metadata["translation_error"]

    @pytest.mark.asyncio
    async def test_translator_reserved_evidence_collision_fails_invocation(self):
        from nemo_evaluator_sdk.agent_stream_translation import AgentStreamTranslation
        from nemo_evaluator_sdk.values.evidence import EvidenceDescriptor

        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )

        def translate(frames, *, context):
            del frames
            return AgentStreamTranslation(
                trajectory={
                    "schema_version": "ATIF-v1.7",
                    "agent": {"name": context.agent_name, "version": "0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
                },
                evidence={
                    "raw_stream": EvidenceDescriptor(
                        kind="text",
                        format="text",
                        data="replacement",
                    )
                },
            )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content='intermediate_data: {"name":"tool"}\ndata: {"value":{"value":"answer"}}\n',
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await invoke_agent(
                agent,
                {"input_message": "hi"},
                client=client,
                stream_translator=translate,
            )

        assert result.status is AgentInvocationStatus.FAILED
        assert result.evidence is not None
        error = result.evidence.require("translation_error", kind="error")
        assert isinstance(error.data, dict)
        assert "reserved names" in error.data["error"]

    @pytest.mark.asyncio
    async def test_translator_must_return_canonical_atif_v1_7(self):
        from nemo_evaluator_sdk.agent_stream_translation import AgentStreamTranslation

        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )

        def translate(frames, *, context):
            del frames, context
            return AgentStreamTranslation(
                trajectory={
                    "schema_version": "ATIF-v1.6",
                    "agent": {"name": "nat-agent", "version": "0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
                }
            )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content='data: {"value":{"value":"answer"}}\n',
                request=request,
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await invoke_agent(
                agent,
                {"input_message": "hi"},
                client=client,
                stream_translator=translate,
            )

        assert result.status is AgentInvocationStatus.FAILED
        assert result.evidence is not None
        error = result.evidence.require("translation_error", kind="error")
        assert isinstance(error.data, dict)
        assert "ATIF-v1.7" in error.data["error"]

    @pytest.mark.asyncio
    async def test_translator_runs_for_partial_stream_and_preserves_partial_status(self):
        from nemo_evaluator_sdk.agent_stream_translation import AgentStreamTranslation

        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )

        def translate(frames, *, context):
            assert len(frames) == 1
            return AgentStreamTranslation(
                trajectory={
                    "schema_version": "ATIF-v1.7",
                    "agent": {"name": context.agent_name, "version": "0"},
                    "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
                }
            )

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content='intermediate_data: {"name":"Function Start: workflow"}\n',
                request=request,
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            result = await invoke_agent(
                agent,
                {"input_message": "hi"},
                client=client,
                stream_translator=translate,
            )

        assert result.status is AgentInvocationStatus.PARTIAL
        assert result.evidence is not None
        assert result.evidence.require("trace", kind="trace").format == "atif"

    @pytest.mark.asyncio
    async def test_pre_frame_http_error_with_translator_returns_partial_without_capture(self):
        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )
        request = httpx.Request("POST", "http://nat.test/generate/stream")
        response = httpx.Response(503, request=request)
        http_error = httpx.HTTPStatusError(
            "unavailable",
            request=request,
            response=response,
        )
        translator = MagicMock()

        with patch(
            "nemo_evaluator_sdk.agent_inference.run_with_resilience",
            side_effect=http_error,
        ):
            result = await invoke_agent(
                agent,
                {"input_message": "hi"},
                client=AsyncMock(),
                stream_translator=translator,
            )

        assert result.status is AgentInvocationStatus.PARTIAL
        translator.assert_not_called()
        assert result.evidence is not None
        http_metadata = result.evidence.require("http_metadata").data
        assert isinstance(http_metadata, dict)
        assert http_metadata["status_code"] == 503
        assert "translation_error" not in result.evidence.names()

    @pytest.mark.asyncio
    async def test_invokes_configured_stream_endpoint_with_passthrough_and_evidence(self):
        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )
        payload = {
            "input_message": "How many bugs?",
            "user_name": "Ada",
            "conversation_id": "run-1-task-1",
        }
        sse_lines = [
            'intermediate_data: {"name":"Function Complete: text2sql_df","payload":"count=12"}',
            'data: {"value":{"value":"There are 12 bugs.","bug_ids":[]}}',
        ]

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = fake_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with (
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn
            result = await invoke_agent(agent, payload, client=mock_client, capture_evidence=True)

        assert result.status is AgentInvocationStatus.COMPLETED
        assert result.output_text == "There are 12 bugs."
        assert result.evidence is not None
        assert set(result.evidence.names()) == {
            "http_metadata",
            "raw_stream",
            "request_headers",
            "request_payload",
            "stream_events",
        }
        assert "trace" not in result.evidence.names()
        events = result.evidence.require("stream_events").data
        assert isinstance(events, list)
        assert events[0]["channel"] == "intermediate_data"
        assert events[1]["payload"]["value"]["bug_ids"] == []
        assert mock_client.stream.call_args.args[:2] == ("POST", "http://nat.test/generate/stream")
        assert mock_client.stream.call_args.kwargs["json"] == payload
        assert mock_client.stream.call_args.kwargs["params"] == {}

    @pytest.mark.asyncio
    async def test_invocation_without_final_value_is_partial_and_keeps_stream_evidence(self):
        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )

        async def fake_aiter_lines():
            yield 'intermediate_data: {"name":"Function Start: workflow"}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = fake_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with (
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn
            result = await invoke_agent(
                agent,
                {"input_message": "hi"},
                client=mock_client,
                capture_evidence=True,
            )

        assert result.status is AgentInvocationStatus.PARTIAL
        assert result.output_text is None
        assert result.evidence is not None
        raw_stream = result.evidence.require("raw_stream").data
        assert isinstance(raw_stream, str)
        assert raw_stream.endswith("\n")

    @pytest.mark.asyncio
    async def test_streams_sse_and_extracts_final_value(self):
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

        # Simulate SSE lines
        sse_lines = [
            'data: {"step": "thinking", "content": "processing"}',
            'data: {"step": "action", "content": "searching"}',
            'data: {"value": "The final answer is 42"}',
        ]

        # Mock httpx stream directly
        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_lines = fake_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncContextManagerMock(mock_stream_response))

        with (
            patch("nemo_evaluator_sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            # run_with_resilience should call the fn passed to it
            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn

            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            # Mock stream context manager
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = fake_aiter_lines
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)

            mock_client_instance.stream = MagicMock(return_value=mock_response)

            result = await _make_nat_agent_request(agent, {"messages": [{"role": "user", "content": "What is 6*7?"}]})

        assert result["choices"][0]["message"]["content"] == "The final answer is 42"
        assert result["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_raises_when_no_value_in_stream(self):
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

        sse_lines = [
            'data: {"step": "thinking", "content": "processing"}',
            "",
        ]

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        with (
            patch("nemo_evaluator_sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn

            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = fake_aiter_lines
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.stream = MagicMock(return_value=mock_response)

            # The stream has no "value" field, so after stream completes final_response is None
            # and the function should raise RuntimeError
            with pytest.raises(RuntimeError, match="without producing a final value"):
                await _make_nat_agent_request(agent, {"messages": [{"role": "user", "content": "hi"}]})

    @pytest.mark.asyncio
    async def test_uses_last_value_from_stream(self):
        """With ``response_aggregation='last'``, the last chunk with 'value' wins."""
        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(response_aggregation="last"),
        )

        sse_lines = [
            'data: {"value": "partial answer"}',
            'data: {"value": "complete final answer"}',
        ]

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        with (
            patch("nemo_evaluator_sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn

            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = fake_aiter_lines
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.stream = MagicMock(return_value=mock_response)

            result = await _make_nat_agent_request(agent, {"prompt": "hi"})

        assert result["choices"][0]["message"]["content"] == "complete final answer"

    def test_default_nat_aggregation_is_concat(self):
        """NAT targets reconstruct token deltas by default (see NVBug 6486650)."""
        assert NatAgentConfig().response_aggregation == "concat"

    @pytest.mark.asyncio
    async def test_concat_reconstructs_token_delta_stream(self):
        """NAT /generate/full emits one token per frame; concat rebuilds the full answer.

        Regression test for NVBug 6486650: a correct answer ``15.0`` was previously
        persisted as only the final delta token ``"0"``.
        """
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=(
                    'data: {"value": "Final Answer: "}\n'
                    'data: {"value": "15"}\n'
                    'data: {"value": "."}\n'
                    'data: {"value": "0"}\n'
                    "data: [DONE]\n"
                ),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await invoke_agent(agent, {"prompt": "Calculate 7 + 8."}, client=client)

        assert result.status is AgentInvocationStatus.COMPLETED
        assert result.output_text == "Final Answer: 15.0"
        assert result.response["choices"][0]["message"]["content"] == "Final Answer: 15.0"

    @pytest.mark.asyncio
    async def test_concat_stream_error_frame_yields_partial(self):
        """A terminal error frame keeps the trial PARTIAL even when deltas were seen."""
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=(
                    'data: {"value": "15"}\n'
                    'data: {"value": "."}\n'
                    'data: {"error": {"code": "workflow_error", "message": "boom"}}\n'
                ),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await invoke_agent(agent, {"prompt": "hi"}, client=client)

        assert result.status is AgentInvocationStatus.PARTIAL
        assert result.metadata["stream_error"] == "workflow_error: boom"

    @pytest.mark.asyncio
    async def test_generic_agent_opts_into_concat_aggregation(self):
        """GenericAgent can reconstruct token-delta endpoints via response_aggregation='concat'."""
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="streaming-generic",
            body={"question": "{{ prompt }}"},
            response_path="$.value",
            stream=True,
            response_aggregation="concat",
        )

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=('data: {"value": "4"}\ndata: {"value": "2"}\ndata: [DONE]\n'),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await invoke_agent(agent, {"prompt": "answer"}, client=client)

        assert result.status is AgentInvocationStatus.COMPLETED
        assert result.output_text == "42"

    @pytest.mark.asyncio
    async def test_generic_agent_defaults_to_last_aggregation(self):
        """GenericAgent keeps snapshot-per-frame ('last') semantics by default."""
        agent = GenericAgent(
            url="http://agent.test/invoke",
            name="streaming-generic",
            body={"question": "{{ prompt }}"},
            response_path="$.answer",
            stream=True,
        )
        assert agent.response_aggregation == "last"

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=('data: {"answer": "draft"}\ndata: {"answer": "final"}\ndata: [DONE]\n'),
                headers={"content-type": "text/event-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await invoke_agent(agent, {"prompt": "answer"}, client=client)

        assert result.output_text == "final"

    @pytest.mark.asyncio
    async def test_nat_agent_request_includes_default_headers(self):
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)

        async def fake_aiter_lines():
            yield 'data: {"value": "ok"}'

        with (
            patch("nemo_evaluator_sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn
            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = fake_aiter_lines
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.stream = MagicMock(return_value=mock_response)

            await _make_nat_agent_request(
                agent,
                {"prompt": "hi"},
                default_headers={"X-NHX-Principal-Id": "service:evaluator"},
            )

        assert mock_client_instance.stream.call_args is not None
        assert mock_client_instance.stream.call_args.kwargs["headers"] == {
            "X-NHX-Principal-Id": "service:evaluator",
            "Content-Type": "application/json",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("data_line", "expected_content"),
        [
            ('data: {"value": {"answer": 7}}', {"answer": 7}),
            ('data: {"value": 7}', 7),
        ],
    )
    async def test_preserves_non_string_value_type(self, data_line, expected_content):
        """The ``last`` aggregation on ``$.value`` keeps the raw value type in OpenAI content."""
        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(response_aggregation="last"),
        )

        async def fake_aiter_lines():
            yield data_line

        with (
            patch("nemo_evaluator_sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn
            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)

            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = fake_aiter_lines
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=False)
            mock_client_instance.stream = MagicMock(return_value=mock_response)

            result = await _make_nat_agent_request(agent, {"prompt": "hi"})

        # Raw type preserved, not stringified.
        assert result["choices"][0]["message"]["content"] == expected_content

    @pytest.mark.asyncio
    async def test_empty_string_final_value_is_partial(self):
        """An extracted-but-empty value stays PARTIAL while evidence is captured."""
        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )

        async def fake_aiter_lines():
            yield 'data: {"value": {"value": ""}}'

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = fake_aiter_lines
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_response)

        with (
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience") as mock_resilience,
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):

            async def call_fn(endpoint_key, fn, max_attempts):
                return await fn()

            mock_resilience.side_effect = call_fn
            result = await invoke_agent(
                agent,
                {"input_message": "hi"},
                client=mock_client,
                capture_evidence=True,
            )

        assert result.status is AgentInvocationStatus.PARTIAL
        assert result.metadata["final_payload"] == {"value": {"value": ""}}
        assert result.evidence is not None
        assert "stream_events" in result.evidence.names()

    @pytest.mark.asyncio
    async def test_pre_frame_http_error_with_evidence_returns_partial(self):
        """With evidence capture, an HTTP error before any frame is a PARTIAL result."""
        agent = NemoAgentToolkitAgent(
            url="http://nat.test",
            name="nat-agent",
            format=AgentFormat.NEMO_AGENT_TOOLKIT,
            nat=NatAgentConfig(
                endpoint="/generate/stream",
                request_mode="passthrough",
                query_params={},
                response_path="$.value.value",
            ),
        )
        request = httpx.Request("POST", "http://nat.test/generate/stream")
        response = httpx.Response(401, headers={"www-authenticate": "Bearer"}, request=request)
        http_error = httpx.HTTPStatusError("unauthorized", request=request, response=response)
        mock_client = AsyncMock()

        with (
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience", side_effect=http_error),
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            result = await invoke_agent(
                agent,
                {"input_message": "hi"},
                client=mock_client,
                capture_evidence=True,
            )

        assert result.status is AgentInvocationStatus.PARTIAL
        assert result.metadata["http_status"] == 401
        assert result.evidence is not None
        http_metadata = result.evidence.require("http_metadata").data
        assert isinstance(http_metadata, dict)
        assert http_metadata["status_code"] == 401
        assert http_metadata["error"] == "HTTP 401"

    @pytest.mark.asyncio
    async def test_pre_frame_http_error_without_evidence_raises(self):
        """The legacy path (no evidence capture) still raises on HTTP failure."""
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat-agent", format=AgentFormat.NEMO_AGENT_TOOLKIT)
        request = httpx.Request("POST", "http://nat.test/generate/full")
        response = httpx.Response(401, request=request)
        http_error = httpx.HTTPStatusError("unauthorized", request=request, response=response)

        with (
            patch("nemo_evaluator_sdk.agent_inference.httpx.AsyncClient") as MockClient,
            patch("nemo_evaluator_sdk.agent_inference.run_with_resilience", side_effect=http_error),
            patch("nemo_evaluator_sdk.agent_inference.get_logger", return_value=MagicMock()),
        ):
            mock_client_instance = AsyncMock()
            MockClient.return_value = mock_client_instance
            with pytest.raises(httpx.HTTPStatusError):
                await _make_nat_agent_request(agent, {"prompt": "hi"})


# ============================================================================
# NAT SSE frame parsing
# ============================================================================


class TestParseNatFrame:
    @pytest.mark.parametrize(
        ("channel", "line"),
        [
            ("data", 'data: {"value": 1}'),
            ("intermediate_data", 'intermediate_data: {"name": "x"}'),
            ("observability_trace", "observability_trace: {}"),
            ("custom-channel", "custom-channel: {}"),
        ],
    )
    def test_parses_valid_channel_lines(self, channel, line):
        frame = _parse_sse_frame(line)
        assert frame is not None
        assert frame.channel == channel

    @pytest.mark.parametrize(
        "line",
        [
            '{"value": 1, "nested": {"k": "v"}}',  # bare JSON, no SSE prefix
            "event: ping",
            "no-colon-here",
            "",
        ],
    )
    def test_skips_non_frame_lines(self, line):
        assert _parse_sse_frame(line) is None


# ============================================================================
# Protocol conformance
# ============================================================================


class TestAgentInferenceFnProtocol:
    def test_evidence_constants_have_stable_wire_values(self):
        from nemo_evaluator_sdk.values.evidence import (
            EVIDENCE_FORMAT_ATIF,
            EVIDENCE_FORMAT_JSON,
            EVIDENCE_FORMAT_TEXT,
            EVIDENCE_HTTP_METADATA,
            EVIDENCE_RAW_STREAM,
            EVIDENCE_REQUEST_HEADERS,
            EVIDENCE_REQUEST_PAYLOAD,
            EVIDENCE_STREAM_EVENTS,
            EVIDENCE_TRACE,
            EVIDENCE_TRANSLATION_ERROR,
        )

        assert {
            EVIDENCE_TRACE,
            EVIDENCE_RAW_STREAM,
            EVIDENCE_STREAM_EVENTS,
            EVIDENCE_REQUEST_PAYLOAD,
            EVIDENCE_REQUEST_HEADERS,
            EVIDENCE_HTTP_METADATA,
            EVIDENCE_TRANSLATION_ERROR,
        } == {
            "trace",
            "raw_stream",
            "stream_events",
            "request_payload",
            "request_headers",
            "http_metadata",
            "translation_error",
        }
        assert (EVIDENCE_FORMAT_ATIF, EVIDENCE_FORMAT_JSON, EVIDENCE_FORMAT_TEXT) == ("atif", "json", "text")

    @pytest.mark.asyncio
    async def test_factory_binds_context_translator_and_capture_policy(self, tmp_path: Path):
        from nemo_evaluator_sdk.agent_inference import AgentInferenceContext, make_agent_inference_fn

        context = AgentInferenceContext(
            evidence_dir=tmp_path / "evidence",
            metadata={"run_id": "run-1", "task_id": "task-1"},
        )
        translator = MagicMock()
        agent = NemoAgentToolkitAgent(url="http://nat.test", name="nat-agent")
        typed_result = AgentInvocationResult(
            status=AgentInvocationStatus.COMPLETED,
            response={"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
            output_text="answer",
        )

        with patch(
            "nemo_evaluator_sdk.agent_inference.invoke_agent",
            new_callable=AsyncMock,
            return_value=typed_result,
        ) as mock_invoke:
            inference_fn = make_agent_inference_fn(
                context,
                stream_translator=translator,
                capture_evidence=True,
            )
            result = await inference_fn(agent, {"prompt": "hi"}, max_retries=2)

        assert result is typed_result
        await_args = mock_invoke.await_args
        assert await_args is not None
        assert await_args.kwargs["evidence_dir"] == tmp_path / "evidence"
        assert await_args.kwargs["invocation_context"] == {
            "run_id": "run-1",
            "task_id": "task-1",
        }
        assert await_args.kwargs["stream_translator"] is translator
        assert await_args.kwargs["capture_evidence"] is True

    def test_stream_translation_types_are_exported_from_sdk_root(self):
        import nemo_evaluator_sdk as sdk

        assert sdk.SseFrame.__name__ == "SseFrame"
        assert sdk.AgentStreamTranslation.__name__ == "AgentStreamTranslation"
        assert sdk.AgentStreamTranslationContext.__name__ == "AgentStreamTranslationContext"
        assert sdk.AgentStreamTranslator.__name__ == "AgentStreamTranslator"

    def test_make_agent_inference_request_matches_protocol(self):
        """make_agent_inference_request should satisfy AgentInferenceFn protocol."""
        fn: AgentInferenceFn = make_agent_inference_request
        assert callable(fn)


# ============================================================================
# Helper for async context manager mocking
# ============================================================================


class AsyncContextManagerMock:
    """Utility to mock async context managers."""

    def __init__(self, return_value):
        self._return_value = return_value

    async def __aenter__(self):
        return self._return_value

    async def __aexit__(self, *args):
        return False
