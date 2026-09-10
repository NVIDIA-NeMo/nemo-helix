# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the chat CLI command."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click import UsageError
from nemo_platform_ext.cli.app import app
from nemo_platform_ext.cli.commands.use_cases.chat import _parse_model_and_workspace
from typer.testing import CliRunner

REMOTE_ERROR_EXIT_CODE = 3

# =============================================================================
# Unit Tests for _parse_model_and_workspace
# =============================================================================


@pytest.mark.parametrize(
    "model,workspace_flag,workspace_config,expected",
    [
        # Model only, workspace from config
        ("my-model", None, "default", ("default", "default/my-model")),
        # Model with inline workspace
        ("custom/my-model", None, "default", ("custom", "custom/my-model")),
        # Workspace flag takes precedence over config
        ("my-model", "explicit", "default", ("explicit", "explicit/my-model")),
        # Inline workspace, no config needed
        ("custom/my-model", None, None, ("custom", "custom/my-model")),
        # Workspace flag only, no config
        ("my-model", "explicit", None, ("explicit", "explicit/my-model")),
    ],
)
def test_parse_model_and_workspace_success(
    model: str,
    workspace_flag: str | None,
    workspace_config: str | None,
    expected: tuple[str, str],
) -> None:
    """Test successful workspace and model resolution."""
    result = _parse_model_and_workspace(model, workspace_flag, workspace_config)
    assert result == expected


def test_parse_model_and_workspace_conflict_error() -> None:
    """Test error when workspace specified both inline and via flag."""
    with pytest.raises(UsageError) as exc_info:
        _parse_model_and_workspace("custom/my-model", "explicit", None)
    assert "both in model name" in str(exc_info.value)
    assert "custom" in str(exc_info.value)
    assert "explicit" in str(exc_info.value)


def test_parse_model_and_workspace_no_workspace_error() -> None:
    """Test error when no workspace can be determined."""
    with pytest.raises(UsageError) as exc_info:
        _parse_model_and_workspace("my-model", None, None)
    assert "No workspace specified" in str(exc_info.value)


# =============================================================================
# CLI Syntax Validation Tests
#
# These tests verify the CLI accepts valid syntax. They run against a fake
# server URL so commands fail with connection errors (exit code 3), not
# syntax errors (exit code 2).
# =============================================================================


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """Create a CLI test runner with isolated config."""
    for var in list(os.environ):
        if var.startswith("NMP_"):
            monkeypatch.delenv(var, raising=False)

    config_file = tmp_path / "config.yaml"
    config_file.touch()
    monkeypatch.setenv("NMP_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("NMP_BASE_URL", "http://localhost:9999")

    return CliRunner()


def _streaming_response(chunks: list[bytes], status_code: int | None = None) -> MagicMock:
    """Fake typed ``BinaryContent`` response: ``stream()`` yields *chunks*."""
    response = MagicMock()

    @contextmanager
    def _stream():
        yield iter(chunks)

    response.stream = MagicMock(side_effect=_stream)
    response.http_response = SimpleNamespace(status_code=status_code) if status_code is not None else SimpleNamespace()
    return response


def _mock_streaming_response(*chunks: str, usage: dict | None = None) -> MagicMock:
    events = [json.dumps({"choices": [{"delta": {"content": chunk}}]}) for chunk in chunks]
    if usage is not None:
        events.append(json.dumps({"choices": [], "usage": usage}))
    events.append("[DONE]")
    return _streaming_response([("".join(f"data: {event}\n\n" for event in events)).encode()])


def _mock_streaming_error_response(message: str, status_code: int | None = None) -> MagicMock:
    return _streaming_response([f"event: error\ndata: {message}\n\n".encode()], status_code=status_code)


def _mock_client_with_openai_response(response: MagicMock) -> MagicMock:
    mock_client = MagicMock()
    mock_client.workspace = "default"
    mock_client.stream_openai.return_value = response
    return mock_client


def test_chat_missing_model_fails_with_usage_error(runner: CliRunner) -> None:
    """Chat without model argument should fail with usage error."""
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 2  # Typer/Click usage error


def test_chat_model_only(runner: CliRunner) -> None:
    """Chat with just model argument parses correctly."""
    result = runner.invoke(app, ["chat", "my-model", "hello"])
    assert result.exit_code == REMOTE_ERROR_EXIT_CODE


def test_chat_model_with_inline_workspace(runner: CliRunner) -> None:
    """Chat with workspace/model syntax parses correctly."""
    result = runner.invoke(app, ["chat", "my-workspace/my-model", "hello"])
    assert result.exit_code == REMOTE_ERROR_EXIT_CODE


def test_chat_with_workspace_flag(runner: CliRunner) -> None:
    """Chat with --workspace flag parses correctly."""
    result = runner.invoke(app, ["chat", "my-model", "hello", "--workspace", "test-ws"])
    assert result.exit_code == REMOTE_ERROR_EXIT_CODE


def test_chat_with_provider_flag(runner: CliRunner) -> None:
    """Chat with --provider flag parses correctly."""
    result = runner.invoke(
        app, ["chat", "nvidia/model-id", "hello", "--provider", "nvidia-build", "--workspace", "test-ws"]
    )
    assert result.exit_code == REMOTE_ERROR_EXIT_CODE


def test_chat_provider_rejects_workspace_prefix(runner: CliRunner) -> None:
    """Chat with --provider containing workspace prefix fails with helpful error."""
    result = runner.invoke(
        app, ["chat", "nvidia/model-id", "hello", "--provider", "workspace/build", "--workspace", "default"]
    )
    assert result.exit_code == 2
    assert "Invalid provider name 'workspace/build'" in result.output
    assert "workspace prefix" in result.output
    assert "Use '--provider build' instead" in result.output


def test_chat_with_temperature(runner: CliRunner) -> None:
    """Chat with --temperature parses correctly."""
    result = runner.invoke(app, ["chat", "ws/model", "hello", "--temperature", "0.7"])
    assert result.exit_code == REMOTE_ERROR_EXIT_CODE


def test_chat_with_max_tokens(runner: CliRunner) -> None:
    """Chat with --max-tokens parses correctly."""
    result = runner.invoke(app, ["chat", "ws/model", "hello", "--max-tokens", "512"])
    assert result.exit_code == REMOTE_ERROR_EXIT_CODE


def test_chat_with_system_message(runner: CliRunner) -> None:
    """Chat with --system-message parses correctly."""
    result = runner.invoke(app, ["chat", "ws/model", "hello", "--system-message", "You are helpful."])
    assert result.exit_code == REMOTE_ERROR_EXIT_CODE


def test_chat_all_options(runner: CliRunner) -> None:
    """Chat with all options parses correctly."""
    result = runner.invoke(
        app,
        [
            "chat",
            "nvidia/llama-3.3-nemotron",
            "What is 2+2?",
            "--provider",
            "nvidia-build",
            "--workspace",
            "default",
            "--temperature",
            "0.5",
            "--max-tokens",
            "100",
            "--system-message",
            "Be concise.",
        ],
    )
    assert result.exit_code == REMOTE_ERROR_EXIT_CODE


def test_chat_temperature_rejects_non_numeric(runner: CliRunner) -> None:
    """Temperature option rejects non-numeric values."""
    result = runner.invoke(app, ["chat", "ws/model", "hi", "--temperature", "hot"])
    assert result.exit_code == 2  # Usage error


def test_chat_max_tokens_rejects_non_integer(runner: CliRunner) -> None:
    """Max-tokens option rejects non-integer values."""
    result = runner.invoke(app, ["chat", "ws/model", "hi", "--max-tokens", "many"])
    assert result.exit_code == 2  # Usage error


def test_chat_prompt_runs_once_with_plain_text_output(runner: CliRunner) -> None:
    """A prompt sends one request, prints plain text, and exits."""
    response = _mock_streaming_response("3", "91")
    mock_client = _mock_client_with_openai_response(response)

    with (
        patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client),
        patch("nemo_platform_ext.cli.chat_tui.Prompt.ask") as mock_prompt,
    ):
        result = runner.invoke(
            app,
            ["chat", "my-model", "What is 17 * 23? Reply with just the number."],
        )

    assert result.exit_code == 0
    assert result.stdout == "391\n"
    assert "NeMo Platform Chat Session" not in result.stdout
    assert "\x1b[" not in result.stdout
    mock_prompt.assert_not_called()

    post = mock_client.stream_openai
    post.assert_called_once()
    _, kwargs = post.call_args
    assert kwargs["workspace"] == "default"
    assert kwargs["body"].root["model"] == "default/my-model"
    assert kwargs["body"].root["messages"] == [
        {"role": "user", "content": "What is 17 * 23? Reply with just the number."}
    ]
    assert kwargs["body"].root["stream"] is True


def test_chat_one_shot_includes_system_message(runner: CliRunner) -> None:
    """One-shot chat should prepend the system message to request history."""
    response = _mock_streaming_response("ok")
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model", "hi", "--system-message", "Be concise."])

    assert result.exit_code == 0
    _, kwargs = mock_client.stream_openai.call_args
    assert kwargs["body"].root["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]


def test_chat_text_output_strips_thinking_tags_when_no_regular_content(runner: CliRunner) -> None:
    """Text output should not leak reasoning markup to scripted callers."""
    response = _mock_streaming_response("<think>scratch work</think>")
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model", "hi"])

    assert result.exit_code == 0
    assert result.stdout == "\n"
    assert "<think>" not in result.stdout


def test_chat_text_output_strips_split_thinking_tags(runner: CliRunner) -> None:
    """Text streaming should handle thinking tags split across chunks."""
    response = _mock_streaming_response("visible <thi", "nk>scratch", "</think> done")
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model", "hi"])

    assert result.exit_code == 0
    assert result.stdout == "visible  done\n"
    assert "<think>" not in result.stdout
    assert "scratch" not in result.stdout


def test_chat_text_output_strips_split_closing_thinking_tag(runner: CliRunner) -> None:
    """Text streaming should handle closing thinking tags split across chunks."""
    response = _mock_streaming_response("<think>x</thi", "nk>after")
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model", "hi"])

    assert result.exit_code == 0
    assert result.stdout == "after\n"


def test_chat_stream_error_event_fails(runner: CliRunner) -> None:
    """SSE error events should fail the command instead of looking like an empty response."""
    response = _mock_streaming_error_response("backend exploded")
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model", "hi"])

    assert result.exit_code == 1
    assert "Streaming chat request failed: backend exploded" in result.output
    assert "Unexpected error" not in result.output


def test_chat_empty_stream_error_event_includes_status_code(runner: CliRunner) -> None:
    """Empty SSE error events should include HTTP status when available."""
    response = _mock_streaming_error_response("", status_code=503)
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model", "hi"])

    assert result.exit_code == 1
    assert "Streaming chat request failed (HTTP 503)" in result.output


def test_chat_prompt_takes_precedence_over_piped_stdin(runner: CliRunner) -> None:
    """When both PROMPT and stdin are present, PROMPT is the one-shot input."""
    response = _mock_streaming_response("from prompt")
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model", "prompt wins"], input="stdin loses")

    assert result.exit_code == 0
    assert result.stdout == "from prompt\n"
    _, kwargs = mock_client.stream_openai.call_args
    assert kwargs["body"].root["messages"] == [{"role": "user", "content": "prompt wins"}]


def test_chat_interactive_with_prompt_sends_initial_message_then_prompts(runner: CliRunner) -> None:
    """--interactive with a prompt pre-sends the message and keeps the REPL open."""
    response = _mock_streaming_response("hello")
    mock_client = _mock_client_with_openai_response(response)
    captured_messages = None

    def capture_post(**kwargs):
        nonlocal captured_messages
        captured_messages = deepcopy(kwargs["body"].root["messages"])
        return response

    mock_client.stream_openai.side_effect = capture_post

    with (
        patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client),
        patch("nemo_platform_ext.cli.commands.use_cases.chat._is_interactive_chat_session", return_value=True),
        patch("nemo_platform_ext.cli.chat_tui.Prompt.ask", side_effect=KeyboardInterrupt) as mock_prompt,
    ):
        result = runner.invoke(app, ["chat", "my-model", "hi", "--interactive"])

    assert result.exit_code == 0
    mock_prompt.assert_called_once()

    post = mock_client.stream_openai
    post.assert_called_once()
    assert captured_messages == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("exception", [KeyboardInterrupt, EOFError])
def test_chat_interactive_interrupt_during_initial_response_exits_gracefully(
    runner: CliRunner, exception: type[BaseException]
) -> None:
    response = _mock_streaming_response("unused")
    response.stream.side_effect = exception
    mock_client = _mock_client_with_openai_response(response)

    with (
        patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client),
        patch("nemo_platform_ext.cli.commands.use_cases.chat._is_interactive_chat_session", return_value=True),
        patch("nemo_platform_ext.cli.chat_tui.Prompt.ask") as mock_prompt,
    ):
        result = runner.invoke(app, ["chat", "my-model", "hi", "--interactive"])

    assert result.exit_code == 0
    assert "Chat session ended" in result.stdout
    mock_prompt.assert_not_called()


def test_chat_interactive_preserves_model_history_across_shared_tui_turns(runner: CliRunner) -> None:
    """The shared TUI must leave model transcript management with the model-chat command."""
    responses = [_mock_streaming_response("hello"), _mock_streaming_response("follow-up answer")]
    mock_client = _mock_client_with_openai_response(responses[0])
    captured_messages: list[list[dict[str, str]]] = []

    def capture_post(**kwargs):
        captured_messages.append(deepcopy(kwargs["body"].root["messages"]))
        return responses[len(captured_messages) - 1]

    mock_client.stream_openai.side_effect = capture_post

    with (
        patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client),
        patch("nemo_platform_ext.cli.commands.use_cases.chat._is_interactive_chat_session", return_value=True),
        patch(
            "nemo_platform_ext.cli.chat_tui.Prompt.ask",
            side_effect=["What did I just say?", KeyboardInterrupt],
        ),
    ):
        result = runner.invoke(app, ["chat", "my-model", "hi", "--interactive"])

    assert result.exit_code == 0
    assert captured_messages == [
        [{"role": "user", "content": "hi"}],
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "What did I just say?"},
        ],
    ]


def test_chat_interactive_discards_user_turn_after_empty_response(runner: CliRunner) -> None:
    responses = [_mock_streaming_response(), _mock_streaming_response("answer")]
    mock_client = _mock_client_with_openai_response(responses[0])
    captured_messages: list[list[dict[str, str]]] = []

    def capture_post(**kwargs):
        captured_messages.append(deepcopy(kwargs["body"].root["messages"]))
        return responses[len(captured_messages) - 1]

    mock_client.stream_openai.side_effect = capture_post

    with (
        patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client),
        patch("nemo_platform_ext.cli.commands.use_cases.chat._is_interactive_chat_session", return_value=True),
        patch("nemo_platform_ext.cli.chat_tui.Prompt.ask", side_effect=["second turn", KeyboardInterrupt]),
    ):
        result = runner.invoke(app, ["chat", "my-model", "empty turn", "--interactive"])

    assert result.exit_code == 0
    assert captured_messages == [
        [{"role": "user", "content": "empty turn"}],
        [{"role": "user", "content": "second turn"}],
    ]


def test_chat_interactive_requires_tty(runner: CliRunner) -> None:
    """--interactive fails fast when the REPL cannot safely read from a terminal."""
    result = runner.invoke(app, ["chat", "my-model", "hi", "--interactive"])

    assert result.exit_code == 2
    assert "Interactive chat requires a terminal" in result.output


def test_chat_reads_prompt_from_stdin_in_non_tty_mode(runner: CliRunner) -> None:
    """Piped stdin can supply the one-shot prompt."""
    response = _mock_streaming_response("from stdin")
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model"], input="hello from stdin")

    assert result.exit_code == 0
    assert result.stdout == "from stdin\n"
    _, kwargs = mock_client.stream_openai.call_args
    assert kwargs["body"].root["messages"] == [{"role": "user", "content": "hello from stdin"}]


def test_chat_json_output_includes_content_thinking_model_and_usage(runner: CliRunner) -> None:
    """One-shot JSON output is script-friendly and separates reasoning text."""
    response = _mock_streaming_response(
        "<think>scratch work</think>",
        "final answer",
        usage={"prompt_tokens": 4, "completion_tokens": 2},
    )
    mock_client = _mock_client_with_openai_response(response)

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model", "hi", "--output-format", "json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "content": "final answer",
        "thinking": "scratch work",
        "model": "default/my-model",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }


def test_chat_non_tty_without_prompt_requires_prompt(runner: CliRunner) -> None:
    """Non-TTY mode fails fast instead of entering the prompt loop with no input."""
    mock_client = MagicMock()
    mock_client.workspace = "default"

    with patch("nemo_platform_ext.cli.core.context.CLIContext.typed_client", return_value=mock_client):
        result = runner.invoke(app, ["chat", "my-model"])

    assert result.exit_code == 2
    assert "One-shot chat requires a prompt" in result.output


# =============================================================================
# Provider Routing Tests
#
# These tests verify the correct API endpoints are called for provider routing.
# =============================================================================


def test_chat_provider_routing_uses_v1_prefix(runner: CliRunner) -> None:
    """Chat with --provider must use v1/chat/completions endpoint.

    This test ensures the trailing_uri includes the 'v1/' prefix required
    by OpenAI-compatible APIs. Without this prefix, requests return 404.
    """
    # Track the trailing_uri passed to the provider post method
    captured_trailing_uri = None
    captured_kwargs = None

    def mock_post(trailing_uri: str, **kwargs) -> MagicMock:
        nonlocal captured_trailing_uri
        nonlocal captured_kwargs
        captured_trailing_uri = trailing_uri
        captured_kwargs = kwargs
        return _mock_streaming_response("Hello")

    mock_client = MagicMock()
    mock_client.workspace = "default"
    mock_client.stream_provider = mock_post

    with patch(
        "nemo_platform_ext.cli.core.context.CLIContext.typed_client",
        return_value=mock_client,
    ):
        result = runner.invoke(
            app,
            [
                "chat",
                "nvidia/llama-3.3-nemotron-super-49b-v1",
                "Hello!",
                "--provider",
                "build",
                "--max-tokens",
                "50",
            ],
        )

    # Verify the correct endpoint was called
    assert result.exit_code == 0
    assert captured_trailing_uri == "v1/chat/completions", (
        f"Expected trailing_uri='v1/chat/completions', got '{captured_trailing_uri}'. "
        "Provider routing must include 'v1/' prefix for OpenAI-compatible APIs."
    )
    assert captured_kwargs is not None
    assert captured_kwargs["workspace"] == "default"
    assert captured_kwargs["name"] == "build"
    assert captured_kwargs["body"].root["model"] == "nvidia/llama-3.3-nemotron-super-49b-v1"
    assert captured_kwargs["body"].root["messages"] == [{"role": "user", "content": "Hello!"}]
    assert captured_kwargs["body"].root["max_tokens"] == 50
    assert captured_kwargs["body"].root["stream"] is True
