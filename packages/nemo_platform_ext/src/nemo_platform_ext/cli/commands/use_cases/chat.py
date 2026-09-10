# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI commands for interactive chat."""

from __future__ import annotations

import json
import select
import sys
from typing import Annotated, Any, Callable, Literal, TypedDict, cast

import click
import typer
from nemo_platform_plugin.inference_gateway.client import InferenceGatewayClient
from nemo_platform_plugin.inference_gateway.types import JsonBody

from nemo_platform_ext.cli.chat_tui import (
    StreamingResponse,
    collect_stream_response,
    parse_thinking,
    run_chat_tui,
    stream_text_response,
)
from nemo_platform_ext.cli.core.api import build_kwargs, is_tty
from nemo_platform_ext.cli.core.autocomplete import autocomplete_model_entity
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.stdin_utils import is_stdin_available
from nemo_platform_ext.ui.prompts import is_interactive


class ChatMessage(TypedDict):
    """Type for chat messages."""

    role: Literal["system", "user", "assistant"]
    content: str


ChatOutputFormat = Literal["text", "json", "raw"]


def _parse_model_and_workspace(
    model: str,
    workspace_flag: str | None,
    workspace_from_config: str | None,
) -> tuple[str, str]:
    """Parse model argument and resolve workspace for model entity routing.

    Args:
        model: Model argument, either "model-name" or "workspace/model-name"
        workspace_flag: Explicit --workspace flag value
        workspace_from_config: Workspace from client config

    Returns:
        Tuple of (workspace, model_entity_id) where model_entity_id is "workspace/model-name"

    Raises:
        click.UsageError: If workspace is specified both inline and via flag,
                          or if no workspace can be determined
    """
    inline_workspace = None
    model_name = model

    # Check for inline workspace (model entity names don't contain /)
    if "/" in model:
        inline_workspace, model_name = model.split("/", 1)

    # Validate no conflict between inline and flag
    if inline_workspace and workspace_flag:
        raise click.UsageError(
            f"Workspace specified both in model name ('{inline_workspace}') and "
            f"via --workspace flag ('{workspace_flag}'). Please use only one method."
        )

    # Resolve final workspace (flag > inline > config)
    final_workspace = workspace_flag or inline_workspace or workspace_from_config

    if not final_workspace:
        raise click.UsageError(
            "No workspace specified. Provide workspace via --workspace flag, "
            "include it in model name (workspace/model-name), or configure a default workspace."
        )

    # Return workspace and full model entity ID
    model_entity_id = f"{final_workspace}/{model_name}"
    return final_workspace, model_entity_id


def _is_interactive_chat_session() -> bool:
    """Return whether the process can safely run the Rich chat REPL."""
    return is_interactive() and is_tty()


def _resolve_chat_mode(prompt: str | None, interactive: bool) -> tuple[bool, str | None]:
    """Resolve whether chat should run once and the prompt to send."""
    can_run_interactive = _is_interactive_chat_session()

    if interactive:
        if not can_run_interactive:
            raise click.UsageError(
                "Interactive chat requires a terminal. Remove --interactive for one-shot mode or run in a TTY."
            )
        return False, prompt

    if prompt:
        return True, prompt

    stdin_prompt = _read_stdin_prompt()
    if stdin_prompt:
        return True, stdin_prompt

    return not can_run_interactive, None


def _read_stdin_prompt() -> str | None:
    """Read a prompt from stdin only when data is ready."""
    if not is_stdin_available() or not _stdin_has_data_ready():
        return None

    stdin_prompt = sys.stdin.read().strip()
    return stdin_prompt or None


def _stdin_has_data_ready() -> bool:
    """Return whether reading stdin should complete without blocking."""
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        # Some platforms cannot probe non-socket stdin. Keep piped stdin support
        # and rely on the non-TTY stream to provide data or EOF.
        return True
    return bool(readable)


def _resolve_chat_output_format(
    state: CLIContext,
    output_format: ChatOutputFormat | None,
) -> ChatOutputFormat:
    """Resolve the one-shot output format, respecting compatible global config.

    The chat command streams plain conversational text and picks its own JSON
    shape when asked, so the global non-TTY ``table -> json`` shortcut does not
    apply here — opt out of it via ``apply_non_tty_default=False``. Otherwise a
    piped or redirected ``nemo chat`` would silently switch from text to JSON
    and break shell pipelines that just want the model's reply on stdout.
    """
    if output_format is not None:
        return output_format

    configured = state.get_output_format(apply_non_tty_default=False)
    return configured if configured in ("json", "raw") else "text"


@handle_errors
def chat(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Argument(
            help="Model entity name (from 'nemo models list') or model ID when using --provider",
            autocompletion=autocomplete_model_entity,
        ),
    ],
    prompt: Annotated[
        str | None,
        typer.Argument(
            help="Prompt for one-shot mode. Takes precedence over piped stdin.",
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            help="Provider name for direct provider routing (bypasses model entity routing)",
        ),
    ] = None,
    workspace: Annotated[str | None, typer.Option(help="Workspace name")] = None,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            help="Start the terminal chat UI; cannot be used with piped stdin. With PROMPT, send it first.",
            rich_help_panel="Chat Options",
        ),
    ] = False,
    output_format: Annotated[
        ChatOutputFormat | None,
        typer.Option(
            "--output-format",
            "--format",
            "-f",
            help="Output format for one-shot responses.",
            show_choices=True,
            rich_help_panel="Output Options",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            help="Sampling temperature (0.0 to 2.0)",
            rich_help_panel="Model Options",
        ),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option(
            help="Maximum tokens to generate",
            rich_help_panel="Model Options",
        ),
    ] = None,
    system_message: Annotated[
        str | None,
        typer.Option(
            "--system-message",
            help="System message to set context for the conversation",
            rich_help_panel="Model Options",
        ),
    ] = None,
) -> None:
    """
    Start an interactive chat session with a model.

    By default, uses model entity routing where the model name should match
    what's shown in 'nemo models list'.

    Use --provider for direct provider routing, where the model argument is
    passed directly to the provider's API.

    Passing PROMPT sends one message and exits unless --interactive is set.
    Omitting PROMPT in a TTY starts the interactive chat UI. In non-TTY
    contexts, PROMPT may also be piped on stdin. Piped stdin is read in full
    before sending. If both PROMPT and piped stdin are provided, PROMPT takes
    precedence.

    Examples:
      nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5
      nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5 "What is machine learning?"
      nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5 "What is machine learning?" --interactive
      echo "What is machine learning?" | nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5
      nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5 "What is machine learning?" -f json
      nemo chat nvidia/llama-3.3-nemotron-super-49b-v1.5 --provider nvidia-build
    """
    state: CLIContext = ctx.obj
    run_once, effective_prompt = _resolve_chat_mode(prompt, interactive)
    if run_once and not effective_prompt:
        raise click.UsageError("One-shot chat requires a prompt. Provide PROMPT or pipe text on stdin.")
    client = state.typed_client(InferenceGatewayClient)
    chat_output_format = _resolve_chat_output_format(state, output_format) if run_once else "text"

    # Get workspace from client config if available
    workspace_from_config: str | None = client.workspace

    if provider:
        # Provider routing: pass model directly to the provider
        if "/" in provider:
            suggested = provider.split("/")[-1]
            raise click.UsageError(
                f"Invalid provider name '{provider}'. Provider names should not include a workspace prefix.\n"
                f"[yellow]Hint:[/] Use '--provider {suggested}' instead."
            )

        resolved_workspace = workspace or workspace_from_config

        if not resolved_workspace:
            raise click.UsageError(
                "No workspace specified. Provide workspace via --workspace flag or configure a default workspace."
            )

        def get_response(body: dict[str, Any]) -> StreamingResponse:
            return client.stream_provider(
                trailing_uri="v1/chat/completions",
                workspace=resolved_workspace,
                name=provider,
                body=JsonBody(body),
            )

        model_for_body = model
        display_info = {"Provider": f"{resolved_workspace}/{provider}", "Model": model}
    else:
        # Model entity routing (default): use OpenAI-compatible gateway
        resolved_workspace, model_entity_id = _parse_model_and_workspace(model, workspace, workspace_from_config)

        def get_response(body: dict[str, Any]) -> StreamingResponse:
            return client.stream_openai(
                trailing_uri="v1/chat/completions",
                workspace=resolved_workspace,
                body=JsonBody(body),
            )

        model_for_body = model_entity_id
        display_info = {"Model": model_for_body}

    if run_once:
        _run_one_shot(
            user_message=cast(str, effective_prompt),  # UsageError above guarantees non-None.
            model_for_body=model_for_body,
            get_response_func=get_response,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            output_format=chat_output_format,
        )
    else:
        _run_chat_session(
            model_for_body=model_for_body,
            get_response_func=get_response,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message,
            user_message=effective_prompt,
            display_info=display_info,
        )


def _run_one_shot(
    user_message: str,
    model_for_body: str,
    get_response_func: Callable[[dict[str, Any]], StreamingResponse],
    temperature: float | None,
    max_tokens: int | None,
    system_message: str | None,
    output_format: ChatOutputFormat,
) -> None:
    """Send one chat message and emit the response in the requested format."""
    history: list[ChatMessage] = []
    if system_message:
        history.append({"role": "system", "content": system_message})

    if output_format == "text":
        _stream_one_shot_text(user_message, history, model_for_body, get_response_func, temperature, max_tokens)
    else:
        result = _process_one_shot_message(
            user_message, history, model_for_body, get_response_func, temperature, max_tokens
        )
        _print_one_shot_response(result, model_for_body, output_format)


def _run_chat_session(
    model_for_body: str,
    get_response_func: Callable[[dict[str, Any]], StreamingResponse],
    temperature: float | None,
    max_tokens: int | None,
    system_message: str | None,
    user_message: str | None,
    display_info: dict[str, str],
) -> None:
    """Run model chat through the shared terminal UI."""
    history: list[ChatMessage] = []
    if system_message:
        history.append({"role": "system", "content": system_message})
    current_user_input = ""

    def send_turn(user_input: str) -> StreamingResponse:
        nonlocal current_user_input
        current_user_input = user_input
        return _create_one_shot_response(
            user_input,
            history,
            model_for_body,
            get_response_func,
            temperature,
            max_tokens,
        )

    def record_assistant_message(assistant_message: str) -> None:
        history.extend(
            [
                {"role": "user", "content": current_user_input},
                {"role": "assistant", "content": assistant_message},
            ]
        )

    run_chat_tui(
        send_turn=send_turn,
        record_assistant_message=record_assistant_message,
        display_info=display_info,
        temperature=temperature,
        max_tokens=max_tokens,
        system_message=system_message,
        initial_message=user_message,
    )


def _create_one_shot_response(
    user_input: str,
    history: list[ChatMessage],
    model_for_body: str,
    get_response_func: Callable[[dict[str, Any]], StreamingResponse],
    temperature: float | None,
    max_tokens: int | None,
) -> StreamingResponse:
    """Build and send one chat request."""
    body = build_kwargs(
        model=model_for_body,
        messages=[*history, {"role": "user", "content": user_input}],
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return get_response_func(body)


def _process_one_shot_message(
    user_input: str,
    history: list[ChatMessage],
    model_for_body: str,
    get_response_func: Callable[[dict[str, Any]], StreamingResponse],
    temperature: float | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Send one chat message and collect the streamed response without Rich UI."""
    response = _create_one_shot_response(
        user_input, history, model_for_body, get_response_func, temperature, max_tokens
    )

    raw_message, usage = collect_stream_response(response)
    thinking_content, regular_content = parse_thinking(raw_message)
    return {
        "content": regular_content,
        "thinking": thinking_content,
        "raw": raw_message,
        "usage": usage,
    }


def _stream_one_shot_text(
    user_input: str,
    history: list[ChatMessage],
    model_for_body: str,
    get_response_func: Callable[[dict[str, Any]], StreamingResponse],
    temperature: float | None,
    max_tokens: int | None,
) -> None:
    """Send one chat message and stream plain text output."""
    response = _create_one_shot_response(
        user_input, history, model_for_body, get_response_func, temperature, max_tokens
    )
    stream_text_response(response)


def _print_one_shot_response(
    result: dict[str, Any], model_for_body: str, output_format: Literal["json", "raw"]
) -> None:
    """Print a one-shot response in a script-friendly format."""
    if output_format == "json":
        payload = {
            "content": result["content"],
            "thinking": result["thinking"],
            "model": model_for_body,
            "usage": result["usage"],
        }
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return

    typer.echo(result["raw"])
