# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo guardrail`` command group, backed by the typed Guardrail client."""

from __future__ import annotations

from typing import Annotated

import typer
from nemo_guardrails_plugin.cli_commands.configs import app as configs_app
from nemo_platform_ext.cli.core.api import build_kwargs
from nemo_platform_ext.cli.core.code_generator import handle_code_generation
from nemo_platform_ext.cli.core.context import CLIContext
from nemo_platform_ext.cli.core.errors import handle_errors
from nemo_platform_ext.cli.core.formatters import format_output
from nemo_platform_ext.cli.core.help_formatter import collect_warnings, create_typer_app
from nemo_platform_ext.cli.core.stdin_utils import (
    build_request_body,
    read_data_input_with_flags,
    read_payload,
    validate_required_fields,
)
from nemo_platform_ext.cli.core.types import EntityOutputFormatOption
from nemo_platform_plugin.guardrail.client import GuardrailClient
from nemo_platform_plugin.guardrail.types import GuardrailCheckRequest

app = create_typer_app(name="guardrail", help="Manage guardrail")

app.add_typer(configs_app, name="configs")


@app.command("check")
@collect_warnings
@handle_errors
def check_guardrail(
    ctx: typer.Context,
    workspace: Annotated[str | None, typer.Option("--workspace")] = None,
    messages: Annotated[
        str | None,
        typer.Option(
            "--messages", help="A list of messages comprising the conversation so far (JSON string) (required)"
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", help="The model to use for completion. Must be one of the available models. (required)"
        ),
    ] = None,
    frequency_penalty: Annotated[
        float | None,
        typer.Option(
            "--frequency-penalty",
            help="Positive values penalize new tokens based on their existing frequency in the text.",
        ),
    ] = None,
    function_call: Annotated[
        str | None,
        typer.Option(
            "--function-call",
            help="Deprecated in favor of tool_choice. 'none' means the model will not call a function and instead generates a message. 'auto' means the model can pick between generating a message or calling a function. Specifying a particular function via {'name': 'my_function'} forces the model to call that function. (JSON string)",
        ),
    ] = None,
    guardrails: Annotated[
        str | None, typer.Option("--guardrails", help="Guardrails specific options for the request. (JSON string)")
    ] = None,
    ignore_eos: Annotated[bool | None, typer.Option("--ignore-eos", help="Ignore the eos when running")] = None,
    logit_bias: Annotated[
        str | None,
        typer.Option(
            "--logit-bias",
            help="Modify the likelihood of specified tokens appearing in the completion. Maps token IDs (as strings) to bias values from -100 to 100. (JSON string)",
        ),
    ] = None,
    logprobs: Annotated[
        bool | None,
        typer.Option(
            "--logprobs",
            help="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message",
        ),
    ] = None,
    max_completion_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-completion-tokens",
            help="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Preferred over max_tokens for reasoning models.",
        ),
    ] = None,
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", help="The maximum number of tokens that can be generated in the chat completion."),
    ] = None,
    n: Annotated[
        int | None, typer.Option("--n", help="How many chat completion choices to generate for each input message.")
    ] = None,
    presence_penalty: Annotated[
        float | None,
        typer.Option(
            "--presence-penalty",
            help="Positive values penalize new tokens based on whether they appear in the text so far.",
        ),
    ] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            help="Constrains effort on reasoning for reasoning models. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.",
        ),
    ] = None,
    response_format: Annotated[
        str | None,
        typer.Option(
            "--response-format",
            help="Format of the response. Use {'type': 'json_object'} for JSON mode or {'type': 'json_schema', 'json_schema': {...}} for structured outputs. (JSON string)",
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", help="If specified, attempts to sample deterministically.")
    ] = None,
    stop: Annotated[
        str | None,
        typer.Option(
            "--stop", help="Up to 4 sequences where the API will stop generating further tokens. (JSON string)"
        ),
    ] = None,
    stream: Annotated[
        bool | None, typer.Option("--stream", help="If set, partial message deltas will be sent, like in ChatGPT.")
    ] = None,
    stream_options: Annotated[
        str | None,
        typer.Option(
            "--stream-options",
            help="Options for streaming response. Only set this when stream=True. Supports include_usage to receive token usage in the final stream chunk. (JSON string)",
        ),
    ] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", help="What sampling temperature to use, between 0 and 2.")
    ] = None,
    tool_choice: Annotated[
        str | None,
        typer.Option(
            "--tool-choice",
            help="Controls which (if any) tool is called by the model. 'none' means no tool is called, 'auto' lets the model decide, 'required' forces a tool call. (JSON string)",
        ),
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            help="A list of tools the model may call. Each tool is an object with a 'type' field and a 'function' definition. (JSON string)",
        ),
    ] = None,
    top_logprobs: Annotated[
        int | None,
        typer.Option("--top-logprobs", help="The number of most likely tokens to return at each token position."),
    ] = None,
    top_p: Annotated[
        float | None,
        typer.Option("--top-p", help="An alternative to sampling with temperature, called nucleus sampling."),
    ] = None,
    user: Annotated[
        str | None,
        typer.Option(
            "--user",
            help="A unique identifier representing your end-user, used by some providers for abuse monitoring.",
        ),
    ] = None,
    vision: Annotated[
        bool | None, typer.Option("--vision", help="Whether this is a vision-capable request with image inputs.")
    ] = None,
    input_file: Annotated[
        str | None,
        typer.Option("--input-file", help="Path to JSON file (use '-' for stdin)", rich_help_panel="Input Options"),
    ] = None,
    input_data: Annotated[
        str | None,
        typer.Option("--input-data", help="Input data for the request (JSON or YAML)", rich_help_panel="Input Options"),
    ] = None,
    output_format: EntityOutputFormatOption = None,
) -> None:
    """Chat completion for the provided conversation.

    [bold red]Required fields:[/] messages, model

    [green]Examples:[/]
    nemo guardrail check --input-file config.json
    nemo guardrail check --input-data '{"messages": {}, "model": "value"}'
    echo '{"json": "data"}' | nemo guardrail check --input-file -
    nemo guardrail check --<option> "value"
    """
    # Read base input (optional if all fields provided via flags)
    if input_file or input_data:
        input_payload = read_data_input_with_flags(input_file=input_file, input_data=input_data)
    else:
        input_payload = {}

    # Apply CLI flag overrides (flags take precedence)
    if workspace is not None:
        input_payload["workspace"] = workspace
    if messages is not None:
        input_payload["messages"] = read_payload("messages", messages)
    if model is not None:
        input_payload["model"] = model
    if frequency_penalty is not None:
        input_payload["frequency_penalty"] = frequency_penalty
    if function_call is not None:
        input_payload["function_call"] = read_payload("function_call", function_call)
    if guardrails is not None:
        input_payload["guardrails"] = read_payload("guardrails", guardrails)
    if ignore_eos is not None:
        input_payload["ignore_eos"] = ignore_eos
    if logit_bias is not None:
        input_payload["logit_bias"] = read_payload("logit_bias", logit_bias)
    if logprobs is not None:
        input_payload["logprobs"] = logprobs
    if max_completion_tokens is not None:
        input_payload["max_completion_tokens"] = max_completion_tokens
    if max_tokens is not None:
        input_payload["max_tokens"] = max_tokens
    if n is not None:
        input_payload["n"] = n
    if presence_penalty is not None:
        input_payload["presence_penalty"] = presence_penalty
    if reasoning_effort is not None:
        input_payload["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        input_payload["response_format"] = read_payload("response_format", response_format)
    if seed is not None:
        input_payload["seed"] = seed
    if stop is not None:
        input_payload["stop"] = read_payload("stop", stop)
    if stream is not None:
        input_payload["stream"] = stream
    if stream_options is not None:
        input_payload["stream_options"] = read_payload("stream_options", stream_options)
    if temperature is not None:
        input_payload["temperature"] = temperature
    if tool_choice is not None:
        input_payload["tool_choice"] = read_payload("tool_choice", tool_choice)
    if tools is not None:
        input_payload["tools"] = read_payload("tools", tools)
    if top_logprobs is not None:
        input_payload["top_logprobs"] = top_logprobs
    if top_p is not None:
        input_payload["top_p"] = top_p
    if user is not None:
        input_payload["user"] = user
    if vision is not None:
        input_payload["vision"] = vision
    # Validate required fields are present after merging
    validate_required_fields(
        input_payload,
        ["messages", "model"],
        "guardrail check",
        {
            "messages": "A list of messages comprising the conversation so far (JSON string) (required)",
            "model": "The model to use for completion. Must be one of the available models. (required)",
        },
    )

    body = build_request_body(
        GuardrailCheckRequest, input_payload, exclude={"workspace"}, command_name="guardrail check"
    )
    kwargs = build_kwargs(workspace=input_payload.get("workspace"), body=body)

    state: CLIContext = ctx.obj
    resolved_output_format = state.get_output_format(output_format)

    if handle_code_generation(GuardrailClient, "check_guardrail", kwargs, resolved_output_format, state):
        return

    result = state.typed_client(GuardrailClient).check_guardrail(**kwargs)

    format_output(
        result,
        is_list=False,
        output_format=resolved_output_format,
        no_truncate=state.get_no_truncate(),
        timestamp_format=state.get_timestamp_format(),
    )
