# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable terminal UI for OpenAI-compatible streaming chat commands."""

from __future__ import annotations

import json
import logging
import re
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Protocol

import click
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from nemo_platform_ext.cli.core.help_formatter import _get_terminal_width


class ExitAction(Enum):
    """Presentation text for leaving a chat TUI."""

    EXIT = ("exit", "Exit the chat", "Chat session ended")
    DETACH = ("detach", "Detach from the session", "Session detached")

    def __init__(self, verb: str, help_text: str, successful_exit_text: str) -> None:
        self.verb = verb
        self.help_text = help_text
        self.successful_exit_text = successful_exit_text


class StreamingResponse(Protocol):
    """Raw byte-streaming response, as returned by typed-client ``BinaryContent`` endpoints.

    ``stream()`` yields the byte chunks of the SSE body; the HTTP status is
    available through ``http_response`` once the stream has been entered.
    """

    def stream(self) -> AbstractContextManager[Iterator[bytes]]: ...

    @property
    def http_response(self) -> Any: ...


@dataclass
class ServerSentEvent:
    """One decoded ``text/event-stream`` event."""

    event: str | None = None
    data: str = ""


class SSEDecoder:
    """Minimal ``text/event-stream`` decoder (event/data fields, blank-line dispatch)."""

    def __init__(self) -> None:
        self._event: str | None = None
        self._data: list[str] = []

    def iter_bytes(self, chunks: Iterator[bytes]) -> Iterator[ServerSentEvent]:
        buffer = b""
        for chunk in chunks:
            buffer += chunk
            while True:
                newline = buffer.find(b"\n")
                if newline == -1:
                    break
                line = buffer[:newline]
                buffer = buffer[newline + 1 :]
                event = self._decode_line(line.rstrip(b"\r").decode("utf-8", errors="replace"))
                if event is not None:
                    yield event
        if buffer:
            event = self._decode_line(buffer.decode("utf-8", errors="replace"))
            if event is not None:
                yield event
        event = self._flush()
        if event is not None:
            yield event

    def _decode_line(self, line: str) -> ServerSentEvent | None:
        if not line:
            return self._flush()
        if line.startswith(":"):
            return None
        field, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None

    def _flush(self) -> ServerSentEvent | None:
        if self._event is None and not self._data:
            return None
        event = ServerSentEvent(event=self._event, data="\n".join(self._data))
        self._event = None
        self._data = []
        return event


SendTurn = Callable[[str], StreamingResponse]
RecordAssistantMessage = Callable[[str], None]


class _StreamingThinkingFilter:
    """Incrementally strip thinking tags while preserving visible content."""

    def __init__(self) -> None:
        self._pending = ""
        self._in_thinking = False

    def feed(self, text: str) -> str:
        """Return visible text that is safe to print immediately."""
        if not text:
            return ""

        data = self._pending + text
        self._pending = ""
        visible = ""

        while data:
            if self._in_thinking:
                close_index = data.find("</think>")
                if close_index >= 0:
                    data = data[close_index + len("</think>") :]
                    self._in_thinking = False
                    continue

                pending_length = _partial_tag_suffix_length(data, "</think>")
                self._pending = data[-pending_length:] if pending_length else ""
                break

            open_index = data.find("<think>")
            if open_index >= 0:
                visible += data[:open_index]
                data = data[open_index + len("<think>") :]
                self._in_thinking = True
                continue

            pending_length = _partial_tag_suffix_length(data, "<think>")
            visible += data[:-pending_length] if pending_length else data
            self._pending = data[-pending_length:] if pending_length else ""
            break

        return visible

    def finish(self) -> str:
        """Flush any buffered non-thinking text at the end of the stream."""
        if not self._pending:
            return ""

        pending = self._pending
        self._pending = ""
        if self._in_thinking:
            logging.debug("Dropping incomplete thinking content at end of streamed chat response")
            return ""

        return pending


USER_PANEL_WIDTH_RATIO = 0.8
ASSISTANT_PANEL_WIDTH_RATIO = 0.95
LIVE_REFRESH_RATE = 20
ANSI_MOVE_UP_CLEAR_LINE = "\033[1A\033[2K"
THINKING_TAG_PATTERN = r"<think>(.*?)</think>"
THINKING_UNCLOSED_PATTERN = r"<think>(.*?)$"

terminal_width = _get_terminal_width()
console = Console(width=terminal_width)


def run_chat_tui(
    *,
    send_turn: SendTurn,
    display_info: dict[str, str],
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_message: str | None = None,
    initial_message: str | None = None,
    record_assistant_message: RecordAssistantMessage | None = None,
    exit_action: ExitAction = ExitAction.EXIT,
) -> None:
    """Run the shared interactive chat UI against a caller-provided transport."""
    root_logger = logging.getLogger()
    initial_level = root_logger.level
    try:
        root_logger.setLevel(logging.CRITICAL)
        _print_welcome_header(display_info, temperature, max_tokens, system_message, exit_action)

        last_thinking_content = ""
        thinking_displayed = False

        try:
            if initial_message:
                result = _process_user_message(initial_message, send_turn)
                if result:
                    assistant_message, last_thinking_content = result
                    if record_assistant_message is not None:
                        record_assistant_message(assistant_message)

            while True:
                console.print()
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]", console=console).strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    _clear_prompt_line()
                    new_thinking_state = _handle_special_command(
                        user_input,
                        last_thinking_content,
                        thinking_displayed,
                        exit_action,
                    )
                    if new_thinking_state is not None:
                        thinking_displayed = new_thinking_state
                    continue

                _clear_prompt_line()
                result = _process_user_message(user_input, send_turn)
                if result:
                    assistant_message, last_thinking_content = result
                    thinking_displayed = False
                    if record_assistant_message is not None:
                        record_assistant_message(assistant_message)

        except (KeyboardInterrupt, EOFError):
            _exit_gracefully(exit_action)

    finally:
        root_logger.setLevel(initial_level)


def collect_stream_response(response: StreamingResponse) -> tuple[str, Any | None]:
    """Collect content chunks from an OpenAI-compatible streaming response."""
    raw_message = ""
    usage = None

    for delta in _iter_stream_deltas(response):
        if delta.get("usage") is not None:
            usage = delta["usage"]

        content = _stream_delta_content(delta)
        if content:
            raw_message += content

    return raw_message, usage


def stream_text_response(response: StreamingResponse) -> None:
    """Stream plain text output while hiding thinking markup."""
    thinking_filter = _StreamingThinkingFilter()

    for delta in _iter_stream_deltas(response):
        content = _stream_delta_content(delta)
        if not content:
            continue

        visible_content = thinking_filter.feed(content)
        if visible_content:
            click.echo(visible_content, nl=False)
            sys.stdout.flush()

    trailing_content = thinking_filter.finish()
    if trailing_content:
        click.echo(trailing_content, nl=False)
        sys.stdout.flush()
    click.echo()


def parse_thinking(text: str) -> tuple[str, str]:
    """Return thinking content and visible content parsed from think tags."""
    matches = re.findall(THINKING_TAG_PATTERN, text, re.DOTALL)
    regular_content = re.sub(THINKING_TAG_PATTERN, "", text, flags=re.DOTALL)
    thinking_parts = list(matches)

    unclosed_match = re.search(THINKING_UNCLOSED_PATTERN, regular_content, re.DOTALL)
    if unclosed_match:
        thinking_parts.append(unclosed_match.group(1))
        regular_content = re.sub(THINKING_UNCLOSED_PATTERN, "", regular_content, flags=re.DOTALL)

    if thinking_parts:
        return "\n\n".join(thinking_parts), regular_content.strip()

    return "", text


def _partial_tag_suffix_length(text: str, tag: str) -> int:
    max_length = min(len(text), len(tag) - 1)
    for length in range(max_length, 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0


def _iter_stream_deltas(response: StreamingResponse) -> Iterator[dict[str, Any]]:
    """Yield parsed OpenAI-compatible streaming delta payloads."""
    with response.stream() as chunks:
        for event in SSEDecoder().iter_bytes(chunks):
            if event.event == "error":
                raise click.ClickException(_format_streaming_error(response, event.data))
            if event.event is not None:
                continue
            if not event.data:
                continue
            if event.data.startswith("[DONE]"):
                break

            try:
                yield json.loads(event.data)
            except json.JSONDecodeError:
                logging.debug("Failed to parse JSON stream event: %s", event.data)


def _format_streaming_error(response: StreamingResponse, data: str) -> str:
    if data:
        return f"Streaming chat request failed: {data}"

    try:
        status_code = getattr(response.http_response, "status_code", None)
    except RuntimeError:
        status_code = None
    if isinstance(status_code, int):
        return f"Streaming chat request failed (HTTP {status_code})"
    return "Streaming chat request failed"


def _stream_delta_content(delta: dict[str, Any]) -> str:
    choices = delta.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return ""

    chunk_delta = choices[0].get("delta", {})
    if not isinstance(chunk_delta, dict):
        return ""

    content = chunk_delta.get("content")
    return content if isinstance(content, str) else ""


def _process_user_message(user_input: str, send_turn: SendTurn) -> tuple[str, str] | None:
    panel_width = int(terminal_width * USER_PANEL_WIDTH_RATIO)
    user_panel = Panel(
        user_input,
        title="[bold cyan]You[/bold cyan]",
        title_align="right",
        border_style="cyan",
        padding=(0, 1),
        width=panel_width,
    )
    console.print(Align.right(user_panel))

    response = send_turn(user_input)
    assistant_message, thinking_content = _stream_response(response)

    if assistant_message:
        return assistant_message, thinking_content

    logging.warning("Received empty response from API")
    console.print(Panel.fit("⚠ Received empty response from model", border_style="yellow", padding=(0, 1)))
    return None


def _clear_prompt_line() -> None:
    console.file.write(ANSI_MOVE_UP_CLEAR_LINE)
    console.file.flush()


def _exit_gracefully(exit_action: ExitAction) -> None:
    console.print("\n")
    console.print(Panel.fit(f"[bold]{exit_action.successful_exit_text}[/bold]", border_style="dim", padding=(0, 2)))
    raise click.exceptions.Exit(0)


def _handle_special_command(
    command: str,
    last_thinking: str,
    thinking_displayed: bool,
    exit_action: ExitAction,
) -> bool | None:
    if command in {"/thinking", "/t"}:
        if not last_thinking:
            console.print(Panel.fit("No reasoning content in the last response", border_style="yellow", padding=(0, 1)))
            return None

        if thinking_displayed:
            console.print(Panel.fit("Reasoning already displayed above", border_style="yellow dim", padding=(0, 1)))
            return True

        thinking_panel = Panel(
            Markdown(last_thinking),
            title="[bold yellow]💭 Reasoning[/bold yellow]",
            title_align="left",
            border_style="yellow dim",
            padding=(0, 1),
        )
        console.print(thinking_panel)
        return True

    if command in {"/help", "/h"}:
        console.print(
            Panel(
                "[cyan]/thinking[/cyan] - Show/hide model reasoning from last response\n"
                "[cyan]/help[/cyan] - Show this help message\n"
                f"[cyan]Ctrl+C[/cyan] - {exit_action.help_text}",
                title="[bold]Available Commands[/bold]",
                border_style="blue",
                padding=(0, 1),
            )
        )
        return None

    console.print(
        Panel.fit(
            f"Unknown command: {command}. Type /help for available commands.",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    return None


def _print_welcome_header(
    display_info: dict[str, str],
    temperature: float | None,
    max_tokens: int | None,
    system_message: str | None,
    exit_action: ExitAction,
) -> None:
    config_lines = [f"[cyan]{key}:[/cyan] {value}" for key, value in display_info.items()]

    if temperature is not None:
        config_lines.append(f"[cyan]Temperature:[/cyan] {temperature}")
    if max_tokens is not None:
        config_lines.append(f"[cyan]Max Tokens:[/cyan] {max_tokens}")
    if system_message:
        config_lines.append(f"[cyan]System:[/cyan] {system_message}")

    welcome_panel = Panel(
        "\n".join(config_lines),
        title="[bold green]🤖 NeMo Platform Chat Session[/bold green]",
        subtitle=f"Press Ctrl+C to {exit_action.verb}",
        border_style="green",
        padding=(1, 2),
    )
    console.print(welcome_panel)


def _is_inside_thinking_tag(message: str) -> bool:
    return message.count("<think>") > message.count("</think>")


def _create_thinking_preview(message: str) -> str:
    thinking_match = re.search(THINKING_UNCLOSED_PATTERN, message, re.DOTALL)
    current_thinking = thinking_match.group(1).strip() if thinking_match else ""

    if not current_thinking:
        return "[dim]💭 Thinking...[/dim]"

    lines = current_thinking.split("\n")
    preview = "\n".join(lines[-3:]) if len(lines) > 1 else current_thinking
    if len(preview) > 150:
        preview = "..." + preview[-150:]
    return f"[dim italic]{preview}[/dim italic]"


def _extract_display_text(message: str, inside_thinking: bool) -> str:
    display_text = re.sub(THINKING_TAG_PATTERN, "", message, flags=re.DOTALL)
    if inside_thinking:
        display_text = re.sub(THINKING_UNCLOSED_PATTERN, "", display_text, flags=re.DOTALL)
    return display_text.strip()


def _stream_response(response: StreamingResponse) -> tuple[str, str]:
    full_message = ""
    panel_width = int(terminal_width * ASSISTANT_PANEL_WIDTH_RATIO)

    with Live("", console=console, refresh_per_second=LIVE_REFRESH_RATE) as live:
        for delta in _iter_stream_deltas(response):
            content = _stream_delta_content(delta)
            if not content:
                continue

            full_message += content
            inside_thinking = _is_inside_thinking_tag(full_message)
            display_text = _extract_display_text(full_message, inside_thinking)

            if inside_thinking and not display_text:
                thinking_indicator = Panel(
                    _create_thinking_preview(full_message),
                    title="[bold yellow dim]💭 Reasoning[/bold yellow dim]",
                    title_align="left",
                    border_style="yellow dim",
                    padding=(0, 1),
                    width=panel_width,
                )
                live.update(Align.left(thinking_indicator))
            else:
                response_panel = Panel(
                    Markdown(display_text) if display_text else "",
                    title="[bold magenta]Assistant[/bold magenta]",
                    title_align="left",
                    border_style="magenta",
                    padding=(0, 1),
                    width=panel_width,
                )
                live.update(Align.left(response_panel))

    thinking_content, regular_content = parse_thinking(full_message)
    if thinking_content:
        console.print("[dim]💭 Reasoning available! Type [dim cyan]/thinking[/] to view it[/]", markup=True)

    return regular_content or full_message, thinking_content
