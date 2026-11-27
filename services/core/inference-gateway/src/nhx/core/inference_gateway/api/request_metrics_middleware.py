# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe logical inference requests at the client-facing ASGI boundary."""

import asyncio
import json
import re
from collections.abc import Iterator
from time import monotonic
from typing import Any

from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import metrics

_PROXY_PATH = re.compile(r"/v2/workspaces/[^/]+/(?:(?P<openai>openai)|provider/[^/]+|model/.+)/-/(?P<operation>.+)")
_PROXY_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}
_PROTOCOLS = {
    "chat/completions": "openai_chat",
    "completions": "openai_completions",
    "responses": "openai_responses",
    "messages": "anthropic_messages",
}
_OPENAI_CHOICE_PROTOCOLS = {"openai_chat", "openai_completions"}
_SSE_BOUNDARY = re.compile(rb"(?>\r\n|\r|\n){2}")
# Bound how much response data telemetry holds or parses. An oversized frame is
# skipped, and later frames can still supply output or termination evidence.
_MAX_OBSERVED_BODY_BYTES = 256 * 1024
_MAX_TRACKED_CHOICES = 128


def _json_object(data: bytes | bytearray) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (ValueError, RecursionError):
        return {}
    return payload if isinstance(payload, dict) else {}


class _SSEFrames:
    def __init__(self) -> None:
        self._pending = b""
        self._discarding = False
        self.skipped_frame = False

    def feed(self, body: bytes) -> Iterator[tuple[str, dict[str, Any]]]:
        data = self._pending + body
        start = 0
        for boundary in _SSE_BOUNDARY.finditer(data):
            if not self._discarding and boundary.start() - start <= _MAX_OBSERVED_BODY_BYTES:
                event = b""
                fields = []
                for line in data[start : boundary.start()].splitlines():
                    field, _, value = line.partition(b":")
                    if field == b"event":
                        event = value.removeprefix(b" ")
                    elif field == b"data":
                        fields.append(value.removeprefix(b" "))
                if event or fields:
                    yield event.decode("ascii", errors="replace"), _json_object(b"\n".join(fields))
            else:
                self.skipped_frame = True
            self._discarding = False
            start = boundary.end()
        if len(data) - start > _MAX_OBSERVED_BODY_BYTES:
            self._discarding = True
            self.skipped_frame = True
        # Preserve enough of a discarded frame to recognize a split delimiter.
        self._pending = data[max(start, len(data) - 3) :] if self._discarding else data[start:]


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_chat_output(delta: dict[str, Any]) -> bool:
    if _has_text(delta.get("content")) or _has_text(delta.get("refusal")):
        return True
    function = delta.get("function_call")
    if isinstance(function, dict) and (_has_text(function.get("name")) or _has_text(function.get("arguments"))):
        return True
    images = delta.get("images")
    if isinstance(images, list) and any(
        isinstance(image, dict) and isinstance(url := image.get("image_url"), dict) and _has_text(url.get("url"))
        for image in images
    ):
        return True
    calls = delta.get("tool_calls")
    return isinstance(calls, list) and any(
        isinstance(call, dict)
        and isinstance(function := call.get("function"), dict)
        and (_has_text(function.get("name")) or _has_text(function.get("arguments")))
        for call in calls
    )


class _ResponseObservation:
    def __init__(self, protocol: str) -> None:
        self.protocol = protocol
        self.status: int | None = None
        self.streaming = False
        self.completed = False
        self.disconnected = False
        self.first_output: float | None = None
        self._nonempty = False
        self._frames: _SSEFrames | None = None
        self._json: bytearray | None = None
        self._terminal_event = False
        self._seen_choices: set[int] = set()
        self._finished_choices: set[int] = set()
        self._unobserved_choice = False
        self._payload_outcome: str | None = None

    def start(self, message: Message) -> None:
        self.status = message["status"]
        headers = Headers(raw=message.get("headers", []))
        media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        self.streaming = media_type == "text/event-stream"
        encoding = headers.get("content-encoding", "identity").strip().lower()
        if 200 <= message["status"] < 300 and encoding in ("", "identity"):
            if self.streaming:
                self._frames = _SSEFrames()
            elif media_type == "application/json":
                self._json = bytearray()

    def _observe_choices(self, payload: dict[str, Any]) -> bool:
        useful = False
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return useful
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            reason = choice.get("finish_reason")
            if reason == "content_filter":
                self._payload_outcome = self._payload_outcome or "policy_blocked"
            index = choice.get("index", position)
            if type(index) is int and 0 <= index < _MAX_TRACKED_CHOICES:
                self._seen_choices.add(index)
                if _has_text(reason):
                    self._finished_choices.add(index)
            else:
                self._unobserved_choice = True
            delta = choice.get("delta")
            useful |= (
                _has_text(choice.get("text"))
                if self.protocol == "openai_completions"
                else isinstance(delta, dict) and _has_chat_output(delta)
            )
        return useful

    def _observe_payload(self, event: str, payload: dict[str, Any]) -> bool:
        kind = payload.get("type") or event
        error = payload.get("error")
        response_status = payload.get("status") if self.protocol == "openai_responses" and not self.streaming else None
        if isinstance(error, dict) and error.get("type") == "guardrails_violation":
            self._payload_outcome = self._payload_outcome or "policy_blocked"
            return False
        error_events = ("error", "response.failed")
        if (
            kind in error_events
            or event in error_events
            or response_status == "failed"
            or isinstance(error, dict)
            or _has_text(error)
        ):
            self._payload_outcome = "stream_error" if self.streaming else "server_error"
            return False
        if kind == "response.incomplete" or response_status == "incomplete":
            self._payload_outcome = self._payload_outcome or "incomplete"
            return False
        if self.protocol in _OPENAI_CHOICE_PROTOCOLS:
            return self._observe_choices(payload)
        if self.protocol == "openai_responses":
            self._terminal_event |= kind == "response.completed"
            if kind in (
                "response.output_text.delta",
                "response.refusal.delta",
                "response.function_call_arguments.delta",
            ):
                return _has_text(payload.get("delta"))
            if kind == "response.output_item.added":
                item = payload.get("item")
                return isinstance(item, dict) and item.get("type") == "function_call" and _has_text(item.get("name"))
        if self.protocol == "anthropic_messages":
            self._terminal_event |= kind == "message_stop"
            if kind == "content_block_delta":
                delta = payload.get("delta")
                return isinstance(delta, dict) and (
                    (delta.get("type") == "text_delta" and _has_text(delta.get("text")))
                    or (delta.get("type") == "input_json_delta" and _has_text(delta.get("partial_json")))
                )
            if kind == "content_block_start":
                block = payload.get("content_block")
                return isinstance(block, dict) and (
                    (block.get("type") == "text" and _has_text(block.get("text")))
                    or (block.get("type") == "tool_use" and _has_text(block.get("name")))
                )
        return False

    def body(self, message: Message, elapsed: float) -> None:
        self.completed = not message.get("more_body", False)
        if self.status is None or not 200 <= self.status < 300:
            return
        body: bytes = message.get("body", b"")
        if self._frames is not None:
            for event, payload in self._frames.feed(body):
                useful = self._observe_payload(event, payload)
                if (
                    useful
                    and self.first_output is None
                    and self._payload_outcome is None
                    and not self._frames.skipped_frame
                ):
                    self.first_output = elapsed
        elif not self.streaming:
            self._nonempty |= bool(body)
            if self._json is not None:
                if len(self._json) + len(body) <= _MAX_OBSERVED_BODY_BYTES:
                    self._json.extend(body)
                else:
                    self._json = None
            if self.completed and self._nonempty:
                if self._json is not None:
                    self._observe_payload("", _json_object(self._json))
                if self._payload_outcome is None:
                    self.first_output = elapsed

    def outcome(self) -> str:
        if not self.completed:
            if self.disconnected:
                return "cancelled"
            return "stream_error" if self.streaming else "server_error"
        if self.status is None or self.status >= 500 or self.status == 424:
            return "server_error"
        if self.status == 429:
            return "rate_limited"
        if self.status >= 400:
            return "client_error"
        if self.status >= 300:
            return "redirect"
        if self._payload_outcome is not None:
            return self._payload_outcome
        if self.streaming:
            if (
                self._frames is None
                or self._frames.skipped_frame
                or self.protocol == "other"
                or self._unobserved_choice
            ):
                return "unobserved"
            # The gateway synthesizes [DONE] after clean iterator EOF. Require
            # completion evidence from the actual output protocol for known APIs.
            if self.protocol in _OPENAI_CHOICE_PROTOCOLS:
                if not self._seen_choices or not self._seen_choices <= self._finished_choices:
                    return "incomplete"
            elif self.protocol in ("openai_responses", "anthropic_messages") and not self._terminal_event:
                return "incomplete"
        return "success"


class InferenceRequestMetricsMiddleware:
    """Record a single outcome per proxy request after the response is sent."""

    def __init__(self, app: ASGIApp, *, prefix: str = "") -> None:
        self._app = app
        self._prefix = prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in _PROXY_METHODS:
            await self._app(scope, receive, send)
            return
        path = scope["path"]
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path + "/"):
            path = path[len(root_path) :]
        match = _PROXY_PATH.fullmatch(path[len(self._prefix) :]) if path.startswith(self._prefix) else None
        if match is None:
            await self._app(scope, receive, send)
            return

        operation = match["operation"].strip("/")
        if (
            scope["method"] == "GET"
            and match["openai"]
            and (operation == "v1/models" or operation.startswith("v1/models/"))
        ):
            await self._app(scope, receive, send)
            return
        operation = operation.removeprefix("v1/").rsplit("/v1/", 1)[-1]
        if operation.startswith("responses/") and operation.count("/") == 1:
            operation = "responses"
        observation = _ResponseObservation(_PROTOCOLS.get(operation, "other"))
        started = monotonic()

        async def receive_observed() -> Message:
            message = await receive()
            if message["type"] == "http.disconnect":
                observation.disconnected = True
            return message

        async def send_observed(message: Message) -> None:
            try:
                await send(message)
            except OSError:
                observation.disconnected = True
                raise
            if message["type"] == "http.response.start":
                observation.start(message)
            elif message["type"] == "http.response.body":
                observation.body(message, monotonic() - started)

        try:
            await self._app(scope, receive_observed, send_observed)
        except (asyncio.CancelledError, ClientDisconnect):
            observation.disconnected = True
            raise
        finally:
            labels = {
                "protocol": observation.protocol,
                "streaming": observation.streaming,
                "outcome": observation.outcome(),
            }
            metrics.requests_counter.add(1, attributes=labels)
            if observation.first_output is not None:
                metrics.time_to_first_useful_output.record(observation.first_output, attributes=labels)
