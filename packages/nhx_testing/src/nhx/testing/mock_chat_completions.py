# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic chat-completions mock for ``pytest_httpserver``.

This module is intentionally service-agnostic. It provides:

- :class:`MockChatCompletionsHandler` — a Werkzeug ``(Request) -> Response``
  callable that routes by ``body["model"]`` and serves queued responses for
  each model.
- :class:`ChatCompletion`, :class:`ChatCompletionStream`, and
  :class:`ErrorResponse` — a tagged union of mock response shapes
  (non-streaming success, streaming SSE success, error).
- :func:`chat_completion` and :func:`chat_completion_chunk` — small builders
  for the common OpenAI-compatible body / chunk shapes so test files don't
  repeat themselves.

The handler is mounted at ``POST /v1/chat/completions`` on the
``pytest_httpserver`` ``HTTPServer`` instance owned by the test fixture. Both
the IGW proxy step and a plugin's outbound HTTP terminate at the same socket,
so a single instance covers both the upstream model call and any rail-side
calls a plugin makes from inside its hooks.
"""

import json
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from werkzeug import Request, Response

# ---------------------------------------------------------------------------
# Mock response shapes (tagged union)
# ---------------------------------------------------------------------------


@dataclass
class ChatCompletion:
    """A single non-streaming chat-completion response.

    Attributes:
        body: The OpenAI-compatible JSON body to return.
        status_code: HTTP status code (default 200).
        headers: Extra response headers. ``Content-Type`` defaults to
            ``application/json`` but tests may override it (e.g. to
            ``text/event-stream``) by including it here.
    """

    body: dict[str, Any]
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass
class ChatCompletionStream:
    """A streaming chat-completion response, rendered as SSE.

    Each entry in :attr:`chunks` is JSON-encoded and emitted as a
    ``data: {...}\\n\\n`` line. When :attr:`emit_done` is true (default) a
    final ``data: [DONE]\\n\\n`` line is appended — matching OpenAI's
    streaming protocol.

    Attributes:
        chunks: Streaming chunks to emit, in order.
        status_code: HTTP status code (default 200).
        headers: Extra response headers. ``Content-Type`` defaults to
            ``text/event-stream`` and ``Cache-Control: no-cache`` is also set
            unless overridden via *headers*.
        emit_done: Append ``data: [DONE]\\n\\n`` after the last chunk.
    """

    chunks: Sequence[dict[str, Any]]
    status_code: int = 200
    headers: Mapping[str, str] = field(default_factory=dict)
    emit_done: bool = True


@dataclass
class ErrorResponse:
    """An error response (status >= 400).

    Attributes:
        status_code: HTTP status code (must be >= 400).
        body: Error body. ``dict`` is serialized as JSON; ``str`` is sent as-is.
        headers: Extra response headers. ``Content-Type`` defaults to
            ``application/json`` for dict bodies and ``text/plain`` for str
            bodies, but tests may override either by including the header here.
    """

    status_code: int
    body: dict[str, Any] | str
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status_code < 400:
            raise ValueError(
                f"ErrorResponse.status_code must be >= 400 (got {self.status_code}); "
                "use ChatCompletion for success responses."
            )


MockResponse = ChatCompletion | ChatCompletionStream | ErrorResponse
"""One of :class:`ChatCompletion`, :class:`ChatCompletionStream`, or :class:`ErrorResponse`."""


# ---------------------------------------------------------------------------
# Response body builders
# ---------------------------------------------------------------------------


def chat_completion(
    *,
    content: str,
    model: str | None = None,
    id_: str | None = None,
    finish_reason: str = "stop",
    role: str = "assistant",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-streaming chat-completion response body.

    Args:
        content: The assistant message content.
        model: The model id echoed in the response body. Defaults to
            ``None``; :meth:`IGWPluginHarness.mock_chat_completions`
            replaces ``None`` with the dispatch key automatically. Only
            set explicitly when the test asserts on the response's
            ``model`` field.
        id_: Response id (emitted as ``"id"`` in the body). If ``None``, a
            random ``chatcmpl-...`` id is generated. The trailing underscore
            avoids shadowing the ``id`` builtin at the call site.
        finish_reason: ``stop``, ``length``, ``content_filter``, ``tool_calls``...
        role: Message role (default ``assistant``).
        extra: Optional dict merged into the top-level body (e.g. ``usage``).
    """
    body: dict[str, Any] = {
        "id": id_ or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": role, "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if extra:
        body.update(extra)
    return body


def chat_completion_chunk(
    *,
    content: str = "",
    model: str | None = None,
    id_: str | None = None,
    finish_reason: str | None = None,
    role: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single OpenAI streaming chunk (without the ``data:`` SSE prefix).

    The ``object`` field is always ``"chat.completion.chunk"`` to match the
    OpenAI streaming protocol. ``role`` should be set on the *first* chunk
    only (``"assistant"``) and omitted on subsequent chunks; ``finish_reason``
    should be set on the *last* chunk only (``"stop"`` etc.) and omitted
    elsewhere — passing ``None`` for either omits the field, matching what
    the OpenAI server emits.
    """
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content:
        delta["content"] = content

    choice: dict[str, Any] = {
        "index": 0,
        "delta": delta,
    }
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason

    chunk: dict[str, Any] = {
        "id": id_ or f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    if extra:
        chunk.update(extra)
    return chunk


def build_chat_completion_stream_from_content(content: str) -> ChatCompletionStream:
    """Build a :class:`ChatCompletionStream` that streams content in three delta chunks, plus a
    terminal chunk with finish_reason="stop".

    This function is useful for mocking streaming responses in tests without having to manually
    construct chunks for a given content string.
    """
    third = max(1, len(content) // 3)
    first_part = content[:third]
    second_part = content[third : 2 * third]
    third_part = content[2 * third :]

    stream_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    chunks = [
        chat_completion_chunk(content=first_part, role="assistant", id_=stream_id),
        chat_completion_chunk(content=second_part, id_=stream_id),
        chat_completion_chunk(content=third_part, id_=stream_id),
        chat_completion_chunk(content="", finish_reason="stop", id_=stream_id),
    ]
    return ChatCompletionStream(chunks=chunks)


def build_content_from_chunks(chunks: Sequence[dict[str, Any]]) -> str:
    """Join the ``delta.content`` of streaming chunks into a single string."""
    return "".join(
        (choice.get("delta") or {}).get("content") or "" for chunk in chunks for choice in (chunk.get("choices") or [])
    )


# ---------------------------------------------------------------------------
# Recorded request (for first-class assertions)
# ---------------------------------------------------------------------------


@dataclass
class RecordedRequest:
    """A request the handler observed.

    Attributes:
        model: The value of ``body["model"]`` at request time, or ``None`` if
            the body wasn't valid JSON or didn't contain a string ``model`` key.
        body: The parsed JSON body, or ``{}`` if the body wasn't valid JSON.
        headers: Header items as a list of ``(name, value)`` tuples — preserved
            as a list (not a dict) so duplicate-name headers (e.g.
            ``Set-Cookie``) survive intact for assertions.
        path: Request path (without query string).
    """

    model: str | None
    body: dict[str, Any]
    headers: list[tuple[str, str]]
    path: str

    def header(self, name: str) -> str | None:
        """Return the first header value matching *name* (case-insensitive)."""
        lower = name.lower()
        for k, v in self.headers:
            if k.lower() == lower:
                return v
        return None


# ---------------------------------------------------------------------------
# MockChatCompletionsHandler
# ---------------------------------------------------------------------------


@dataclass
class MockChatCompletionsHandler:
    """A ``(Request) -> Response`` callable for ``pytest_httpserver``.

    Routes incoming chat-completion requests by ``body["model"]`` and serves
    the queued response for that model. Tracks per-model call counts and a
    full request log so harness-level assertion helpers (and tests directly)
    can inspect what happened.

    Behaviour notes:

    - Responses for a model are consumed in order. By default, when only one
      response remains it is "clamped" — repeat calls reuse the last response.
      This keeps tests robust to upstream-call counts that vary by minor
      library changes (e.g. extra rail probes). Pass ``strict=True`` to
      :meth:`add_responses` to opt out: the queue then drains fully and any
      excess request returns HTTP 500 with a "no mock response queued" body.
    - A request whose ``body["model"]`` has no queued responses returns HTTP
      500 with a descriptive error body. The test should fail loudly rather
      than silently mismatching.
    - A request with a malformed body (no JSON or no string ``model``) is
      logged in :attr:`request_log` but **not** counted in
      :attr:`call_counts` / :attr:`call_order`, and returns HTTP 400.

    Use :meth:`add_responses` to register expected responses for a model.
    Tests typically don't instantiate this class directly — the harness owns
    one instance and exposes :meth:`IGWPluginHarness.mock_chat_completions`
    as the registration entry point.
    """

    responses_by_model: dict[str, deque[MockResponse]] = field(default_factory=dict)
    """Per-model FIFO of responses still to be served."""

    call_counts: dict[str, int] = field(default_factory=dict)
    """How many requests each model has received."""

    request_log: list[RecordedRequest] = field(default_factory=list)
    """Every request the handler observed, in order."""

    call_order: list[str] = field(default_factory=list)
    """The sequence of model values seen across requests (one entry per call)."""

    strict_models: set[str] = field(default_factory=set)
    """Models registered with ``strict=True``: the queue drains fully and excess calls 500."""

    def add_responses(
        self,
        model: str,
        responses: Sequence[MockResponse],
        *,
        strict: bool = False,
    ) -> None:
        """Append *responses* to the queue for *model*.

        Args:
            model: The ``body["model"]`` value to match against.
            responses: One or more mock responses. Must be non-empty.
            strict: When ``True``, the last queued response is **not**
                clamped — the queue drains fully and any further request
                returns HTTP 500. Defaults to ``False`` (clamp the last
                response, so an extra call reuses it).

        Raises:
            ValueError: If *responses* is empty.
        """
        if not responses:
            raise ValueError(
                f"MockChatCompletionsHandler.add_responses({model!r}, ...) was called with no responses. "
                "Provide at least one MockResponse, or skip registration entirely."
            )
        self.responses_by_model.setdefault(model, deque()).extend(responses)
        self.call_counts.setdefault(model, 0)
        if strict:
            self.strict_models.add(model)

    def reset(self) -> None:
        """Clear all queued responses, call counts, and request log."""
        self.responses_by_model.clear()
        self.call_counts.clear()
        self.request_log.clear()
        self.call_order.clear()
        self.strict_models.clear()

    # ------------------------------------------------------------------
    # Werkzeug handler protocol
    # ------------------------------------------------------------------

    def __call__(self, request: Request) -> Response:
        raw = request.get_data()
        try:
            body = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            body = {}

        model = body.get("model") if isinstance(body, dict) else None
        recorded = RecordedRequest(
            model=model if isinstance(model, str) else None,
            body=body if isinstance(body, dict) else {},
            headers=list(request.headers.items()),
            path=request.path,
        )
        self.request_log.append(recorded)

        if not isinstance(model, str):
            return Response(
                json.dumps(
                    {"error": "MockChatCompletionsHandler: request body did not contain a string 'model' field."}
                ),
                status=400,
                content_type="application/json",
            )

        self.call_order.append(model)
        self.call_counts[model] = self.call_counts.get(model, 0) + 1

        queue = self.responses_by_model.get(model)
        if not queue:
            registered = sorted(self.responses_by_model)
            return Response(
                json.dumps(
                    {
                        "error": (
                            f"MockChatCompletionsHandler: no mock response queued for model={model!r}. "
                            f"Registered models: {registered}."
                        )
                    }
                ),
                status=500,
                content_type="application/json",
            )

        # Strict: drain fully; non-strict: keep the last response so an extra
        # call reuses it (resilience against upstream-library churn).
        if model in self.strict_models or len(queue) > 1:
            mock = queue.popleft()
        else:
            mock = queue[0]

        return _render(mock)


def _render(mock: MockResponse) -> Response:
    if isinstance(mock, ChatCompletion):
        # Defaults first; user-supplied headers win on conflict so tests can
        # override Content-Type (e.g. for SSE) without fighting the helper.
        headers = {"Content-Type": "application/json", **dict(mock.headers)}
        return Response(
            json.dumps(mock.body),
            status=mock.status_code,
            headers=headers,
        )
    if isinstance(mock, ChatCompletionStream):
        # Capture the chunk list now so the closure doesn't accidentally bind
        # to a mutable Sequence the caller might modify later.
        chunks = list(mock.chunks)
        emit_done = mock.emit_done

        def _sse_body() -> Iterable[bytes]:
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            if emit_done:
                yield b"data: [DONE]\n\n"

        # Werkzeug accepts Iterable[bytes] via the ``response`` argument and
        # streams the body one chunk at a time — matching how real NIMs serve
        # SSE. Defaults first; user-supplied headers win on conflict.
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            **dict(mock.headers),
        }
        return Response(
            response=_sse_body(),
            status=mock.status_code,
            headers=headers,
        )
    if isinstance(mock, ErrorResponse):
        if isinstance(mock.body, str):
            body_text = mock.body
            default_content_type = "text/plain"
        else:
            body_text = json.dumps(mock.body)
            default_content_type = "application/json"
        headers = {"Content-Type": default_content_type, **dict(mock.headers)}
        return Response(body_text, status=mock.status_code, headers=headers)
    raise TypeError(f"Unknown MockResponse type: {type(mock).__name__}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Type aliases for ergonomic predicate-style assertions
# ---------------------------------------------------------------------------


BodyPredicate = Callable[[dict[str, Any]], bool]
"""A ``(body) -> bool`` callable used by ``IGWPluginHarness.assert_request_body_for``."""


__all__ = [
    "BodyPredicate",
    "ChatCompletion",
    "ChatCompletionStream",
    "ErrorResponse",
    "MockResponse",
    "MockChatCompletionsHandler",
    "RecordedRequest",
    "chat_completion",
    "chat_completion_chunk",
]
