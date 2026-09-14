# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Union

import aiohttp
from aiohttp import ClientSession
from fastapi import HTTPException, Request
from fastapi import status as http_status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jinja2 import Environment as JinjaEnvironment
from multidict import CIMultiDict, CIMultiDictProxy
from nemo_platform import AsyncNeMoPlatform
from nemo_platform.types.inference.virtual_model import VirtualModel as SDKVirtualModel
from nemo_platform_plugin.client.adapter import client_from_platform
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.client.errors import NotFoundError as ClientNotFoundError
from nemo_platform_plugin.inference_middleware import (
    BackendFormat,
    ImmediateResponse,
    InferenceMiddlewareContext,
    InferenceMiddlewareError,
    InferenceResponse,
)
from nemo_platform_plugin.refs import ENTITY_REF_PATTERN
from nemo_platform_plugin.secrets.client import AsyncSecretsClient
from nmp.common.entities.utils import parse_model_entity_ref
from nmp.core.inference_gateway.api.backend_format import resolve_backend_format
from nmp.core.inference_gateway.api.errors import (
    raise_model_entity_not_found,
    raise_no_providers_for_model_entity,
    raise_unresolved_provider_secret,
)
from nmp.core.inference_gateway.api.middleware_registry import (
    MiddlewareRegistry,
    build_inference_response,
    execute_post_response_middleware,
    execute_request_middleware,
    execute_response_middleware,
)
from nmp.core.inference_gateway.api.mock_provider import handle_mock_request, is_mock_provider
from nmp.core.inference_gateway.api.typed_request import build_inference_request
from pydantic import BaseModel

if TYPE_CHECKING:
    from nmp.core.inference_gateway.api.model_cache import ModelCache

ResponseResult = Union[dict[str, Any], AsyncIterator[dict[str, Any]]]
"""Either a fully-buffered JSON response (``dict``) or a lazy SSE stream
(``AsyncIterator[dict]``).  Mirrors the type defined in
``nemo_platform_plugin.inference_middleware`` and used throughout the middleware chain."""

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 4096

# A model name safe to blind-substring-replace inside an error body: the entity-ref
# shape (``[\w.-]`` segments with an optional ``/``), which both the entity store
# (ENTITY_REF_PATTERN) and HuggingFace repo ids conform to. Its character set has no
# JSON-structural characters, so replacing it can't corrupt a JSON body. See
# _rewrite_model_field_in_error_body.
_SAFE_MODEL_NAME = re.compile(ENTITY_REF_PATTERN)

# OpenAPI extra configuration for proxy endpoints that accept arbitrary JSON bodies.
# This is used to generate proper requestBody schema in OpenAPI spec for POST/PUT/PATCH
# methods, enabling the SDK to have a proper `body` parameter instead of requiring `extra_body`.
PROXY_OPENAPI_EXTRA: dict = {
    "requestBody": {
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                }
            }
        },
        "required": False,
    },
    "responses": {
        "200": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "additionalProperties": True,
                    }
                }
            }
        }
    },
}

REQUEST_HEADERS_TO_DROP = frozenset(
    {
        "host",  # upstream shouldn't see the host header of our server
        "authorization",  # any auth to upstreams should be from Secrets, not the request
        "x-api-key",  # like authorization: upstream auth comes from Secrets. Anthropic/Bedrock
        # backends read this header, so a client's placeholder (e.g. "not-used") would 401 upstream.
        "x-forwarded-host",  # some backends (e.g. LiteLLM) use this to alter routing
        "x-forwarded-proto",
        "x-forwarded-for",
    }
)
RESPONSE_HEADERS_TO_DROP = frozenset(
    {
        "date",  # fastapi adds its own date response header
    }
)
# On the error path, FastAPI builds a brand-new JSON body for the HTTPException; it does
# not reuse the upstream response body, so most upstream headers either describe a body
# we're not forwarding (Content-Length, Content-Type, ...) or are hop-by-hop and shouldn't
# cross this boundary at all. Rather than deny-listing every such header (and needing to
# remember to add the next one), only forward headers a caller actually needs.
ERROR_RESPONSE_HEADERS_TO_FORWARD = frozenset(
    {
        "retry-after",  # read by the evaluator's resilience layer to set its retry backoff floor
    }
)


def normalize_proxy_url(host_url: str, trailing_uri: str) -> str:
    """Construct the proxy URL, handling duplicate /v1 paths and leading slashes.

    If host_url ends with /v1 (or /v1/) and trailing_uri starts with v1/,
    strip the v1/ prefix from trailing_uri to avoid /v1/v1 duplication.

    Args:
        host_url: The base URL of the model provider
        trailing_uri: The path suffix to append (e.g., "v1/chat/completions" or "/v1/chat/completions")

    Returns:
        The normalized URL without duplicate /v1 segments or leading slashes
    """
    host_url = host_url.rstrip("/")
    trailing_uri = trailing_uri.lstrip("/")

    if host_url.endswith("/v1") and trailing_uri.startswith("v1/"):
        trailing_uri = trailing_uri[3:]  # Remove "v1/" prefix

    return f"{host_url}/{trailing_uri}"


@dataclass
class NextRequestInfo:
    """Information needed to make the next proxied request to an upstream service."""

    url: str
    """Target URL for the proxied request"""

    body: bytes | None
    """Request body bytes, if any"""

    headers: CIMultiDict[str]
    """HTTP headers for the proxied request"""

    method: str
    """HTTP method (GET, POST, etc.)"""

    query_params: dict[str, str]
    """Query parameters to include in the request"""


_DEFAULT_AUTH_HEADER_FORMAT = "Authorization: Bearer {{ auth_secret }}"
# Renders HTTP header values, not HTML. Autoescape would corrupt secrets
# containing characters like `&`, `<`, `>`, or quotes.
_JINJA_ENV = JinjaEnvironment(autoescape=False)  # noqa: S701  # nosec B701


def render_auth_header(secret_value: str, auth_header_format: str | None) -> tuple[str, str]:
    """Render an auth header name and value from a Jinja2 format template.

    The template must contain exactly one variable named ``auth_secret``, which is
    substituted with *secret_value* at render time.  If *auth_header_format* is
    ``None``, the default ``"Authorization: Bearer {{ auth_secret }}"`` is used.

    Args:
        secret_value: The raw API key / secret to inject into the template.
        auth_header_format: Jinja2 template string, e.g. ``"X-Api-Key: {{ auth_secret }}"``.

    Returns:
        ``(header_name, header_value)`` tuple ready to set on the outgoing request.
    """
    template_str = auth_header_format or _DEFAULT_AUTH_HEADER_FORMAT
    rendered = _JINJA_ENV.from_string(template_str).render(auth_secret=secret_value)
    header_name, _, header_value = rendered.partition(": ")
    return header_name, header_value


async def build_next_request(
    request: Request,
    host_url: str,
    trailing_uri: str,
    auth_token: str | None = None,
    auth_header_format: str | None = None,
    body: dict | None = None,
    default_extra_body: dict | None = None,
    default_extra_headers: dict | None = None,
    required_extra_body: dict | None = None,
    required_extra_headers: dict | None = None,
    request_headers: dict[str, str] | None = None,
) -> NextRequestInfo:
    """
    This function is meant to handle generic transformations from the user-request
    and build the foundation for the next upstream proxy request. Specific endpoints might need
    to do additional mutations after this function if they want to add auth, change the body, etc.
    Once those mutations have happened to this returned NextRequestInfo, the caller could then
    take NextRequestInfo and pass it into proxy_request.

    Args:
        request: The incoming FastAPI request
        host_url: The base URL of the model provider (e.g., "https://api.openai.com/v1")
        trailing_uri: The path suffix to append (e.g., "v1/chat/completions")
        auth_token: Raw secret value to inject into the auth header. When set,
            *auth_header_format* controls which header name and value format are used.
        auth_header_format: Jinja2 template string controlling the auth header, e.g.
            ``"X-Api-Key: {{ auth_secret }}"``. Defaults to
            ``"Authorization: Bearer {{ auth_secret }}"`` when ``None``.
        body: If the caller doesn't want to proxy the request's body,
              they can pass this argument instead. If not passed, we'll
              proxy the body when appropriate.
        default_extra_body: Default body parameters that can be overridden by user request
        default_extra_headers: Default headers that can be overridden by user request
        required_extra_body: Required body parameters that cannot be overridden by user request
        required_extra_headers: Required headers that cannot be overridden by user request
        request_headers: Override for the incoming request headers. When provided, these are
            used as the header source instead of ``request.headers``. Pass
            ``InferenceRequest.headers`` here to forward request-middleware header mutations
            to the backend. Headers in :data:`REQUEST_HEADERS_TO_DROP` are still removed
            regardless of source.

    Returns:
        NextRequestInfo with the prepared request data
    """
    next_url = normalize_proxy_url(host_url, trailing_uri)

    source_headers = request_headers if request_headers is not None else request.headers
    headers = CIMultiDict((k, v) for k, v in source_headers.items() if k.lower() not in REQUEST_HEADERS_TO_DROP)

    # Add default_extra_headers (request headers take precedence)
    if default_extra_headers:
        for key, value in default_extra_headers.items():
            if key not in headers:
                headers[key] = value

    # Add required_extra_headers (they always override everything)
    if required_extra_headers:
        for key, value in required_extra_headers.items():
            headers[key] = value

    body_bytes = await _get_body_bytes(
        request=request,
        body=body,
        default_extra_body=default_extra_body,
        required_extra_body=required_extra_body,
    )

    if body_bytes is not None:
        headers["content-length"] = str(len(body_bytes))

    if auth_token:
        header_name, header_value = render_auth_header(auth_token, auth_header_format)
        headers[header_name] = header_value

    return NextRequestInfo(
        url=next_url,
        body=body_bytes,
        headers=headers,
        method=request.method,
        query_params=dict(request.query_params),
    )


async def _get_body_bytes(
    request: Request,
    body: dict | None,
    default_extra_body: dict | None,
    required_extra_body: dict | None,
) -> bytes | None:
    """Build the request body bytes with merged extra parameters.

    Merge order (later values override earlier):
    1. default_extra_body - provides defaults that can be overridden
    2. incoming_body (from request or body param) - user's request values
    3. required_extra_body - enforced values that cannot be overridden

    Args:
        request: The incoming FastAPI request
        body: Optional pre-parsed body dict to use instead of reading from request
        default_extra_body: Default body parameters that can be overridden by user request
        required_extra_body: Required body parameters that cannot be overridden by user request

    Returns:
        Merged body as JSON bytes, or None for non-body HTTP methods
    """
    if request.method not in ["POST", "PUT", "PATCH"]:
        return None

    if body is not None:
        incoming_body = body
    else:
        incoming_body_bytes = await request.body()
        try:
            incoming_body = json.loads(incoming_body_bytes)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Request body is not JSON, using raw bytes")
            return incoming_body_bytes

    # Merge: default_extra_body < incoming_body < required_extra_body
    default_extra_body = default_extra_body or {}
    required_extra_body = required_extra_body or {}
    merged_body = {**default_extra_body, **incoming_body, **required_extra_body}
    return json.dumps(merged_body).encode()


def _filter_response_headers(headers: CIMultiDictProxy[str]) -> CIMultiDict[str]:
    """Filter out response headers that should not be forwarded to the client."""
    return CIMultiDict((k, v) for k, v in headers.items() if k.lower() not in RESPONSE_HEADERS_TO_DROP)


def _filter_error_response_headers(headers: CIMultiDictProxy[str]) -> CIMultiDict[str]:
    """Filter response headers for forwarding onto a synthesized error body.

    Unlike :func:`_filter_response_headers`, which drops a known-bad set and
    forwards everything else, this only forwards headers on
    :data:`ERROR_RESPONSE_HEADERS_TO_FORWARD`. The error path raises a new
    HTTPException with its own body rather than forwarding the upstream body
    byte-for-byte, so most upstream headers (framing headers describing that
    body, hop-by-hop headers, anything backend-internal) should not cross this
    boundary at all; an allowlist means a new upstream header type can never
    leak through by default the way a deny-list would require remembering to
    block it.
    """
    return CIMultiDict((k, v) for k, v in headers.items() if k.lower() in ERROR_RESPONSE_HEADERS_TO_FORWARD)


def _close_response(response: aiohttp.ClientResponse | None):
    """Close an aiohttp response if it's open."""
    if response and not response.closed:
        response.close()


_MAX_ERROR_BODY_LEN = 2048


async def _read_error_body(response: aiohttp.ClientResponse) -> str:
    """Read a truncated text snippet from an error response for diagnostic logging."""
    try:
        raw = await response.read()
        return raw.decode("utf-8", errors="replace")[:_MAX_ERROR_BODY_LEN]
    except Exception:
        return ""


async def proxy_request(http_client: ClientSession, next_request_info: NextRequestInfo) -> StreamingResponse:
    """Execute a proxied HTTP request and stream the response back to the client.

    This function forwards the request to an upstream service and streams the response
    back without buffering. It handles both regular and server-sent event responses.

    Certain backend errors (401/403/404) are wrapped in 502 Bad Gateway with a clear
    error message indicating the error came from the backend, not the gateway. Other
    backend errors (429, 422, 5xx, etc.) are passed through with their original status.
    In both cases the backend's error body is included in the detail for diagnostics.

    Args:
        http_client: The HTTP client session to use for the request
        next_request_info: Information about the request to proxy

    Returns:
        StreamingResponse containing the proxied response

    Raises:
        HTTPException: 502 for certain backend errors (401/403/404) or network errors,
            original status for other backend errors, 500 for internal errors
    """
    response: aiohttp.ClientResponse | None = None
    try:
        response = await http_client.request(
            next_request_info.method,
            url=next_request_info.url,
            headers=next_request_info.headers,
            params=next_request_info.query_params,
            data=next_request_info.body,
            timeout=None,
        )

        if response.status >= 400:
            error_body = await _read_error_body(response)
            error_headers = dict(_filter_error_response_headers(response.headers))
            _close_response(response)
            logger.warning(
                "Backend error %d from %s: %s",
                response.status,
                next_request_info.url,
                error_body,
            )
            if response.status in (401, 403, 404):
                detail = (
                    f"Backend returned {response.status}: {error_body}"
                    if error_body
                    else f"Backend returned {response.status}"
                )
                raise HTTPException(status_code=502, detail=detail, headers=error_headers)
            else:
                detail = error_body if error_body else str(response.status)
                raise HTTPException(status_code=response.status, detail=detail, headers=error_headers)

        response_headers = _filter_response_headers(response.headers)

        # The original implementation branched Response/StreamingResponse
        # based on resp_headers['content-type'] being 'text/event-stream'. However, always
        # returning StreamingResponse doesn't seem to cause issues for non-sse
        # requests as of now. It also has the advantage of not needing to buffer
        # the entire response before sending data back to the client.
        # I could see us needing to change this in the future,
        # but this keeps it simple for now.
        async def event_stream_generator():
            try:
                async for chunk in response.content.iter_chunked(DEFAULT_CHUNK_SIZE):
                    yield chunk
            finally:
                # Note: we explicitly don't use a `with` block for the request to manage
                # the lifecycle of the response. If we did, the __exit__ would be called when we
                # return the StreamingResponse, which would close things before we even start
                # streaming the data. That means we need to handle cleaning up the response ourselves
                # in this try/finally.
                _close_response(response)

        return StreamingResponse(
            event_stream_generator(),
            status_code=response.status,
            headers=response_headers,
        )
    except HTTPException:
        raise
    except aiohttp.ClientError as e:
        # Network error (connection failed, timeout, DNS, etc.)
        _close_response(response)
        raise HTTPException(status_code=502, detail=f"Backend networking error: {e}")
    except Exception as e:
        _close_response(response)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


async def _parse_sse_chunks(
    chunks: AsyncIterable[bytes | bytearray | memoryview | str],
) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed JSON objects from SSE chunks.

    Accepts ``AsyncIterable`` (not just ``AsyncIterator``) and any of the
    byte-like types so that ``StreamingResponse.body_iterator`` (declared as
    ``AsyncIterable[str | bytes | memoryview[int]]``) can be passed without a
    cast at the call site.
    """
    buffer = ""
    async for chunk in chunks:
        if isinstance(chunk, str):
            buffer += chunk
        else:
            buffer += bytes(chunk).decode("utf-8", errors="replace")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                return
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield parsed


async def _parse_sse_stream(response: aiohttp.ClientResponse) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed JSON objects from an SSE (text/event-stream) response.

    Each ``data:`` line is decoded and yielded as a dict. ``data: [DONE]``
    terminates the stream. Malformed or non-JSON data lines are silently skipped.
    aiohttp decodes compressed response chunks before they reach this parser.
    The underlying response is always closed when the generator exits.
    """
    try:
        async for parsed in _parse_sse_chunks(response.content.iter_any()):
            yield parsed
    finally:
        _close_response(response)


async def fetch_proxy_response(
    http_client: ClientSession,
    next_request_info: NextRequestInfo,
    served_model_name: str | None = None,
    restored_model_id: str | None = None,
) -> tuple[ResponseResult, CIMultiDict[str], int]:
    """Execute a proxied HTTP request and return the response as a ``ResponseResult``.

    Unlike :func:`proxy_request`, this function does **not** stream the response
    directly to the client.  The caller receives a :data:`ResponseResult` and is
    responsible for applying response middleware and then streaming via
    :func:`stream_response_result`.

    aiohttp decodes the upstream ``Content-Encoding`` for this middleware-aware
    path. ``Content-Type: text/event-stream`` responses remain backed by the live
    response, so the iterator **must** be fully consumed or the connection will
    leak. Other responses are buffered and parsed as JSON.

    When *served_model_name* and *restored_model_id* are both supplied, upstream
    error bodies (a 4xx/5xx *status* response) have the served model name rewritten
    to the entity ref before the body is placed on the ``HTTPException`` — matching
    the happy-path rewrite so ``served_model_name`` never leaks in these error cases.
    See :func:`_rewrite_model_field_in_error_body`. This is best-effort UX scrubbing,
    not a hard guarantee: an error a provider delivers *inside* an HTTP-200 SSE stream
    (an error event mid-stream) does not pass through here — such events only have
    their structured ``model`` fields rewritten by
    :func:`_rewrite_model_field_in_stream`, so a served name embedded in that event's
    free-form message string is not scrubbed.

    Returns:
        A ``(response_result, headers, status_code)`` tuple.

    Raises:
        HTTPException: Same conditions as :func:`proxy_request`.
    """
    response: aiohttp.ClientResponse | None = None
    try:
        response = await http_client.request(
            next_request_info.method,
            url=next_request_info.url,
            headers=next_request_info.headers,
            params=next_request_info.query_params,
            data=next_request_info.body,
            timeout=None,
            auto_decompress=True,
        )

        if response.status >= 400:
            error_body = await _read_error_body(response)
            error_headers = dict(_filter_error_response_headers(response.headers))
            _close_response(response)
            logger.warning(
                "Backend error %d from %s: %s",
                response.status,
                next_request_info.url,
                error_body,
            )
            # Rewrite the served model name out of the error body so the caller
            # never sees the upstream's served_model_name for a non-2xx *status*
            # response, matching the happy-path rewrite. (An error delivered inside
            # an HTTP-200 SSE stream does not reach here; see the docstring.)
            if served_model_name is not None and restored_model_id is not None:
                error_body = _rewrite_model_field_in_error_body(error_body, served_model_name, restored_model_id)
            if response.status in (401, 403, 404):
                detail = (
                    f"Backend returned {response.status}: {error_body}"
                    if error_body
                    else f"Backend returned {response.status}"
                )
                raise HTTPException(status_code=502, detail=detail, headers=error_headers)
            else:
                detail = error_body if error_body else str(response.status)
                raise HTTPException(status_code=response.status, detail=detail, headers=error_headers)

        response_headers = _filter_response_headers(response.headers)
        status_code = response.status
        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            # Ownership of the live aiohttp response is transferred to the
            # generator for streaming responses.
            result: ResponseResult = _parse_sse_stream(response)
        else:
            # aiohttp has already decoded Content-Encoding for this request.
            raw = await response.read()
            _close_response(response)
            try:
                result = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Inference Gateway could not parse upstream response from %s as JSON",
                    next_request_info.url,
                    exc_info=exc,
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Inference Gateway could not parse the upstream response as JSON "
                        "for inference middleware processing."
                    ),
                ) from exc
            if not isinstance(result, dict):
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Inference Gateway received an upstream JSON response that is not an object, "
                        "which inference middleware cannot process."
                    ),
                )

        return result, response_headers, status_code

    except HTTPException:
        raise
    except aiohttp.ClientPayloadError as exc:
        _close_response(response)
        logger.warning(
            "Inference Gateway could not decode upstream response from %s",
            next_request_info.url,
            exc_info=exc,
        )
        raise HTTPException(
            status_code=502,
            detail="Inference Gateway could not decode the upstream response body.",
        ) from exc
    except aiohttp.ClientError as exc:
        _close_response(response)
        raise HTTPException(status_code=502, detail=f"Backend networking error: {exc}")
    except Exception as exc:
        _close_response(response)
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


# Headers that describe the upstream body encoding/framing and must be dropped
# when stream_response_result re-serializes the payload.  Forwarding them after
# body mutation produces truncated responses (wrong content-length), failed
# decompression (stale content-encoding), and framing errors (transfer-encoding).
_BODY_HEADERS_TO_STRIP = frozenset({"content-length", "content-encoding", "transfer-encoding", "content-type"})


def _rewrite_model_field(payload: Any, served_model_name: str, restored_model_id: str) -> None:
    """Rewrite ``served_model_name`` -> ``restored_model_id`` wherever ``model`` may appear in *payload*.

    Operates in place on dict-shaped payloads. Covers the three locations across
    OpenAI- and Anthropic-shaped responses (both buffered and per-chunk) where the
    upstream model identifier is surfaced:

    * Top-level ``payload["model"]`` — OpenAI Chat Completions, OpenAI Completions,
      OpenAI ``ChatCompletionChunk`` (every streamed chunk has ``model``), and
      non-streaming Anthropic Messages.
    * ``payload["message"]["model"]`` — Anthropic Messages streaming
      ``message_start`` event embeds a ``Message`` object that carries ``model``
      (subsequent ``content_block_*``/``message_delta``/``message_stop``/``ping``
      events do not).
    * ``payload["response"]["model"]`` — OpenAI Responses API events
      (``response.created``/``response.in_progress``/``response.completed`` etc.)
      embed a ``Response`` object that carries ``model``.

    The rewrite is a strict equality match on *served_model_name* — values that
    don't match are left alone. This keeps the function safe to apply to chunks
    where the field is absent (we no-op) or where the upstream returned a value
    we didn't seed (also no-op, with a debug log so unexpected drift surfaces).
    """
    if not isinstance(payload, dict):
        return

    if payload.get("model") == served_model_name:
        payload["model"] = restored_model_id

    nested_message = payload.get("message")
    if isinstance(nested_message, dict) and nested_message.get("model") == served_model_name:
        nested_message["model"] = restored_model_id

    nested_response = payload.get("response")
    if isinstance(nested_response, dict) and nested_response.get("model") == served_model_name:
        nested_response["model"] = restored_model_id


def _rewrite_model_field_in_error_body(
    error_body: str,
    served_model_name: str,
    restored_model_id: str,
) -> str:
    """Rewrite ``served_model_name`` -> ``restored_model_id`` inside an upstream error body.

    Error responses never reach :func:`_rewrite_model_field` on the happy path
    because :func:`fetch_proxy_response` raises before the response body is handed
    to the middleware pipeline. Without this, the upstream ``served_model_name``
    leaks in error cases (rate-limit messages, content-policy blocks, "model not
    found", etc.). This restores parity with the happy-path rewrite.

    This is **best-effort UX scrubbing**, not a guarantee. It does a single
    substring replacement over the raw body, so it is format-agnostic — it works
    the same whether the body is a JSON object, plain text, or something we can't
    parse — and, because it makes exactly one pass over the *original* string, it
    cannot double-apply the swap (a parse-then-blanket-replace approach mangles
    ``model-1`` → ``default/model-1`` → ``default/default/model-1`` whenever the
    restored id contains the served name, which is the common case).

    To keep that blind replacement from corrupting a JSON body, both names must be
    plain model ids — they are matched against :data:`ENTITY_REF_PATTERN`
    (``[\\w.-]`` segments with an optional ``/``), the same shape the entity store
    and HuggingFace both use, whose character set contains no JSON-structural
    characters (no quotes, braces, brackets, colons, backslashes). If either name
    falls outside that shape (or is empty), the scrub is skipped entirely and the
    body is returned untouched — the served name persists rather than risk mangling
    the response.
    """
    if not error_body or not served_model_name:
        return error_body

    if not (_SAFE_MODEL_NAME.match(served_model_name) and _SAFE_MODEL_NAME.match(restored_model_id)):
        logger.warning(
            "Skipping error-body model-name scrub: a model name is not a plain model id "
            "(served=%r restored=%r); leaving the body untouched to avoid corrupting it.",
            served_model_name,
            restored_model_id,
        )
        return error_body

    return error_body.replace(served_model_name, restored_model_id)


async def _rewrite_model_field_in_stream(
    chunks: AsyncIterator[dict[str, Any]],
    served_model_name: str,
    restored_model_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Wrap an SSE chunk iterator and rewrite the ``model`` field in each chunk.

    Mutates each chunk in place via :func:`_rewrite_model_field` before yielding.
    The wrapper is transparent: malformed or non-dict chunks (which
    :func:`_parse_sse_chunks` already filters out) would still pass through
    unchanged if they reached us.
    """
    async for chunk in chunks:
        _rewrite_model_field(chunk, served_model_name, restored_model_id)
        yield chunk


def _strip_body_headers(headers: CIMultiDict[str] | dict) -> dict[str, str]:
    """Return a copy of *headers* with body-framing headers removed."""
    return {k: v for k, v in dict(headers).items() if k.lower() not in _BODY_HEADERS_TO_STRIP}


def _active_response_result(response_result: Any) -> Any:
    if isinstance(response_result, InferenceResponse):
        return response_result.typed_body if response_result.typed_body is not None else response_result.result
    return response_result


def _is_streaming_response_result(response_result: Any) -> bool:
    if isinstance(response_result, InferenceResponse):
        return not isinstance(response_result.result, dict)
    return not isinstance(response_result, dict | BaseModel)


def _serialization_response_result(response_result: Any) -> Any:
    if isinstance(response_result, InferenceResponse) and _is_streaming_response_result(response_result):
        return response_result.result
    return _active_response_result(response_result)


def _json_ready_payload(payload: Any) -> Any:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


def _build_inference_response_with_annotations(inference_response: InferenceResponse) -> InferenceResponse:
    annotations = inference_response.response_body_annotations

    if not annotations:
        return inference_response

    response_result = inference_response.result
    # For streaming responses, annotations are accumulated but not serialized
    # in the initial implementation.
    if not isinstance(response_result, dict):
        return inference_response

    if isinstance(inference_response.typed_body, BaseModel):
        result_with_annotations: dict[str, Any] = inference_response.typed_body.model_dump(mode="json")
    else:
        result_with_annotations = dict(response_result)

    # Merge annotations into the result, only if the key is not already present.
    result_with_annotations.update(
        {key: value for key, value in annotations.items() if key not in result_with_annotations}
    )

    return InferenceResponse(
        result=result_with_annotations,
        headers=dict(inference_response.headers),
    )


async def stream_response_result(
    response_result: Any,
    status_code: int,
    headers: CIMultiDict[str] | dict,
) -> StreamingResponse:
    """Convert a response result or envelope back into a :class:`StreamingResponse`.

    - ``dict`` → streamed as a single JSON body.
    - Pydantic model → dumped as JSON and streamed as a single JSON body.
    - ``AsyncIterator`` → re-encoded as SSE (``data: {...}\\n\\n`` per chunk,
      terminated with ``data: [DONE]\\n\\n``). Pydantic chunks are dumped with
      ``mode="json"`` before serialization.

    Body-framing headers (``content-length``, ``content-encoding``,
    ``transfer-encoding``, ``content-type``) are stripped from *headers* and
    replaced with values that match the re-serialized payload.  Forwarding the
    upstream values after body mutation would produce truncated or garbled
    responses for the caller.
    """
    safe_headers = _strip_body_headers(headers)
    response_payload = _serialization_response_result(response_result)

    if isinstance(response_payload, dict | BaseModel):
        encoded = json.dumps(_json_ready_payload(response_payload)).encode()

        async def _json_gen():
            yield encoded

        return StreamingResponse(
            _json_gen(),
            status_code=status_code,
            headers={**safe_headers, "content-type": "application/json"},
        )
    else:
        # AsyncIterator — re-encode as SSE
        async def _sse_gen():
            async for chunk in response_payload:
                yield f"data: {json.dumps(_json_ready_payload(chunk))}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            _sse_gen(),
            status_code=status_code,
            headers={**safe_headers, "content-type": "text/event-stream"},
        )


async def virtual_model_proxy(
    *,
    request: Request,
    workspace: str,
    vm_name: str,
    virtual_model: "SDKVirtualModel",
    trailing_uri: str,
    json_body: dict[str, Any],
    http_client: ClientSession,
    model_cache: "ModelCache",
    registry: "MiddlewareRegistry",
    request_nemo_client: AsyncNemoClient | None = None,
) -> Response:
    """Execute the full VirtualModel middleware pipeline and return a streaming response.

    Shared implementation for both ``openai_proxy`` and ``model_entity_proxy``.
    The caller is responsible for:

    - Looking up the :class:`~nmp.core.inference_gateway.api.virtual_model_cache.VirtualModelCache`
      and confirming the VM exists.
    - Providing *json_body* — the already-parsed request body dict.  Pass ``{}``
      for bodyless methods (e.g. GET).

    Pipeline: seed ``body["model"]`` → build context → request middleware → resolve model entity
    → proxy with served model name → restore model entity → response middleware →
    :func:`stream_response_result` → post-response fire-and-forget (non-streaming only).

    If the VirtualModel is in :attr:`MiddlewareRegistry.broken_vms` (because a
    referenced ``config_id`` was deleted upstream, a plugin failed validation, or
    a referenced plugin isn't loaded) the request short-circuits with a 503
    instead of silently bypassing the middleware chain. Recovery is automatic
    once the next IGW polling cycle re-resolves the VM cleanly.
    """
    if (workspace, vm_name) in registry.broken_vms:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Middleware configuration unavailable for VirtualModel "
                f"'{workspace}/{vm_name}'. A referenced config may have been "
                "deleted, or a plugin failed to validate it."
            ),
        )

    request_middleware_calls = registry.request_middleware_calls.get((workspace, vm_name), [])
    logger.debug(
        "virtual_model_proxy entry: workspace=%s vm_name=%s body_model_in=%r "
        "vm_default_model_entity=%r request_middleware_count=%d",
        workspace,
        vm_name,
        json_body.get("model"),
        virtual_model.default_model_entity,
        len(request_middleware_calls),
    )

    # Seed body["model"] if a default is set on the virtual model.
    if virtual_model.default_model_entity:
        json_body["model"] = virtual_model.default_model_entity

    # Build per-request context.  original_request captures the state after model
    # seeding but before any plugin runs; plugins receive a separate InferenceRequest
    # instance so mutations don't affect the snapshot.
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    original_request = build_inference_request(
        body=dict(json_body),
        headers=dict(request.headers),
        path=trailing_uri,
    )
    ctx = InferenceMiddlewareContext(
        request_id=request_id,
        virtual_model_name=vm_name,
        workspace=workspace,
        original_request=original_request,
        request_nemo_client=request_nemo_client,
    )
    initial_request = build_inference_request(
        body=json_body,
        headers=dict(request.headers),
        path=trailing_uri,
    )

    # Request middleware chain.
    try:
        modified_request = await execute_request_middleware(
            request_middleware_calls, registry.plugins, ctx, initial_request
        )
    except InferenceMiddlewareError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    backend_format = ctx.backend_format or BackendFormat.OPENAI_CHAT
    ctx.backend_format = backend_format

    # ImmediateResponse → skip proxy.
    if isinstance(modified_request, ImmediateResponse):
        inference_response = InferenceResponse(
            result=modified_request.data,
            headers={},
            response_body_annotations={
                **ctx.response_body_annotations,
                **modified_request.response_body_annotations,
            },
        )
        response_status = 200
    else:
        # Resolve body["model"] → model entity → provider → proxy.
        json_body = modified_request.body
        proxy_path = modified_request.path  # middleware may have rewritten the path
        try:
            # Use model-entity-aware parsing: split on the first '/' only so LoRA
            # composite ids (e.g. "ws/base&adapters/adapter_ws/adapter") survive
            # intact as the entity name. Matches ModelCache.rebuild_model_entity_map.
            modified_model_ref = parse_model_entity_ref(json_body["model"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                http_status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Could not resolve model entity from body['model'] after request middleware: {exc}",
            ) from exc

        resolved_model_entity = model_cache.get_from_model_entity(modified_model_ref.workspace, modified_model_ref.name)
        if resolved_model_entity is None:
            raise_model_entity_not_found(modified_model_ref.workspace, modified_model_ref.name)
        if not resolved_model_entity.model_providers:
            raise_no_providers_for_model_entity(modified_model_ref.workspace, modified_model_ref.name)

        backend_format = (
            resolve_backend_format(resolved_model_entity, virtual_model)
            or ctx.backend_format
            or BackendFormat.OPENAI_CHAT
        )
        ctx.backend_format = backend_format
        resolved_served_model_name, resolved_model_provider_info = resolved_model_entity.model_providers[0]

        if (
            resolved_model_provider_info.model_provider.api_key_secret_name
            and not resolved_model_provider_info.secret_value
        ):
            raise_unresolved_provider_secret(
                resolved_model_provider_info.model_provider.workspace,
                resolved_model_provider_info.model_provider.name,
            )

        if is_mock_provider(resolved_model_provider_info.model_provider.name):
            # Pass json_body explicitly so that middleware body mutations are visible to
            # the mock handler (consistent with the real-backend path).  body["model"] is
            # still the entity ID here — the served-model rewrite has not happened yet —
            # which is correct because the mock-response-map is keyed by entity ID.
            # See test_openai_router::test_virtual_model_proxy_mock_provider_keeps_qualified_body_model.
            ctx.proxied_request = build_inference_request(
                body=dict(json_body),
                headers=dict(modified_request.headers),
                path=proxy_path,
            )
            mock_response = await handle_mock_request(
                request=request,
                trailing_uri=proxy_path,
                default_extra_headers=resolved_model_provider_info.model_provider.default_extra_headers,
                request_body=json_body,
            )
            # Match fetch_proxy_response error semantics: 401/403/404 → 502, else passthrough.
            if mock_response.status_code >= 400:
                if mock_response.status_code in (401, 403, 404):
                    raise HTTPException(
                        status_code=502,
                        detail=f"Backend returned {mock_response.status_code}",
                    )
                raise HTTPException(
                    status_code=mock_response.status_code,
                    detail=str(mock_response.status_code),
                )
            if isinstance(mock_response, StreamingResponse):
                proxy_response_result = _parse_sse_chunks(mock_response.body_iterator)
                response_headers = CIMultiDict(mock_response.headers)
                response_status = mock_response.status_code
            elif isinstance(mock_response, JSONResponse):
                # JSONResponse.body is typed as Any | bytes | memoryview; the latter
                # isn't directly accepted by json.loads, so normalize to bytes first.
                response_body = mock_response.body
                if isinstance(response_body, memoryview):
                    response_body = bytes(response_body)
                proxy_response_result = json.loads(response_body)
                # Mirror the dict guard from fetch_proxy_response so mock and real
                # backends share the same contract.
                if not isinstance(proxy_response_result, dict):
                    raise HTTPException(
                        status_code=502,
                        detail="Inference middleware requires JSON object upstream responses; mock provider returned a non-object.",
                    )
                response_headers = CIMultiDict(mock_response.headers)
                response_status = mock_response.status_code
            else:
                # handle_mock_request is typed JSONResponse | StreamingResponse;
                # any other return type bypasses response middleware and is a contract violation.
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"handle_mock_request returned unsupported response type {type(mock_response).__name__}; "
                        "expected JSONResponse or StreamingResponse so response middleware can process it."
                    ),
                )

        # For real backends the upstream expects the served-model name in the body.
        else:
            json_body["model"] = resolved_served_model_name

            # Snapshot what was forwarded to the backend (body and headers both copied so
            # the restore of json_body["model"] below and any future header mutations cannot
            # reach ctx.proxied_request). Headers are taken from the post-middleware request
            # rather than the post-build NextRequestInfo to preserve the historical contract
            # plugins observe in process_response — auth/host stripping and provider extras
            # are an upstream-bound concern, not part of what middleware "saw forwarded."
            ctx.proxied_request = build_inference_request(
                body=dict(json_body),
                headers=dict(modified_request.headers),
                path=proxy_path,
            )

            next_request_info = await build_next_request(
                request,
                host_url=resolved_model_provider_info.model_provider.host_url,
                trailing_uri=proxy_path,
                auth_token=resolved_model_provider_info.secret_value,
                auth_header_format=resolved_model_provider_info.model_provider.auth_header_format,
                body=json_body,
                default_extra_body=resolved_model_provider_info.model_provider.default_extra_body,
                default_extra_headers=resolved_model_provider_info.model_provider.default_extra_headers,
                required_extra_body=resolved_model_provider_info.model_provider.required_extra_body,
                required_extra_headers=resolved_model_provider_info.model_provider.required_extra_headers,
                request_headers=modified_request.headers,
            )
            proxy_response_result, response_headers, response_status = await fetch_proxy_response(
                http_client,
                next_request_info,
                served_model_name=resolved_served_model_name,
                restored_model_id=f"{modified_model_ref.workspace}/{modified_model_ref.name}",
            )
            json_body["model"] = f"{modified_model_ref.workspace}/{modified_model_ref.name}"

        # Rewrite the served-model name back to the post-middleware model entity reference
        # in the response body so the user never sees the upstream's served_model_name. This
        # runs *before* response middleware so plugins observe the entity-keyed view their
        # request-side counterparts produced. For non-streaming results we mutate the dict
        # in place; for streams we wrap the iterator so each SSE chunk is rewritten as it
        # is yielded. The rewrite is a strict-equality swap, so it is safely a no-op for
        # the mock-provider branch above (mocks were never sent the served name) and for
        # any chunks that legitimately do not carry a `model` field (e.g. Anthropic
        # `content_block_*` / `message_delta` / `message_stop` / `ping` events).
        restored_model_id = f"{modified_model_ref.workspace}/{modified_model_ref.name}"
        if isinstance(proxy_response_result, dict):
            _rewrite_model_field(proxy_response_result, resolved_served_model_name, restored_model_id)
        else:
            proxy_response_result = _rewrite_model_field_in_stream(
                proxy_response_result, resolved_served_model_name, restored_model_id
            )

        inference_response = InferenceResponse(
            result=proxy_response_result,
            headers=dict(response_headers),
            response_body_annotations=dict(ctx.response_body_annotations),
        )

    # Response middleware chain.
    response_middleware_calls = registry.response_middleware_calls.get((workspace, vm_name), [])
    if response_middleware_calls:
        inference_response = build_inference_response(
            inference_response.result,
            inference_response.headers,
            ctx.backend_format,
            response_body_annotations=inference_response.response_body_annotations,
        )
        try:
            inference_response = await execute_response_middleware(
                response_middleware_calls,
                registry.plugins,
                ctx,
                inference_response,
            )
        except InferenceMiddlewareError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    # Build final streaming response.
    is_streaming = _is_streaming_response_result(inference_response)
    response_with_annotations = _build_inference_response_with_annotations(inference_response)
    final_response = await stream_response_result(
        response_with_annotations,
        response_status,
        response_with_annotations.headers,
    )

    # Post-response middleware (fire-and-forget, non-streaming only).
    post_response_middleware_calls = registry.post_response_middleware_calls.get((workspace, vm_name), [])
    if post_response_middleware_calls and not is_streaming:
        post_response = inference_response
        if not response_middleware_calls:
            post_response = build_inference_response(
                inference_response.result,
                inference_response.headers,
                ctx.backend_format,
                response_body_annotations=inference_response.response_body_annotations,
            )
        post_response_task = asyncio.create_task(
            execute_post_response_middleware(
                post_response_middleware_calls,
                registry.plugins,
                ctx,
                post_response,
            )
        )
        # Test harnesses can opt in to observing fire-and-forget tasks by
        # initialising ``app.state.pending_post_response_tasks = []`` at fixture
        # setup; production never sets the attribute, so the getattr keeps the
        # production hot path free of test-only state.
        pending = getattr(request.app.state, "pending_post_response_tasks", None)
        if pending is not None:
            pending.append(post_response_task)

    return final_response


async def retrieve_secret_value(workspace: str, secret_name: str, secrets_sdk: AsyncNeMoPlatform) -> str:
    """
    Retrieve a raw API key from the Platform Secrets service.

    Args:
        workspace: The workspace containing the secret
        secret_name: The name of the secret to retrieve
        secrets_sdk: The async NeMoPlatform SDK client configured for secrets service

    Returns:
        The raw secret string (e.g., "sk-ant-...")

    Raises:
        HTTPException: If the secret is not found or there's an error retrieving it
    """
    try:
        logger.debug(f"Retrieving API key from secrets service: {workspace}/{secret_name}")
        secrets = client_from_platform(secrets_sdk, AsyncSecretsClient)
        response = (await secrets.access_secret(name=secret_name, workspace=workspace)).data()
        api_key = response.value

        if not api_key:
            logger.error(f"API key secret found but data is empty: {workspace}/{secret_name}")
            raise HTTPException(status_code=500, detail=f"API key secret is empty: {workspace}/{secret_name}")

        logger.debug(f"Successfully retrieved API key from secret {workspace}/{secret_name}")
        return api_key
    except ClientNotFoundError as e:
        logger.error(f"API key secret not found: {workspace}/{secret_name}")
        raise HTTPException(status_code=500, detail=f"API key secret not found: {workspace}/{secret_name}") from e
    except HTTPException:
        # Re-raise HTTPException as-is
        raise
    except Exception as e:
        logger.exception(f"Error retrieving API key: {workspace}/{secret_name}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve API key: {e}") from e
