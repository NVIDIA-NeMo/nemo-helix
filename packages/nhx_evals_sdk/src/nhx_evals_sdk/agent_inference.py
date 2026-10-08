# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP inference for generic and NeMo Agent Toolkit targets.

Public agent variants are normalized into one transport description and then
executed as either a blocking JSON request or a JSON SSE stream. The typed
``invoke_agent`` path preserves status and evidence, while
``make_agent_inference_request`` retains the legacy OpenAI-like dictionary
contract and failure behavior.
"""

# ruff: noqa: I001 - the vendored SDK mirror uses different import-order settings.

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable
from urllib.parse import urlparse

import httpx
from httpx import Timeout
from pydantic import BaseModel, ConfigDict, Field

from nhx_evals_sdk.agent_stream_translation import (
    SseFrame,
    AgentStreamTranslation,
    AgentStreamTranslationContext,
    AgentStreamTranslator,
)
from nhx_evals_sdk.inference import get_logger, requests_log_var
from nhx_evals_sdk.resilience.api import run_with_resilience
from nhx_evals_sdk.resilience.classifier import endpoint_identity
from nhx_evals_sdk.templates import render_template
from nhx_evals_sdk.values.agents import (
    Agent,
    GenericAgent,
    NatAgentConfig,
    NemoAgentToolkitAgent,
    StreamAggregation,
)
from nhx_evals_sdk.values.evidence import (
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
    CandidateEvidence,
    EvidenceDescriptor,
)

# Default timeout for agent requests (seconds).
_DEFAULT_TIMEOUT = 120.0


class AgentInvocationStatus(str, Enum):
    """Agent invocation outcome before it is adapted into an agent-eval trial."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class AgentInvocationResult(BaseModel):
    """Typed agent response with optional evidence and partial-run status."""

    model_config = ConfigDict(extra="forbid")

    status: AgentInvocationStatus
    response: dict[str, Any]
    output_text: str | None = None
    evidence: CandidateEvidence | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentInferenceContext(BaseModel):
    """Per-invocation persistence and identity supplied by an evaluator."""

    model_config = ConfigDict(extra="forbid")

    evidence_dir: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class _HttpAgentInvocation(BaseModel):
    """Resolved transport request shared by every HTTP agent variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str
    payload: dict[str, Any]
    query_params: dict[str, str] = Field(default_factory=dict)
    response_path: str
    trajectory_path: str | None = None
    stream: bool = False
    response_path_field: str = "response_path"
    response_aggregation: StreamAggregation = "last"


# SSE field names look like ``data``, ``intermediate_data``, ``observability_trace``;
# require the pre-colon token to match before treating a line as a frame, so a bare
# JSON line (e.g. ``{"value": 1}``) is not mis-split at an interior colon.
_SSE_CHANNEL_PATTERN = re.compile(r"^[A-Za-z_][\w-]*$")


class _StreamCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_count: int = 0
    raw_lines: list[str] = Field(default_factory=list)
    frames: list[SseFrame] = Field(default_factory=list)
    final_payload: Any | None = None
    # Raw extracted response value (any JSON type); preserved so the OpenAI-like
    # response keeps the original type instead of an unconditional ``str()`` cast.
    final_value: Any | None = None
    final_trajectory: Any | None = None
    output_text: str | None = None
    status_code: int | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class AgentInferenceFn(Protocol):
    """Callable protocol for agent inference function dependency injection."""

    def __call__(
        self,
        agent: Agent,
        request: dict,
        *,
        client: httpx.AsyncClient | None = None,
        max_retries: int | None,
        api_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Awaitable[dict | AgentInvocationResult]: ...


AgentInferenceFnFactory: TypeAlias = Callable[[AgentInferenceContext], AgentInferenceFn]


def make_agent_inference_fn(
    context: AgentInferenceContext,
    *,
    stream_translator: AgentStreamTranslator | None = None,
    capture_evidence: bool = False,
) -> AgentInferenceFn:
    """Bind evaluator-owned context and stream policy to ``invoke_agent``."""
    return partial(
        invoke_agent,
        evidence_dir=context.evidence_dir,
        invocation_context=dict(context.metadata),
        stream_translator=stream_translator,
        capture_evidence=capture_evidence,
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def new_agent_inference_client(timeout: float | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=Timeout(timeout or _DEFAULT_TIMEOUT))


async def make_agent_inference_request(
    agent: Agent,
    request: dict,
    *,
    client: httpx.AsyncClient | None = None,
    max_retries: int | None = 3,
    api_key: str | None = None,
    default_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict:
    """Run inference and return the legacy OpenAI-like response dictionary."""
    result = await invoke_agent(
        agent,
        request,
        client=client,
        max_retries=max_retries,
        api_key=api_key,
        default_headers=default_headers,
        timeout=timeout,
    )
    if result.status is not AgentInvocationStatus.COMPLETED:
        endpoint = result.metadata.get("endpoint", agent.url)
        raise RuntimeError(
            f"Agent at {endpoint} completed the SSE stream without producing a final value. "
            "Verify that the agent endpoint is functioning correctly."
        )
    return result.response


async def invoke_agent(
    agent: Agent,
    request: dict,
    *,
    client: httpx.AsyncClient | None = None,
    max_retries: int | None = 3,
    api_key: str | None = None,
    default_headers: dict[str, str] | None = None,
    timeout: float | None = None,
    evidence_dir: str | Path | None = None,
    stream_translator: AgentStreamTranslator | None = None,
    invocation_context: Mapping[str, Any] | None = None,
    capture_evidence: bool = False,
) -> AgentInvocationResult:
    """Invoke an agent and preserve structured status and evidence."""
    invocation = _resolve_http_agent_invocation(agent, request)
    return await _invoke_http_agent(
        agent,
        invocation,
        client=client,
        max_retries=max_retries,
        api_key=api_key,
        default_headers=default_headers,
        timeout=timeout,
        evidence_dir=evidence_dir,
        stream_translator=stream_translator,
        invocation_context=invocation_context,
        capture_evidence=capture_evidence,
    )


# ---------------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------------


async def _make_generic_agent_request(
    agent: GenericAgent,
    request: dict,
    *,
    client: httpx.AsyncClient | None = None,
    max_retries: int | None = 3,
    api_key: str | None = None,
    default_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict:
    return await make_agent_inference_request(
        agent,
        request,
        client=client,
        max_retries=max_retries,
        api_key=api_key,
        default_headers=default_headers,
        timeout=timeout,
    )


async def _make_nat_agent_request(
    agent: NemoAgentToolkitAgent,
    request: dict,
    *,
    client: httpx.AsyncClient | None = None,
    max_retries: int | None = 3,
    api_key: str | None = None,
    default_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict:
    return await make_agent_inference_request(
        agent,
        request,
        client=client,
        max_retries=max_retries,
        api_key=api_key,
        default_headers=default_headers,
        timeout=timeout,
    )


def _resolve_http_agent_invocation(agent: Agent, request: dict[str, Any]) -> _HttpAgentInvocation:
    """Normalize a public agent target into one HTTP transport request."""
    if isinstance(agent, GenericAgent):
        context: dict[str, Any] = {**request, "request": request}
        rendered_body = render_template(agent.body, context=context)
        payload = rendered_body if isinstance(rendered_body, dict) else {"args": rendered_body}
        return _HttpAgentInvocation(
            endpoint=agent.url,
            payload=payload,
            response_path=agent.response_path,
            trajectory_path=agent.trajectory_path,
            stream=agent.stream,
            response_aggregation=agent.response_aggregation,
        )

    config = agent.nat or NatAgentConfig()
    endpoint = _nat_endpoint(agent, config)
    payload = request if config.request_mode == "passthrough" else {"input_message": _derive_input_message(request)}
    return _HttpAgentInvocation(
        endpoint=endpoint,
        payload=payload,
        query_params=config.query_params,
        response_path=config.response_path,
        stream=True,
        response_path_field="nat.response_path",
        response_aggregation=config.response_aggregation,
    )


async def _invoke_http_agent(
    agent: Agent,
    invocation: _HttpAgentInvocation,
    *,
    client: httpx.AsyncClient | None = None,
    max_retries: int | None = 3,
    api_key: str | None = None,
    default_headers: dict[str, str] | None = None,
    timeout: float | None = None,
    evidence_dir: str | Path | None = None,
    stream_translator: AgentStreamTranslator | None = None,
    invocation_context: Mapping[str, Any] | None = None,
    capture_evidence: bool = False,
) -> AgentInvocationResult:
    log = get_logger()
    resolved_api_key = api_key or agent.api_key
    effective_timeout = timeout or _DEFAULT_TIMEOUT

    headers: dict[str, str] = {**(default_headers or {}), "Content-Type": "application/json"}
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    endpoint_key = endpoint_identity(invocation.endpoint, model_id=agent.name, auth_identity=resolved_api_key)
    max_attempts = max(1, (max_retries if max_retries is not None else 0) + 1)
    inference_client = client or new_agent_inference_client(timeout=effective_timeout)
    retain_stream_details = capture_evidence or stream_translator is not None

    if not invocation.stream:

        async def _invoke_post() -> dict[str, Any]:
            response = await inference_client.post(
                invocation.endpoint,
                json=invocation.payload,
                headers=headers,
                params=invocation.query_params,
                timeout=effective_timeout,
            )
            response.raise_for_status()
            return response.json()

        log.info("Making agent request to %s", invocation.endpoint)
        try:
            result_data = await run_with_resilience(endpoint_key, _invoke_post, max_attempts=max_attempts)
        except Exception:
            log.exception("Agent request to %s failed after %d attempts", invocation.endpoint, max_attempts)
            raise
        finally:
            if client is None:
                await inference_client.aclose()

        response_value = _extract_jsonpath(
            result_data,
            invocation.response_path,
            field_name=invocation.response_path_field,
        )
        response = _openai_response(str(response_value))
        # The synthesized response keeps only the extracted text, so carry the agent's own token
        # usage across: it is the sole token source for a row evaluation, whose trials are otherwise
        # built with no measurements at all. Model targets already return their full completion.
        usage = result_data.get("usage") if isinstance(result_data, Mapping) else None
        if isinstance(usage, Mapping):
            response["usage"] = dict(usage)
        if invocation.trajectory_path:
            trajectory = _extract_jsonpath(
                result_data,
                invocation.trajectory_path,
                field_name="trajectory_path",
                required=False,
            )
            if trajectory is not None:
                response["trajectory"] = trajectory
        requests_log_var.get([]).append({"request": invocation.payload, "response": result_data})
        log.info("Agent request to %s completed", invocation.endpoint)
        return AgentInvocationResult(
            status=AgentInvocationStatus.COMPLETED,
            response=response,
            output_text=_openai_response_text(response),
        )

    async def _invoke_stream() -> _StreamCapture:
        capture = _StreamCapture()
        # In "concat" mode each data frame carries a token-level delta, so the
        # final output is the ordered join of every matched value rather than
        # the last one. Accumulate parts here and materialize after the stream.
        aggregate = invocation.response_aggregation == "concat"
        value_parts: list[str] = []
        try:
            async with inference_client.stream(
                "POST",
                invocation.endpoint,
                json=invocation.payload,
                headers=headers,
                params=invocation.query_params,
                timeout=effective_timeout,
            ) as response:
                capture.status_code = response.status_code if isinstance(response.status_code, int) else None
                capture.response_headers = _string_headers(response.headers)
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    if retain_stream_details:
                        capture.raw_lines.append(raw_line)
                    frame = _parse_sse_frame(raw_line)
                    if frame is None:
                        continue
                    capture.event_count += 1
                    if retain_stream_details:
                        capture.frames.append(frame)
                    if frame.channel != "data" or frame.payload == "[DONE]":
                        continue
                    capture.final_payload = frame.payload
                    value = _extract_jsonpath(
                        frame.payload,
                        invocation.response_path,
                        field_name=invocation.response_path_field,
                        required=False,
                    )
                    if value is not None:
                        if aggregate:
                            value_parts.append(value if isinstance(value, str) else str(value))
                        else:
                            capture.final_value = value
                            # Preserve the original type in the response; expose
                            # ``output_text`` only when the value is already textual.
                            capture.output_text = value if isinstance(value, str) else None
                    if invocation.trajectory_path:
                        trajectory = _extract_jsonpath(
                            frame.payload,
                            invocation.trajectory_path,
                            field_name="trajectory_path",
                            required=False,
                        )
                        if trajectory is not None:
                            capture.final_trajectory = trajectory
                    capture.error = capture.error or _stream_error(frame.payload)
        except Exception as exc:
            if capture.event_count == 0:
                raise
            capture.error = f"{type(exc).__name__}: {exc}"
        if aggregate and value_parts:
            capture.final_value = "".join(value_parts)
            capture.output_text = capture.final_value
        return capture

    log.info("Making streaming agent request to %s", invocation.endpoint)
    try:
        capture = await run_with_resilience(endpoint_key, _invoke_stream, max_attempts=max_attempts)
    except Exception as exc:
        log.exception("Streaming agent request to %s failed after %d attempts", invocation.endpoint, max_attempts)
        # When evidence capture or a stream translator is enabled, surface an HTTP
        # failure that occurred before the first stream frame as a PARTIAL result
        # with http_metadata evidence instead of raising, so the trial stays
        # inspectable.
        # The legacy dict-returning path keeps capture disabled, so it still raises.
        http_error = _http_status_error(exc) if capture_evidence or stream_translator is not None else None
        if http_error is None:
            raise
        capture = _StreamCapture(
            status_code=http_error.response.status_code,
            response_headers=_string_headers(http_error.response.headers),
            error=f"HTTP {http_error.response.status_code}",
        )
    finally:
        if client is None:
            await inference_client.aclose()

    # COMPLETED only when a non-empty value was extracted and no terminal stream
    # error occurred. An extracted-but-empty value (e.g. "") stays PARTIAL.
    has_output = capture.final_value is not None and capture.final_value != ""
    status = AgentInvocationStatus.COMPLETED if has_output and capture.error is None else AgentInvocationStatus.PARTIAL
    response = _openai_response(capture.final_value)
    if capture.final_trajectory is not None:
        response["trajectory"] = capture.final_trajectory
    evidence = (
        _stream_evidence(capture, invocation.payload, headers)
        if capture_evidence or stream_translator is not None
        else None
    )
    translation_metadata: dict[str, Any] = {}
    if stream_translator is not None and capture.frames:
        values = dict(invocation_context or {})
        context = AgentStreamTranslationContext(
            agent_name=agent.name,
            endpoint=invocation.endpoint,
            request_payload=invocation.payload,
            final_payload=capture.final_payload,
            output_text=capture.output_text,
            run_id=_optional_string(values.get("run_id")),
            task_id=_optional_string(values.get("task_id")),
            invocation_id=_optional_string(values.get("invocation_id")),
            conversation_id=_optional_string(invocation.payload.get("conversation_id")),
            http_status=capture.status_code,
            stream_error=capture.error,
        )
        try:
            raw_translation = stream_translator(capture.frames, context=context)
            translation = AgentStreamTranslation.model_validate(
                raw_translation.model_dump(mode="python")
                if isinstance(raw_translation, AgentStreamTranslation)
                else raw_translation
            )
            schema_version = translation.trajectory.get("schema_version")
            if schema_version != "ATIF-v1.7":
                raise ValueError(
                    f"Agent stream translators must return a canonical ATIF-v1.7 trajectory, got {schema_version}"
                )
            reserved = {
                EVIDENCE_TRACE,
                EVIDENCE_RAW_STREAM,
                EVIDENCE_STREAM_EVENTS,
                EVIDENCE_REQUEST_PAYLOAD,
                EVIDENCE_REQUEST_HEADERS,
                EVIDENCE_HTTP_METADATA,
            }
            collisions = reserved.intersection(translation.evidence)
            if collisions:
                raise ValueError(f"translator evidence uses reserved names: {sorted(collisions)}")
            descriptors = dict(evidence.descriptors) if evidence is not None else {}
            descriptors[EVIDENCE_TRACE] = EvidenceDescriptor(
                kind=EVIDENCE_TRACE,
                format=EVIDENCE_FORMAT_ATIF,
                data=translation.trajectory,
            )
            descriptors.update(translation.evidence)
            evidence = CandidateEvidence(
                descriptors=descriptors,
                metadata=dict(evidence.metadata) if evidence is not None else {},
            )
            translation_metadata = translation.metadata
        except Exception as exc:
            status = AgentInvocationStatus.FAILED
            descriptors = dict(evidence.descriptors) if evidence is not None else {}
            descriptors[EVIDENCE_TRANSLATION_ERROR] = EvidenceDescriptor(
                kind="error",
                format=EVIDENCE_FORMAT_JSON,
                data={"error_type": type(exc).__name__, "error": str(exc)},
            )
            evidence = CandidateEvidence(descriptors=descriptors)
            translation_metadata = {
                EVIDENCE_TRANSLATION_ERROR: str(exc),
                f"{EVIDENCE_TRANSLATION_ERROR}_type": type(exc).__name__,
            }
    if evidence is not None and evidence_dir is not None:
        evidence = _persist_stream_evidence(evidence, Path(evidence_dir))

    # Record request/response for audit
    requests_log = requests_log_var.get([])
    requests_log.append({"request": invocation.payload, "response": capture.final_payload})

    log.info("Streaming agent request to %s completed", invocation.endpoint)
    return AgentInvocationResult(
        status=status,
        response=response,
        output_text=capture.output_text,
        evidence=evidence,
        metadata={
            "endpoint": invocation.endpoint,
            "event_count": capture.event_count,
            "final_payload": capture.final_payload,
            "http_status": capture.status_code,
            "stream_error": capture.error,
            **translation_metadata,
        },
    )


def _nat_endpoint(agent: NemoAgentToolkitAgent, config: NatAgentConfig) -> str:
    if urlparse(config.endpoint).scheme:
        return config.endpoint
    return f"{agent.url.rstrip('/')}/{config.endpoint.lstrip('/')}"


def _parse_sse_frame(raw_line: str) -> SseFrame | None:
    line = raw_line.strip()
    if not line or line.startswith("event:") or ":" not in line:
        return None
    channel, payload_text = line.split(":", 1)
    channel = channel.strip()
    # Only treat the line as a frame when the pre-colon token is a valid SSE
    # field name; otherwise it is a bare payload line (e.g. raw JSON) and is skipped.
    if not _SSE_CHANNEL_PATTERN.match(channel):
        return None
    payload_text = payload_text.strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        payload = payload_text
    return SseFrame(channel=channel, payload=payload, raw=raw_line)


def _http_status_error(exc: BaseException) -> httpx.HTTPStatusError | None:
    """Return the first ``HTTPStatusError`` in the exception's ``__cause__`` chain.

    A non-retryable HTTP error re-raises directly, while a retryable one that
    exhausts attempts is wrapped by the resilience scheduler with the original
    error chained via ``from exc``; walk the chain to find either.
    """
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, httpx.HTTPStatusError):
            return current
        current = current.__cause__
    return None


def _stream_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if error is None and isinstance(payload.get("value"), dict):
        error = payload["value"].get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if code and message:
            return f"{code}: {message}"
        return str(message or code or error)
    if error is not None:
        return str(error)
    return None


def _openai_response(content: Any) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _openai_response_text(response: dict[str, Any]) -> str | None:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


def _string_headers(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    sensitive = {"authorization", "cookie", "proxy-authorization", "set-cookie", "x-api-key"}
    return {key: "<redacted>" if key.lower() in sensitive else value for key, value in headers.items()}


def _stream_evidence(
    capture: _StreamCapture,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> CandidateEvidence:
    raw_stream = "\n".join(capture.raw_lines) + ("\n" if capture.raw_lines else "")
    values: dict[str, tuple[str, str, Any]] = {
        EVIDENCE_RAW_STREAM: ("agent_stream", EVIDENCE_FORMAT_TEXT, raw_stream),
        EVIDENCE_STREAM_EVENTS: (
            "agent_stream_events",
            EVIDENCE_FORMAT_JSON,
            [frame.model_dump(mode="json") for frame in capture.frames],
        ),
        EVIDENCE_REQUEST_PAYLOAD: (EVIDENCE_REQUEST_PAYLOAD, EVIDENCE_FORMAT_JSON, payload),
        EVIDENCE_REQUEST_HEADERS: (EVIDENCE_REQUEST_HEADERS, EVIDENCE_FORMAT_JSON, _redact_headers(headers)),
        EVIDENCE_HTTP_METADATA: (
            EVIDENCE_HTTP_METADATA,
            EVIDENCE_FORMAT_JSON,
            {
                "status_code": capture.status_code,
                "headers": _redact_headers(capture.response_headers),
                "error": capture.error,
            },
        ),
    }
    descriptors: dict[str, EvidenceDescriptor] = {}
    for name, (kind, format_name, data) in values.items():
        descriptors[name] = EvidenceDescriptor(kind=kind, format=format_name, data=data)
    return CandidateEvidence(descriptors=descriptors)


def _evidence_filename(
    name: str,
    descriptor: EvidenceDescriptor,
    *,
    reserved_filenames: set[str],
    used_filenames: set[str],
) -> str:
    suffix = "txt" if descriptor.format in {EVIDENCE_FORMAT_TEXT, "txt"} else "json"
    canonical_trace = name == EVIDENCE_TRACE and descriptor.format == EVIDENCE_FORMAT_ATIF
    if canonical_trace:
        filename = "atif_trace.json"
    else:
        stem = "".join(char if char.isalnum() or char in "-_." else "-" for char in name)
        stem = stem.strip("-_.")[:96] or "evidence"
        filename = f"{stem}.{suffix}"

    filename_key = filename.casefold()
    if not canonical_trace and (filename_key in reserved_filenames or filename_key in used_filenames):
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        stem = filename.rsplit(".", maxsplit=1)[0][:79]
        filename = f"{stem}-{digest}.{suffix}"
        filename_key = filename.casefold()

    if filename_key in used_filenames:
        raise ValueError(f"evidence descriptors map to the same filename: {filename!r}")
    used_filenames.add(filename_key)
    return filename


def _persist_stream_evidence(evidence: CandidateEvidence, root: Path) -> CandidateEvidence:
    """Replace one SDK-owned invocation directory with file-backed evidence."""
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    canonical_trace = evidence.descriptors.get(EVIDENCE_TRACE)
    reserved_filenames = (
        {"atif_trace.json"}
        if canonical_trace is not None
        and canonical_trace.data is not None
        and canonical_trace.format == EVIDENCE_FORMAT_ATIF
        else set()
    )
    used_filenames: set[str] = set()
    persisted: dict[str, EvidenceDescriptor] = {}
    for name, descriptor in evidence.descriptors.items():
        if descriptor.data is None:
            persisted[name] = descriptor
            continue
        filename = _evidence_filename(
            name,
            descriptor,
            reserved_filenames=reserved_filenames,
            used_filenames=used_filenames,
        )
        path = root / filename
        if descriptor.format in {EVIDENCE_FORMAT_TEXT, "txt"}:
            path.write_text(str(descriptor.data), encoding="utf-8")
        else:
            path.write_text(
                json.dumps(descriptor.data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        persisted[name] = descriptor.model_copy(update={"ref": str(path.resolve()), "data": None})
    return evidence.model_copy(update={"descriptors": persisted})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_input_message(request: dict) -> str:
    """Derive a single input_message string from an inference request.

    Handles both chat-style (``messages``) and completion-style (``prompt``)
    requests.
    """
    if "messages" in request:
        messages = request["messages"]
        # Use the last user message content, or concatenate all messages
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg["content"])
        # Fallback: concatenate all message contents
        return "\n".join(str(msg.get("content", "")) for msg in messages)

    if "prompt" in request:
        return str(request["prompt"])

    raise ValueError("Agent inference request must contain 'messages' or 'prompt'.")


def _extract_jsonpath(
    data: dict[str, Any],
    path: str,
    *,
    field_name: str = "path",
    required: bool = True,
) -> Any:
    """Extract a value from data using a JSONPath expression."""
    # Imported here rather than at module scope: this is jsonpath_ng's only use in the module, and
    # the module sits on the agent_eval run-time path via agent_eval/evaluator.py.
    from jsonpath_ng import parse as jsonpath_parse

    expr = jsonpath_parse(path)
    matches = expr.find(data)
    if not matches:
        if required:
            raise ValueError(f"JSONPath '{path}' ({field_name}) did not match any value in agent response: {data}")
        return None
    return matches[-1].value
