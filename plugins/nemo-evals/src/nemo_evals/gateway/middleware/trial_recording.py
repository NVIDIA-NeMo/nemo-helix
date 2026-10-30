# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo-evals-trial-recording``: records each model call an evaluation trial makes to Intake, in that trial.

An evaluation job issues each trial a trial token whose claims name the trial (see :mod:`nemo_evals.gateway.tokens`).
For every call through a VirtualModel that lists this middleware in ``response_middleware``, the middleware notes
the caller's token, the token usage the provider reported, and when the caller had the whole response, and hands the
call to a :class:`~nemo_evals.gateway.recorder.CallRecorder`. With platform auth on, it trusts only tokens the
platform signed, whatever the gateway accepted; with auth off, only unsigned tokens from ``nemo-evals``. A call made
with any other credential is not recorded.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any

from nemo_evals.gateway.recorder import CallRecorder, FinishedCall, intake_sink
from nemo_evals.gateway.spans import CallerSpan, trial_trace_id
from nemo_evals.gateway.tokens import TrialClaims, claims_a_trial, platform_token_reader
from nemo_evals.gateway.usage import CallApi, TokenUsage, model_from_payload, usage_from_payload
from nemo_helix_plugin.auth import inference_gateway_credential
from nemo_helix_plugin.client.oidc import decode_jwt_claims
from nemo_helix_plugin.inference_middleware import (
    InferenceMiddlewareContext,
    InferenceRequest,
    InferenceResponse,
    NemoInferenceMiddleware,
)
from pydantic import BaseModel, ConfigDict

MIDDLEWARE_NAME = "nemo-evals-trial-recording"
CONFIG_TYPE = "trial_recording"
STARTED_AT_STATE_KEY = "started_at"
ROUTING_SPAN_STATE_KEY = "routing_span"

AGENT_HEADERS = ("x-claude-code-session-id", "session_id", "session-id", "x-codex-turn-metadata", "x-codex-window-id")
AGENT_HEADER_ATTRIBUTE_PREFIX = "nemo.agent.header."
MAX_HEADER_VALUE_LENGTH = 1024


class TrialRecordingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ObservedCall:
    """One call through the middleware: what it was, and the usage its response reported."""

    def __init__(self, ctx: InferenceMiddlewareContext, token: str) -> None:
        request = ctx.original_request
        self.ctx = ctx
        self.token = token
        self.api = CallApi.from_path(request.path)
        self.usage = TokenUsage()
        self.response_model: str | None = None
        self.agent_headers = {
            name: value[:MAX_HEADER_VALUE_LENGTH] for name in AGENT_HEADERS if (value := request.headers.get(name))
        }
        self.caller = CallerSpan.from_headers(request.headers.get("traceparent"), request.headers.get("tracestate"))

    def observe(self, payload: object) -> None:
        if not isinstance(payload, Mapping):
            return
        if (usage := usage_from_payload(self.api, payload)) is not None:
            self.usage.merge(usage)
        self.response_model = model_from_payload(payload) or self.response_model

    def finished(self, started_at: datetime, *, cancelled: bool, routing_span: str | None = None) -> FinishedCall:
        proxied = self.ctx.proxied_request
        model = (proxied.body if proxied is not None else self.ctx.original_request.body).get("model")
        attributes: dict[str, object] = {
            "gen_ai.operation.name": "chat",
            "nemo.virtual_model": f"{self.ctx.workspace}/{self.ctx.virtual_model_name}",
            "nemo.inference.api": self.api.value,
            **{f"{AGENT_HEADER_ATTRIBUTE_PREFIX}{name}": value for name, value in self.agent_headers.items()},
            **self.usage.attributes(),
        }
        if isinstance(model, str):
            attributes["gen_ai.request.model"] = model
        if self.response_model is not None:
            attributes["gen_ai.response.model"] = self.response_model
        return FinishedCall(
            token=self.token,
            name=self.api.value,
            started_at=started_at,
            ended_at=datetime.now(UTC),
            cancelled=cancelled,
            attributes=attributes,
            caller=self.caller,
            span_id=routing_span,
            routing=routing_span is not None,
        )


class TrialRecordingMiddleware(NemoInferenceMiddleware):
    """Record each call made with a trial token to Intake, in the trial.

    It records from ``response_middleware``. Listed in ``request_middleware`` too, a call's span starts when the
    gateway received the call rather than when the response began.
    """

    def __init__(self) -> None:
        super().__init__()
        self._recorder: CallRecorder | None = None

    def supports_middleware_phase(self, phase: str) -> bool:
        return phase in {"request", "response"}

    async def validate_middleware_config(self, config_type: str, config: Any) -> TrialRecordingConfig:
        if config_type != CONFIG_TYPE:
            raise ValueError(f"{MIDDLEWARE_NAME} supports config_type {CONFIG_TYPE!r}, not {config_type!r}")
        return TrialRecordingConfig.model_validate(config or {})

    async def on_startup(self) -> None:
        self._recorder = CallRecorder(platform_token_reader(), intake_sink(self._get_platform_client("on_startup")))
        self._recorder.start()

    async def on_shutdown(self) -> None:
        if self._recorder is not None:
            await self._recorder.stop()

    async def process_request(
        self,
        ctx: InferenceMiddlewareContext,
        request: InferenceRequest,
        middleware_config: TrialRecordingConfig,
    ) -> InferenceRequest:
        ctx.state(MIDDLEWARE_NAME).set(STARTED_AT_STATE_KEY, datetime.now(UTC))
        if (routing_span := self._routing_span(request)) is not None:
            trace_id, span_id = routing_span
            request.headers["traceparent"] = f"00-{trace_id}-{span_id}-01"
            ctx.state(MIDDLEWARE_NAME).set(ROUTING_SPAN_STATE_KEY, span_id)
        return request

    def _routing_span(self, request: InferenceRequest) -> tuple[str, str] | None:
        """The trace and a fresh span id for a trial's call to a provider that calls models back as the trial."""
        credential = inference_gateway_credential(request.headers)
        model = request.body.get("model")
        if credential is None or not isinstance(model, str):
            return None
        trial = TrialClaims.from_token_claims(decode_jwt_claims(credential))
        if trial is None:
            return None
        try:
            if not self.get_inference_url_and_model(model).forwards_caller_credential:
                return None
        except KeyError:
            return None
        caller = CallerSpan.from_headers(request.headers.get("traceparent"), request.headers.get("tracestate"))
        return (caller.trace_id if caller is not None else trial_trace_id(trial)), os.urandom(8).hex()

    async def process_response(
        self,
        ctx: InferenceMiddlewareContext,
        response: InferenceResponse,
        middleware_config: TrialRecordingConfig,
    ) -> InferenceResponse:
        recorder = self._recorder
        credential = inference_gateway_credential(ctx.original_request.headers)
        if recorder is None or credential is None or not claims_a_trial(credential):
            return response
        call = _ObservedCall(ctx, credential)
        started_at = ctx.state(MIDDLEWARE_NAME).get(STARTED_AT_STATE_KEY) or datetime.now(UTC)
        routing_span = ctx.state(MIDDLEWARE_NAME).get(ROUTING_SPAN_STATE_KEY)
        if isinstance(response.result, Mapping):
            call.observe(response.result)
            recorder.record(call.finished(started_at, cancelled=False, routing_span=routing_span))
            return response
        response.result = _follow(response.result, call, recorder, started_at, routing_span)
        return response


async def _follow(
    chunks: AsyncIterator[dict[str, Any]],
    call: _ObservedCall,
    recorder: CallRecorder,
    started_at: datetime,
    routing_span: str | None,
) -> AsyncIterator[dict[str, Any]]:
    """Pass the stream through, recording the call when the caller has all of it or stops reading."""
    cancelled = True
    try:
        async for chunk in chunks:
            call.observe(chunk)
            yield chunk
        cancelled = False
    finally:
        recorder.record(call.finished(started_at, cancelled=cancelled, routing_span=routing_span))
