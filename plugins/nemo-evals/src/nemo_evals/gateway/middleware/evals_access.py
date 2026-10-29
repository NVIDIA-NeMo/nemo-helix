# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nemo-evals-access``: the Inference Gateway access check, refusing each call its caller's evals credential doesn't allow.

A trial token or router key lists the VirtualModels it may call, and a closed policy requires each to be closed-book
(see :mod:`nemo_evals.gateway.policy`). The gateway calls :meth:`ModelPolicyEnforcer.check` for every call on every
route before any VirtualModel's middleware runs, so a caller can't reach a model by a route its policy doesn't name.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import NoReturn

from nemo_evals.gateway.middleware.closed_book import MIDDLEWARE_NAME as CLOSED_BOOK_MIDDLEWARE
from nemo_evals.gateway.middleware.closed_book import ends_closed_book
from nemo_evals.gateway.middleware.trial_recording import MIDDLEWARE_NAME as RECORDING_MIDDLEWARE
from nemo_evals.gateway.policy import ModelAccess, ModelPolicy
from nemo_evals.gateway.recorder import CallRecorder, FinishedCall, intake_sink
from nemo_evals.gateway.spans import CallerSpan
from nemo_evals.gateway.tokens import TrialTokenReader, claims_evals_credential, platform_token_reader
from nemo_evals.gateway.usage import CallApi, TokenUsage
from nemo_helix_plugin.auth import inference_gateway_credential, platform_auth_enabled
from nemo_helix_plugin.inference_middleware import (
    InferenceAccessRequest,
    InferenceMiddlewareError,
    InferenceMiddlewareUnavailableError,
    NemoInferenceMiddleware,
    VirtualModel,
)

logger = logging.getLogger(__name__)

ACCESS_MIDDLEWARE_NAME = "nemo-evals-access"

LORA_ADAPTER_SEPARATOR = "&adapters/"


class RefusalReason(StrEnum):
    PROVIDER_ROUTE = "provider_route"
    UNLISTED_MODEL = "unlisted_model"
    UNPINNED_MODEL = "unpinned_model"
    NOT_CLOSED_BOOK = "not_closed_book"
    UNVERIFIED_CREDENTIAL = "unverified_credential"
    MISSING_CREDENTIAL = "missing_credential"


@dataclass(frozen=True)
class Refusal:
    reason: RefusalReason
    detail: str
    status_code: int = 403


def policy_refusal(
    policy: ModelPolicy, request: InferenceAccessRequest, virtual_model: VirtualModel | None
) -> Refusal | None:
    """Why ``policy`` refuses ``request``, served by ``virtual_model``; ``None`` when it allows it."""
    allowed = ", ".join(policy.models) if policy.models else "any VirtualModel"
    if request.provider_name is not None:
        return Refusal(
            RefusalReason.PROVIDER_ROUTE,
            f"This credential may call only {allowed}, not the provider route to "
            f"{request.workspace}/{request.provider_name}, which skips every VirtualModel's middleware.",
        )
    model = request.virtual_model_id
    if policy.models is not None and model not in policy.models:
        return Refusal(RefusalReason.UNLISTED_MODEL, f"This credential may call only {allowed}, not {model}.")
    if policy.models is not None and LORA_ADAPTER_SEPARATOR in (request.requested_model or ""):
        return Refusal(
            RefusalReason.UNLISTED_MODEL,
            f"This credential may call only {allowed}, not the LoRA adapter {request.requested_model}.",
        )
    if policy.models is not None and (virtual_model is None or not virtual_model.default_model_entity):
        return Refusal(
            RefusalReason.UNPINNED_MODEL,
            f"{model} sets no default_model_entity, so its caller could name any model through it.",
        )
    if policy.model_access is ModelAccess.CLOSED and (
        virtual_model is None or not ends_closed_book(virtual_model.request_middleware)
    ):
        return Refusal(
            RefusalReason.NOT_CLOSED_BOOK,
            f"This credential may call only closed-book models, and {model} doesn't end its request_middleware "
            f"with {CLOSED_BOOK_MIDDLEWARE}.",
        )
    return None


RefusalSink = Callable[[str, InferenceAccessRequest, Refusal], None]
"""Told of each call refused for a credential's policy: the credential, the call, and why."""


class ModelPolicyEnforcer:
    """Refuses each gateway call its caller's evals credential doesn't allow.

    A call without an evals credential is left alone, except that with platform auth off a call to a VirtualModel
    that records trials must carry one: an agent can't escape its policy there by dropping its token. Closed-book
    VirtualModels without recording also serve model evaluations and judges, which carry no evals credential.
    With auth off nothing is signed, so a caller that writes its own token, or leaves it off on any other route, isn't
    stopped.
    """

    def __init__(
        self,
        reader: TrialTokenReader,
        virtual_model: Callable[[str], VirtualModel | None],
        *,
        recording_middleware: str,
        auth_enabled: bool,
        on_refused: RefusalSink | None = None,
    ) -> None:
        self._reader = reader
        self._virtual_model = virtual_model
        self._recording_middleware = recording_middleware
        self._auth_enabled = auth_enabled
        self._on_refused = on_refused

    async def check(self, request: InferenceAccessRequest) -> None:
        """Return when ``request`` may proceed.

        Raises:
            InferenceMiddlewareError: The caller's credential doesn't allow the call.
            InferenceMiddlewareUnavailableError: The credential couldn't be verified.
        """
        credential = inference_gateway_credential(request.headers)
        model = request.virtual_model_id
        if credential is None or not claims_evals_credential(credential):
            if not self._auth_enabled and model is not None:
                virtual_model = self._virtual_model(model)
                if virtual_model is not None and self._records_trials(virtual_model):
                    self._refuse(
                        None,
                        request,
                        Refusal(
                            RefusalReason.MISSING_CREDENTIAL,
                            f"{model} records evaluation trials, so calls to it must carry the credential the "
                            "evaluation issued.",
                            status_code=401,
                        ),
                    )
            return
        try:
            claims = await self._reader.claims(credential)
        except Exception as exc:
            logger.warning("Couldn't verify an evals credential", exc_info=True)
            raise InferenceMiddlewareUnavailableError("The caller's evals credential couldn't be verified.") from exc
        if claims is None:
            self._refuse(
                None,
                request,
                Refusal(
                    RefusalReason.UNVERIFIED_CREDENTIAL,
                    "The caller's credential carries an evals model policy the platform didn't issue.",
                ),
            )
        refusal = policy_refusal(claims, request, self._virtual_model(model) if model is not None else None)
        if refusal is not None:
            self._refuse(credential, request, refusal)

    def _records_trials(self, virtual_model: VirtualModel) -> bool:
        return any(
            call.name == self._recording_middleware
            for call in [*virtual_model.request_middleware, *virtual_model.response_middleware]
        )

    def _refuse(self, credential: str | None, request: InferenceAccessRequest, refusal: Refusal) -> NoReturn:
        logger.warning(
            "Refused a gateway call for its caller's model policy",
            extra={
                "request_id": request.request_id,
                "workspace": request.workspace,
                "virtual_model": request.virtual_model_id,
                "provider": request.provider_name,
                "reason": refusal.reason.value,
            },
        )
        if credential is not None and self._on_refused is not None:
            self._on_refused(credential, request, refusal)
        raise InferenceMiddlewareError(refusal.detail, status_code=refusal.status_code)


def refused_call(credential: str, request: InferenceAccessRequest, refusal: Refusal) -> FinishedCall:
    """A call the gateway refused for its caller's model policy, to record in the caller's trial."""
    now = datetime.now(UTC)
    target = (
        {"nemo.virtual_model": request.virtual_model_id}
        if request.virtual_model_id is not None
        else {"nemo.provider": f"{request.workspace}/{request.provider_name}"}
    )
    api = CallApi.from_path(request.path)
    return FinishedCall(
        token=credential,
        name=api.value,
        started_at=now,
        ended_at=now,
        cancelled=False,
        attributes={
            "gen_ai.operation.name": "chat",
            "nemo.inference.api": api.value,
            **target,
            "error.type": "model_policy",
            "nemo.evals.refusal.reason": refusal.reason.value,
            "nemo.evals.refusal.detail": refusal.detail,
            **TokenUsage(uncached_input_tokens=0, output_tokens=0).attributes(),
        },
        caller=CallerSpan.from_headers(request.headers.get("traceparent"), request.headers.get("tracestate")),
        error=True,
    )


class EvalsAccessMiddleware(NemoInferenceMiddleware):
    """Refuse each gateway call the caller's evals credential doesn't allow, and record the refusal in its trial.

    No VirtualModel lists it: the gateway runs :meth:`check_access` on every call, on every route.
    """

    def __init__(self) -> None:
        super().__init__()
        self._recorder: CallRecorder | None = None
        self._enforcer: ModelPolicyEnforcer | None = None

    def supports_middleware_phase(self, phase: str) -> bool:
        return False

    async def on_startup(self) -> None:
        reader = platform_token_reader()
        recorder = CallRecorder(reader, intake_sink(self._get_platform_client("on_startup")))
        self._recorder = recorder
        self._enforcer = ModelPolicyEnforcer(
            reader,
            self.get_virtual_model,
            recording_middleware=RECORDING_MIDDLEWARE,
            auth_enabled=platform_auth_enabled(),
            on_refused=lambda credential, request, refusal: recorder.record(refused_call(credential, request, refusal)),
        )
        recorder.start()

    async def on_shutdown(self) -> None:
        if self._recorder is not None:
            await self._recorder.stop()

    async def check_access(self, request: InferenceAccessRequest) -> None:
        if self._enforcer is None:
            raise InferenceMiddlewareUnavailableError("The evals model policy check hasn't started.")
        await self._enforcer.check(request)
