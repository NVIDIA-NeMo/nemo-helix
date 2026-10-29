# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from importlib.metadata import entry_points
from typing import Any

import pytest
from nemo_evals.gateway import tokens as tokens_module
from nemo_evals.gateway.middleware import evals_access as evals_access_module
from nemo_evals.gateway.middleware.closed_book import MIDDLEWARE_NAME as CLOSED_BOOK
from nemo_evals.gateway.middleware.evals_access import (
    ACCESS_MIDDLEWARE_NAME,
    EvalsAccessMiddleware,
    ModelPolicyEnforcer,
    Refusal,
    RefusalReason,
    policy_refusal,
)
from nemo_evals.gateway.middleware.trial_recording import MIDDLEWARE_NAME as RECORDING
from nemo_evals.gateway.policy import ModelAccess, ModelPolicy
from nemo_evals.gateway.spans import GATEWAY_CALL_ATTRIBUTE
from nemo_evals.gateway.tokens import (
    RouterKeyClaims,
    TrialClaims,
    TrialTokenReader,
    unsigned_token,
)
from nemo_helix_plugin.client.oidc import decode_jwt_claims, generate_unsigned_jwt
from nemo_helix_plugin.inference_middleware import (
    InferenceAccessRequest,
    InferenceMiddlewareError,
    InferenceMiddlewareUnavailableError,
)
from nemo_helix_plugin.inference_middleware_models import MiddlewareCall, VirtualModel
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span, Status

CB = "models/cb-strong"
CLOSED = ModelPolicy(model_access=ModelAccess.CLOSED, models=(CB,))
TRIAL = TrialClaims(session="run-1:hello", intake_workspace="evals", model_access=ModelAccess.CLOSED, models=(CB,))
TRIAL_TOKEN = unsigned_token(TRIAL)


def _call(name: str) -> MiddlewareCall:
    return MiddlewareCall(name=name, config_type="any", config={})


def _vm(
    name: str, *, request: Sequence[str] = (), response: Sequence[str] = (), default: str | None = "models/served"
) -> VirtualModel:
    workspace, _, short = name.partition("/")
    return VirtualModel(
        workspace=workspace,
        name=short,
        default_model_entity=default,
        request_middleware=[_call(entry) for entry in request],
        response_middleware=[_call(entry) for entry in response],
    )


VIRTUAL_MODELS = {
    CB: _vm(CB, request=[RECORDING, CLOSED_BOOK], response=[RECORDING]),
    "models/open-strong": _vm("models/open-strong", request=[RECORDING], response=[RECORDING]),
    "models/plain": _vm("models/plain"),
    "models/closed-only": _vm("models/closed-only", request=[CLOSED_BOOK]),
}


def _request(
    *, model: str | None = CB, provider: str | None = None, token: str | None = None, requested: str | None = None
) -> InferenceAccessRequest:
    return InferenceAccessRequest(
        request_id="req-1",
        workspace="models",
        path="v1/responses",
        headers={"authorization": f"Bearer {token}"} if token is not None else {},
        virtual_model_id=None if provider is not None else model,
        requested_model=requested,
        provider_name=provider,
    )


@pytest.mark.parametrize(
    ("policy", "request_", "virtual_model", "reason"),
    [
        pytest.param(CLOSED, _request(), VIRTUAL_MODELS[CB], None, id="its own closed-book model"),
        pytest.param(CLOSED, _request(provider="nvidia"), None, RefusalReason.PROVIDER_ROUTE, id="the provider route"),
        pytest.param(
            CLOSED,
            _request(model="models/open-strong"),
            VIRTUAL_MODELS["models/open-strong"],
            RefusalReason.UNLISTED_MODEL,
            id="a model it doesn't list",
        ),
        pytest.param(
            CLOSED,
            _request(),
            _vm(CB, request=[CLOSED_BOOK], default=None),
            RefusalReason.UNPINNED_MODEL,
            id="a listed model that lets the caller pick what it serves",
        ),
        pytest.param(
            CLOSED,
            _request(),
            _vm(CB, request=[CLOSED_BOOK, RECORDING]),
            RefusalReason.NOT_CLOSED_BOOK,
            id="a listed model that no longer ends closed-book",
        ),
        pytest.param(CLOSED, _request(), None, RefusalReason.UNPINNED_MODEL, id="a listed model that's gone"),
        pytest.param(
            CLOSED,
            _request(requested="models/cb-strong&adapters/tuning/strong-lora"),
            VIRTUAL_MODELS[CB],
            RefusalReason.UNLISTED_MODEL,
            id="a LoRA adapter served through a listed model",
        ),
        pytest.param(
            ModelPolicy(model_access=ModelAccess.OPEN, models=("models/open-strong",)),
            _request(model="models/open-strong"),
            VIRTUAL_MODELS["models/open-strong"],
            None,
            id="an open policy's own model",
        ),
        pytest.param(
            ModelPolicy(model_access=ModelAccess.OPEN, models=("models/open-strong",)),
            _request(),
            VIRTUAL_MODELS[CB],
            RefusalReason.UNLISTED_MODEL,
            id="an open policy switching models",
        ),
        pytest.param(
            ModelPolicy(model_access=ModelAccess.OPEN),
            _request(model="models/plain"),
            VIRTUAL_MODELS["models/plain"],
            None,
            id="an open policy that lists no models",
        ),
    ],
)
def test_a_policy_allows_only_its_listed_models_and_closed_book_ones_when_closed(
    policy: ModelPolicy,
    request_: InferenceAccessRequest,
    virtual_model: VirtualModel | None,
    reason: RefusalReason | None,
) -> None:
    refusal = policy_refusal(policy, request_, virtual_model)

    assert (refusal.reason if refusal is not None else None) == reason


class Signed:
    """Stands in for the platform's signature check: trusts the claims of exactly the tokens it was given."""

    def __init__(self, *tokens: str, fail: bool = False) -> None:
        self._tokens = set(tokens)
        self._fail = fail

    async def __call__(self, token: str) -> Mapping[str, Any] | None:
        if self._fail:
            raise OSError("signing keys unreachable")
        return decode_jwt_claims(token) if token in self._tokens else None


def _enforcer(
    *, auth_enabled: bool, signed: Signed | None = None
) -> tuple[ModelPolicyEnforcer, list[tuple[str, Refusal]]]:
    refused: list[tuple[str, Refusal]] = []
    reader = TrialTokenReader(signed if auth_enabled else None)
    enforcer = ModelPolicyEnforcer(
        reader,
        VIRTUAL_MODELS.get,
        recording_middleware=RECORDING,
        auth_enabled=auth_enabled,
        on_refused=lambda credential, _, refusal: refused.append((credential, refusal)),
    )
    return enforcer, refused


async def test_a_trial_that_switches_models_is_refused_and_the_refusal_is_reported_against_its_token() -> None:
    enforcer, refused = _enforcer(auth_enabled=True, signed=Signed(TRIAL_TOKEN))

    await enforcer.check(_request(token=TRIAL_TOKEN))
    with pytest.raises(InferenceMiddlewareError) as refusal:
        await enforcer.check(_request(model="models/open-strong", token=TRIAL_TOKEN))

    assert refusal.value.status_code == 403
    assert [(credential, entry.reason) for credential, entry in refused] == [
        (TRIAL_TOKEN, RefusalReason.UNLISTED_MODEL)
    ]


async def test_a_router_key_is_held_to_its_policy_like_a_trial() -> None:
    router_key = unsigned_token(RouterKeyClaims(model_access=ModelAccess.CLOSED, models=(CB,)))
    enforcer, _ = _enforcer(auth_enabled=True, signed=Signed(router_key))

    await enforcer.check(_request(token=router_key))
    with pytest.raises(InferenceMiddlewareError):
        await enforcer.check(_request(provider="nvidia", token=router_key))


async def test_an_evals_policy_the_platform_did_not_sign_is_refused() -> None:
    """With auth on, a token that writes its own open policy gets nowhere."""
    forged = unsigned_token(TrialClaims(session="run-1:hello", intake_workspace="evals", model_access=ModelAccess.OPEN))
    enforcer, refused = _enforcer(auth_enabled=True, signed=Signed())

    with pytest.raises(InferenceMiddlewareError) as refusal:
        await enforcer.check(_request(model="models/open-strong", token=forged))

    assert refusal.value.status_code == 403
    assert refused == []


async def test_a_policy_that_cannot_be_verified_refuses_the_call_as_unavailable() -> None:
    enforcer, _ = _enforcer(auth_enabled=True, signed=Signed(fail=True))

    with pytest.raises(InferenceMiddlewareUnavailableError):
        await enforcer.check(_request(token=TRIAL_TOKEN))


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("nvapi-provider-key", id="an opaque key"),
        pytest.param(generate_unsigned_jwt("user:alice"), id="a platform token without evals claims"),
    ],
)
async def test_callers_without_an_evals_credential_are_left_alone(token: str) -> None:
    enforcer, _ = _enforcer(auth_enabled=True, signed=Signed())

    await enforcer.check(_request(provider="nvidia", token=token))


@pytest.mark.parametrize(
    ("model", "refused"),
    [
        pytest.param(CB, True, id="a closed-book model that records trials"),
        pytest.param("models/open-strong", True, id="an open model that records trials"),
        pytest.param("models/closed-only", False, id="a closed-book model that only judges and model targets use"),
        pytest.param("models/plain", False, id="a model without evals middleware"),
    ],
)
@pytest.mark.parametrize("token", [None, "anything"], ids=["no credential", "a credential that isn't an evals one"])
async def test_with_auth_off_a_model_that_records_trials_needs_an_evals_credential(
    model: str, refused: bool, token: str | None
) -> None:
    """With auth off the gateway accepts any token, so dropping or replacing the trial token must not escape."""
    enforcer, _ = _enforcer(auth_enabled=False)

    if refused:
        with pytest.raises(InferenceMiddlewareError) as refusal:
            await enforcer.check(_request(model=model, token=token))
        assert refusal.value.status_code == 401
    else:
        await enforcer.check(_request(model=model, token=token))


async def test_with_auth_off_an_unsigned_trial_token_is_enforced() -> None:
    enforcer, refused = _enforcer(auth_enabled=False)

    await enforcer.check(_request(token=TRIAL_TOKEN))
    with pytest.raises(InferenceMiddlewareError):
        await enforcer.check(_request(provider="nvidia", token=TRIAL_TOKEN))

    assert [entry.reason for _, entry in refused] == [RefusalReason.PROVIDER_ROUTE]


def _value(value: Any) -> Any:
    return getattr(value, value.WhichOneof("value"))


def _attributes(span: Span) -> dict[str, Any]:
    return {attribute.key: _value(attribute.value) for attribute in span.attributes}


def _spans(batches: list[tuple[str, list[ResourceSpans]]]) -> list[tuple[str, Span]]:
    return [
        (workspace, span)
        for workspace, batch in batches
        for resource in batch
        for scope in resource.scope_spans
        for span in scope.spans
    ]


class Sink:
    def __init__(self) -> None:
        self.sent: list[tuple[str, list[ResourceSpans]]] = []

    async def __call__(self, workspace: str, spans: list[ResourceSpans]) -> int:
        self.sent.append((workspace, spans))
        return 0


def test_the_access_check_is_registered_for_the_gateway() -> None:
    entries = [e for e in entry_points(group="nemo.inference_middleware") if e.name == ACCESS_MIDDLEWARE_NAME]

    assert entries
    assert all(entry.load() is EvalsAccessMiddleware for entry in entries)


def test_no_virtual_model_can_list_the_access_check() -> None:
    """The gateway runs the access check on every call; listing it on a VirtualModel would suggest otherwise."""
    access = EvalsAccessMiddleware()

    assert not any(access.supports_middleware_phase(phase) for phase in ("request", "response", "post_response"))


async def test_a_call_refused_for_its_trials_policy_is_recorded_in_the_trial_as_a_free_failed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused call costs nothing; recording it with zero tokens keeps the trial's usage counted from the gateway."""
    sink = Sink()
    monkeypatch.setattr(tokens_module, "platform_auth_enabled", lambda: False)
    monkeypatch.setattr(evals_access_module, "platform_auth_enabled", lambda: False)
    monkeypatch.setattr(evals_access_module, "intake_sink", lambda client: sink)
    access = EvalsAccessMiddleware()
    monkeypatch.setattr(access, "_get_platform_client", lambda hook: object())
    request = InferenceAccessRequest(
        request_id="req-1",
        workspace="models",
        path="v1/responses",
        headers={"authorization": f"Bearer {TRIAL_TOKEN}"},
        provider_name="nvidia",
    )

    await access.on_startup()
    try:
        with pytest.raises(InferenceMiddlewareError):
            await access.check_access(request)
    finally:
        await access.on_shutdown()

    [(workspace, span)] = _spans(sink.sent)
    attributes = _attributes(span)
    assert (workspace, span.name, span.status.code) == ("evals", "responses", Status.STATUS_CODE_ERROR)
    assert attributes[GATEWAY_CALL_ATTRIBUTE] is True
    assert attributes["nemo.evals.refusal.reason"] == "provider_route"
    assert attributes["nemo.provider"] == "models/nvidia"
    assert (attributes["gen_ai.usage.input_tokens"], attributes["gen_ai.usage.output_tokens"]) == (0, 0)


async def test_access_is_refused_until_the_policy_check_has_started() -> None:
    request = InferenceAccessRequest(request_id="r", workspace="models", path="v1/responses", headers={})

    with pytest.raises(InferenceMiddlewareUnavailableError):
        await EvalsAccessMiddleware().check_access(request)
