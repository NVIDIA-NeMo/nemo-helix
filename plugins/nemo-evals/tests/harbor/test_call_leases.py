# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task Jobs lease each sandbox's address to its trial, and end the lease before the kill."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from harbor.models.trial.paths import TrialPaths
from nemo_evals.harbor import call_leases
from nemo_evals.harbor.call_leases import CallLeaseRegistrar, TrialIdentity, current_registrar, leasing
from nemo_evals.harbor.opensandbox_environment import AgentChannelOpenSandboxEnvironment
from nemo_evals.intake.mapping import session_id_for
from nemo_helix_plugin.client.client import AsyncNemoClient
from nhx_sandbox.harbor_opensandbox_environment import NemoOpenSandboxEnvironment

STARTED = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
TRIAL = "hello-world__AbC123"


def test_a_trials_identity_is_the_session_publishing_uses_and_is_stable() -> None:
    identity = TrialIdentity.of("run-1-task-0", TRIAL)

    assert identity.session_id == session_id_for("run-1-task-0", TRIAL)
    assert identity == TrialIdentity.of("run-1-task-0", TRIAL)
    assert len(identity.trace_id) == 32 and len(identity.root_span_id) == 16
    assert int(identity.trace_id, 16) and int(identity.root_span_id, 16)
    assert TrialIdentity.of("run-1-task-0", "other__x").trace_id != identity.trace_id


def _registrar(monkeypatch: pytest.MonkeyPatch, answers: list[httpx.Response], seen: list[httpx.Request]) -> Any:
    real_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return answers.pop(0)

    def client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(call_leases.httpx, "AsyncClient", client)
    platform = AsyncNemoClient(base_url="http://platform.test", workspace="evals")
    return CallLeaseRegistrar(platform, workspace="evals", evaluation_name="run-1")


def _lease_json(**overrides: Any) -> dict[str, Any]:
    identity = TrialIdentity.of("run-1-task-0", TRIAL)
    return {
        "name": "lease-sbx-1",
        "workspace": "evals",
        "client_address": "10.244.0.7",
        "session_id": identity.session_id,
        "trace_id": identity.trace_id,
        "root_span_id": identity.root_span_id,
        "role": "agent",
        "started_at": STARTED.isoformat(),
        **overrides,
    }


async def test_the_registrar_leases_through_the_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []
    registrar = _registrar(monkeypatch, [httpx.Response(201, json=_lease_json())], seen)

    leased = await registrar.lease(
        name="lease-sbx-1",
        client_address="10.244.0.7",
        identity=TrialIdentity.of("run-1-task-0", TRIAL),
        role="agent",
        started_at=STARTED,
    )

    assert leased
    [request] = seen
    assert request.method == "POST"
    assert request.url.path == "/apis/inference-gateway/v2/workspaces/evals/call-leases"
    body = json.loads(request.content)
    assert body["session_id"] == session_id_for("run-1-task-0", TRIAL)
    assert body["client_address"] == "10.244.0.7" and body["role"] == "agent"
    assert body["evaluation_name"] == "run-1"


async def test_a_lease_the_gateway_refuses_does_not_fail_the_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    registrar = _registrar(monkeypatch, [httpx.Response(403, json={"detail": "Forbidden"})], [])

    leased = await registrar.lease(
        name="lease-sbx-1",
        client_address="10.244.0.7",
        identity=TrialIdentity.of("run-1-task-0", TRIAL),
        role="agent",
        started_at=STARTED,
    )

    assert not leased


async def test_the_registrar_ends_a_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []
    ended = STARTED + timedelta(minutes=5)
    registrar = _registrar(monkeypatch, [httpx.Response(200, json=_lease_json(ended_at=ended.isoformat()))], seen)

    await registrar.end(name="lease-sbx-1", ended_at=ended)

    [request] = seen
    assert request.method == "PATCH" and request.url.path.endswith("/call-leases/lease-sbx-1")
    assert datetime.fromisoformat(json.loads(request.content)["ended_at"]) == ended


def test_leasing_installs_the_registrar_only_inside_the_block() -> None:
    registrar = object.__new__(CallLeaseRegistrar)

    with leasing(registrar):
        assert current_registrar() is registrar
    assert current_registrar() is None


class _Registrar(CallLeaseRegistrar):
    """Records lease calls, and every event in order with the environment's."""

    def __init__(self, events: list[str], *, accept: bool = True) -> None:
        self.events = events
        self.accept = accept
        self.leases: list[dict[str, Any]] = []

    async def lease(self, **kwargs: Any) -> bool:
        self.leases.append(kwargs)
        self.events.append("lease")
        return self.accept

    async def end(self, *, name: str, ended_at: datetime) -> None:
        self.events.append(f"end {name}")


def _environment(
    monkeypatch: pytest.MonkeyPatch, events: list[str], *, session: str = TRIAL, ip: str | None = "10.244.0.7"
) -> Any:
    environment = object.__new__(AgentChannelOpenSandboxEnvironment)
    environment._call_lease_run_id = "run-1-task-0"
    environment._lease_name = None
    environment._sandbox = None
    environment._record = None
    environment.session_id = session
    environment.trial_paths = TrialPaths(trial_dir=Path("/jobs") / TRIAL)

    async def create(self: Any) -> Any:
        events.append("create")
        self._record = {"sandbox_id": "SBX-1", "ip": ip, "started_at": STARTED.isoformat()}
        return "sandbox"

    async def stop(self: Any, delete: bool) -> None:
        events.append("kill")

    monkeypatch.setattr(NemoOpenSandboxEnvironment, "_create_sandbox", create)
    monkeypatch.setattr(NemoOpenSandboxEnvironment, "stop", stop)
    return environment


async def test_a_sandbox_is_leased_after_create_and_released_before_the_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    registrar = _Registrar(events)
    environment = _environment(monkeypatch, events)

    with leasing(registrar):
        sandbox = await environment._create_sandbox()
        await environment.stop(delete=True)

    assert sandbox == "sandbox"
    assert events == ["create", "lease", "end lease-sbx-1", "kill"]
    [lease] = registrar.leases
    assert lease["client_address"] == "10.244.0.7" and lease["role"] == "agent"
    assert lease["identity"] == TrialIdentity.of("run-1-task-0", TRIAL)
    assert lease["started_at"] == STARTED


async def test_a_separate_verifier_is_leased_to_the_same_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    registrar = _Registrar([])
    environment = _environment(monkeypatch, [], session=f"{TRIAL}__verifier__task")

    with leasing(registrar):
        await environment._create_sandbox()

    [lease] = registrar.leases
    assert lease["role"] == "verifier"
    assert lease["identity"] == TrialIdentity.of("run-1-task-0", TRIAL)


async def test_the_sandbox_is_owned_before_the_lease_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancellation during the lease call must still let stop() find and kill the sandbox."""
    owned: list[Any] = []
    environment = _environment(monkeypatch, [])

    class _Watching(_Registrar):
        async def lease(self, **kwargs: Any) -> bool:
            owned.append(environment._sandbox)
            return True

    with leasing(_Watching([])):
        await environment._create_sandbox()

    assert owned == ["sandbox"]


@pytest.mark.parametrize("reason", ["no registrar", "no ip", "refused"])
async def test_nothing_is_released_when_nothing_was_leased(monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    events: list[str] = []
    environment = _environment(monkeypatch, events, ip=None if reason == "no ip" else "10.244.0.7")
    registrar = _Registrar(events, accept=reason != "refused")

    if reason == "no registrar":
        await environment._create_sandbox()
        await environment.stop(delete=True)
    else:
        with leasing(registrar):
            await environment._create_sandbox()
            await environment.stop(delete=True)

    assert not any(event.startswith("end") for event in events)
    assert events[-1] == "kill"
