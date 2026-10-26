# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call lease endpoints against a real in-process entity store."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from nemo_helix_plugin.call_leases.client import AsyncCallLeasesClient
from nemo_helix_plugin.call_leases.types import CreateCallLeaseRequest, EndCallLeaseRequest
from nhx.core.inference_gateway.api import lease_cache
from nhx.core.inference_gateway.api.call_recording import CallRecorder, global_call_recorder, set_global_call_recorder
from nhx.core.inference_gateway.api.lease_cache import global_lease_cache
from nhx.core.inference_gateway.config import InferenceGatewayConfig
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.testing import create_test_client
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

BASE = "/apis/inference-gateway/v2/workspaces/default/call-leases"
STARTED = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with create_test_client(
        InferenceGatewayService,
        client_type=TestClient,
        service_configs={InferenceGatewayService: InferenceGatewayConfig(refresh_model_cache_interval_sec=0)},
    ) as tc:
        yield tc


def _lease(name: str = "lease-sbx-1", **overrides: Any) -> dict[str, Any]:
    return {
        "name": name,
        "client_address": "10.244.0.7",
        "session_id": "run-1-task-0:hello__abc",
        "trace_id": "0af7651916cd43dd8448eb211c80319c",
        "root_span_id": "b7ad6b7169203331",
        "role": "agent",
        "started_at": STARTED.isoformat(),
        **overrides,
    }


def test_a_lease_is_created_and_read_back(client: TestClient) -> None:
    created = client.post(BASE, json=_lease())

    assert created.status_code == 201, created.text
    read = client.get(f"{BASE}/lease-sbx-1").json()
    assert read["client_address"] == "10.244.0.7"
    assert read["ended_at"] is None


def test_creating_the_same_lease_again_returns_it(client: TestClient) -> None:
    """A task Job that retries a create after a lost response must not fail its trial."""
    first = client.post(BASE, json=_lease()).json()

    again = client.post(BASE, json=_lease())

    assert again.status_code == 200
    assert again.json()["id"] == first["id"]


def test_a_different_lease_under_an_existing_name_is_rejected(client: TestClient) -> None:
    client.post(BASE, json=_lease())

    response = client.post(BASE, json=_lease(client_address="10.244.0.8"))

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("client_address", "sandbox-1.svc"),
        ("trace_id", "0AF7651916CD43DD8448EB211C80319C"),
        ("trace_id", "0" * 32),
        ("root_span_id", "b7ad6b71"),
        ("started_at", "2026-10-26T12:00:00"),
        ("role", "judge"),
    ],
)
def test_malformed_leases_are_rejected(client: TestClient, field: str, value: str) -> None:
    assert client.post(BASE, json=_lease(**{field: value})).status_code == 422


def test_ending_a_lease_is_idempotent_but_cannot_move(client: TestClient) -> None:
    client.post(BASE, json=_lease())
    ended_at = (STARTED + timedelta(minutes=5)).isoformat()

    first = client.patch(f"{BASE}/lease-sbx-1", json={"ended_at": ended_at})
    again = client.patch(f"{BASE}/lease-sbx-1", json={"ended_at": ended_at})
    moved = client.patch(f"{BASE}/lease-sbx-1", json={"ended_at": (STARTED + timedelta(minutes=6)).isoformat()})

    assert first.status_code == 200
    assert datetime.fromisoformat(first.json()["ended_at"]) == STARTED + timedelta(minutes=5)
    assert again.status_code == 200
    assert moved.status_code == 409


def test_a_lease_cannot_end_before_it_started(client: TestClient) -> None:
    client.post(BASE, json=_lease())

    response = client.patch(f"{BASE}/lease-sbx-1", json={"ended_at": (STARTED - timedelta(seconds=1)).isoformat()})

    assert response.status_code == 422


def test_ending_an_unknown_lease_is_not_found(client: TestClient) -> None:
    response = client.patch(f"{BASE}/missing", json={"ended_at": STARTED.isoformat()})

    assert response.status_code == 404


def test_leases_are_listed_by_address(client: TestClient) -> None:
    client.post(BASE, json=_lease())
    client.post(BASE, json=_lease("lease-sbx-2", client_address="10.244.0.8", role="verifier"))

    listed = client.get(BASE, params={"filter[client_address]": "10.244.0.8"}).json()["data"]

    assert [lease["name"] for lease in listed] == ["lease-sbx-2"]


async def test_the_typed_client_creates_and_ends_a_lease(client: TestClient) -> None:
    transport = httpx.ASGITransport(app=client.app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        leases = AsyncCallLeasesClient(base_url="http://testserver", workspace="default", http_client=http_client)
        request = CreateCallLeaseRequest.model_validate(_lease())

        created = (await leases.create_call_lease(body=request)).data()
        ended = (
            await leases.end_call_lease(name=created.name, body=EndCallLeaseRequest(ended_at=STARTED + timedelta(1)))
        ).data()

    assert created.holds(STARTED + timedelta(hours=1))
    assert not ended.holds(STARTED + timedelta(days=2))


# --- leases drive the gateway's recording -----------------------------------------------------


class _Collecting(CallRecorder):
    def __init__(self) -> None:
        super().__init__(self._never)
        self.spans: list[tuple[str, ResourceSpans]] = []

    async def _never(self, workspace: str, spans: list[ResourceSpans]) -> None:
        raise AssertionError("never sends")

    def record(self, workspace: str, span: ResourceSpans) -> None:
        self.spans.append((workspace, span))


@pytest.fixture
def recorded(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Collecting]:
    monkeypatch.setattr(lease_cache, "_LEASE_CACHE", lease_cache.CallLeaseCache())
    previous = global_call_recorder()
    recorder = _Collecting()
    set_global_call_recorder(recorder)
    yield recorder
    set_global_call_recorder(previous)


def _spans(recorded: _Collecting) -> list[Span]:
    return [span for _, batch in recorded.spans for scope in batch.scope_spans for span in scope.spans]


def test_a_new_lease_holds_its_address_at_once(client: TestClient, recorded: _Collecting) -> None:
    """The first calls after a sandbox starts must not wait for the next cache refresh."""
    client.post(BASE, json=_lease())

    lease = global_lease_cache().lookup("10.244.0.7", datetime.now(UTC))

    assert lease is not None and lease.name == "lease-sbx-1"


def test_ending_the_trials_leases_records_one_root_spanning_them(client: TestClient, recorded: _Collecting) -> None:
    client.post(BASE, json=_lease())
    client.post(
        BASE,
        json=_lease(
            "lease-sbx-2",
            client_address="10.244.0.8",
            role="verifier",
            started_at=(STARTED + timedelta(minutes=6)).isoformat(),
        ),
    )

    client.patch(f"{BASE}/lease-sbx-1", json={"ended_at": (STARTED + timedelta(minutes=5)).isoformat()})
    client.patch(f"{BASE}/lease-sbx-2", json={"ended_at": (STARTED + timedelta(minutes=9)).isoformat()})

    first, second = _spans(recorded)
    assert first.span_id.hex() == second.span_id.hex() == "b7ad6b7169203331"
    assert first.start_time_unix_nano == second.start_time_unix_nano == int(STARTED.timestamp() * 10**9)
    assert second.end_time_unix_nano == int((STARTED + timedelta(minutes=9)).timestamp() * 10**9)
    assert global_lease_cache().lookup("10.244.0.7", STARTED + timedelta(minutes=7)) is None


OTLP = "/apis/inference-gateway/v2/workspaces/default/otlp/v1/traces"
PROTOBUF = {"content-type": "application/x-protobuf"}


def _export() -> bytes:
    span = Span(trace_id=b"\xaa" * 16, span_id=b"\x01" * 8, name="claude_code.interaction")
    return ExportTraceServiceRequest(
        resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=[span])])]
    ).SerializeToString()


def _from(client: TestClient, address: str) -> TestClient:
    return TestClient(client.app, client=(address, 50000))


def test_a_leased_agent_exports_its_trace_into_its_trial(client: TestClient, recorded: _Collecting) -> None:
    client.post(BASE, json=_lease(started_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat()))

    response = _from(client, "10.244.0.7").post(OTLP, content=_export(), headers=PROTOBUF)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-protobuf"
    ExportTraceServiceResponse.FromString(response.content)
    [(workspace, _)] = recorded.spans
    [span] = _spans(recorded)
    assert workspace == "default"
    assert span.trace_id.hex() == "0af7651916cd43dd8448eb211c80319c"
    assert span.parent_span_id.hex() == "b7ad6b7169203331"


@pytest.mark.parametrize(
    ("address", "path"),
    [("10.244.0.8", OTLP), ("10.244.0.7", OTLP.replace("/default/", "/other/"))],
    ids=["unleased-address", "other-workspace"],
)
def test_traces_from_outside_a_lease_are_refused(
    client: TestClient, recorded: _Collecting, address: str, path: str
) -> None:
    client.post(BASE, json=_lease(started_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat()))

    response = _from(client, address).post(path, content=_export(), headers=PROTOBUF)

    assert response.status_code == 403
    assert recorded.spans == []


@pytest.mark.parametrize(
    ("body", "headers", "status_code"),
    [
        (b"{}", {"content-type": "application/json"}, 415),
        (b"\xff\xff not protobuf", PROTOBUF, 400),
    ],
    ids=["json", "garbage"],
)
def test_malformed_exports_are_refused(
    client: TestClient, recorded: _Collecting, body: bytes, headers: dict[str, str], status_code: int
) -> None:
    client.post(BASE, json=_lease(started_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat()))

    response = _from(client, "10.244.0.7").post(OTLP, content=body, headers=headers)

    assert response.status_code == status_code


async def test_a_refresh_reads_leases_another_replica_created(client: TestClient) -> None:
    client.post(BASE, json=_lease(started_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat()))
    other_replica = lease_cache.CallLeaseCache()
    transport = httpx.ASGITransport(app=client.app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        platform = AsyncCallLeasesClient(base_url="http://testserver", workspace="default", http_client=http_client)

        await other_replica.refresh(platform)

    lease = other_replica.lookup("10.244.0.7", datetime.now(UTC))
    assert lease is not None and lease.workspace == "default"
