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
from nhx.core.inference_gateway.config import InferenceGatewayConfig
from nhx.core.inference_gateway.service import InferenceGatewayService
from nhx.testing import create_test_client

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
