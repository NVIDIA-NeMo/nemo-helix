# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for /apis/agents/v2/workspaces/{ws}/agents/{name}/network-policy."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nemo_agents_plugin.api.v2 import network_policy as module
from nemo_agents_plugin.entities import Agent
from nemo_helix_plugin.entity_client import NemoEntityNotFoundError

pytest.importorskip("openshell")  # platform-restricted extra; skip where not installed (e.g. CI)

from openshell._proto import openshell_pb2 as pb
from openshell._proto import sandbox_pb2 as sb

BASE = "/apis/agents/v2/workspaces/default/agents/researcher/network-policy"
PYTHON = "/usr/local/bin/python3.13"
EXAMPLE_RULE = {
    "name": "allow_example_com_443",
    "endpoints": [{"host": "example.com", "port": 443}],
    "binaries": [{"path": PYTHON}],
}


def _chunk(
    chunk_id: str, status: str = "pending", *, host: str = "example.com", binary: str = PYTHON
) -> pb.PolicyChunk:
    name = f"allow_{host.replace('.', '_')}_443"
    return pb.PolicyChunk(
        id=chunk_id,
        status=status,
        rule_name=name,
        binary=binary,
        hit_count=3,
        rationale=f"Allow python3.13 to connect to {host}:443 (HTTPS).",
        review_token=f"token-{chunk_id}",
        proposed_rule=sb.NetworkPolicyRule(
            name=name,
            endpoints=[sb.NetworkEndpoint(host=host, port=443, ports=[443], advisor_proposed=True)],
            binaries=[sb.NetworkBinary(path=binary)],
        ),
    )


def _sandbox(name: str, labels: dict[str, str]) -> pb.Sandbox:
    sandbox = pb.Sandbox()
    sandbox.metadata.name = name
    sandbox.metadata.labels.update(labels)
    sandbox.status.phase = pb.SANDBOX_PHASE_READY
    return sandbox


AGENT_LABELS = {"nemo.agents/workspace": "default", "nemo.agents/agent": "researcher"}


class FakeGateway:
    """The deployments backend's view of the gateway, holding sandboxes and their chunks."""

    def __init__(self) -> None:
        self.sandboxes = [
            _sandbox("nhx-job", {**AGENT_LABELS, "nhx.nvidia.com/job_id": "researcher-run-1"}),
            _sandbox("nhx-dep", {**AGENT_LABELS, "nemo.agents/deployment": "researcher-live"}),
            _sandbox("nhx-other", {"nemo.agents/workspace": "default", "nemo.agents/agent": "someone-else"}),
        ]
        self.chunks: dict[str, list[pb.PolicyChunk]] = {
            "nhx-job": [_chunk("c1")],
            "nhx-dep": [],
            "nhx-other": [_chunk("c9")],
        }
        self.approved_with: list[str] = []
        self.rejections: list[tuple[str, str]] = []

    async def list_sandboxes(self) -> list[pb.Sandbox]:
        return self.sandboxes

    async def get_draft_chunks(self, sandbox_name: str) -> list[pb.PolicyChunk]:
        return self.chunks[sandbox_name]

    async def approve_draft_chunk(self, sandbox_name: str, chunk: pb.PolicyChunk) -> None:
        self.approved_with.append(chunk.review_token)
        self._decide(sandbox_name, chunk.id, "approved")

    async def reject_draft_chunk(self, sandbox_name: str, chunk_id: str, reason: str) -> None:
        self.rejections.append((chunk_id, reason))
        self._decide(sandbox_name, chunk_id, "rejected")

    def _decide(self, sandbox_name: str, chunk_id: str, status: str) -> None:
        for chunk in self.chunks[sandbox_name]:
            if chunk.id == chunk_id:
                chunk.status = status


class FakeEntities:
    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.updates = 0

    async def get(self, _entity: type, *, name: str, workspace: str) -> Agent:
        if (name, workspace) != (self.agent.name, self.agent.workspace):
            raise NemoEntityNotFoundError(f"{workspace}/{name}")
        return self.agent

    async def update(self, agent: Agent) -> Agent:
        self.updates += 1
        self.agent = agent
        return agent


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> FakeGateway:
    fake = FakeGateway()

    async def _gateway() -> FakeGateway:
        return fake

    monkeypatch.setattr(module, "_gateway", _gateway)
    return fake


@pytest.fixture
def entities() -> FakeEntities:
    return FakeEntities(Agent(name="researcher", workspace="default"))


@pytest.fixture
def client(gateway: FakeGateway, entities: FakeEntities) -> Iterator[TestClient]:  # noqa: ARG001 — wires the gateway
    app = FastAPI()
    app.include_router(module.router, prefix="/apis/agents/v2/workspaces/{workspace}")
    app.dependency_overrides[module.get_entity_client] = lambda: entities
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _requests(body: dict[str, Any], sandbox: str) -> list[dict[str, Any]]:
    return next(entry["requests"] for entry in body["sandboxes"] if entry["sandbox"] == sandbox)


def test_lists_only_the_agents_own_sandboxes_with_their_requests(client: TestClient) -> None:
    body = client.get(BASE).json()

    assert [(s["sandbox"], s["kind"], s["name"], s["phase"]) for s in body["sandboxes"]] == [
        ("nhx-job", "job", "researcher-run-1", "ready"),
        ("nhx-dep", "deployment", "researcher-live", "ready"),
    ]
    [request] = _requests(body, "nhx-job")
    assert request["status"] == "pending"
    assert request["endpoints"] == ["example.com:443"]
    assert request["binary"] == PYTHON
    assert request["rule"] == EXAMPLE_RULE
    assert request["remembered"] is False
    assert body["rules"] == {}
    assert body["agent_yaml"] == ""


def test_unknown_agent_is_not_found(client: TestClient) -> None:
    assert client.get(BASE.replace("researcher", "nobody")).status_code == 404


def test_remembered_rules_are_shown_when_the_gateway_cannot_be_reached(
    client: TestClient, entities: FakeEntities, monkeypatch: pytest.MonkeyPatch
) -> None:
    entities.agent.network_policies = {"allow_example_com_443": EXAMPLE_RULE}

    async def _no_gateway() -> None:
        raise ValueError("This platform has no OpenShell executor for agents.")

    monkeypatch.setattr(module, "_gateway", _no_gateway)

    body = client.get(BASE).json()

    assert body["rules"] == {"allow_example_com_443": EXAMPLE_RULE}
    assert body["sandboxes"] == []
    assert "no OpenShell executor" in body["gateway_error"]


def test_one_unreadable_sandbox_does_not_hide_the_others(client: TestClient, gateway: FakeGateway) -> None:
    del gateway.chunks["nhx-dep"]

    body = client.get(BASE).json()

    assert _requests(body, "nhx-job")
    assert next(s["error"] for s in body["sandboxes"] if s["sandbox"] == "nhx-dep")


def test_approve_lets_the_sandbox_through_without_changing_the_agent(
    client: TestClient, gateway: FakeGateway, entities: FakeEntities
) -> None:
    body = client.post(f"{BASE}/sandboxes/nhx-job/requests/c1/approve", json={"remember": False}).json()

    assert gateway.approved_with == ["token-c1"]
    assert _requests(body, "nhx-job")[0]["status"] == "approved"
    assert body["rules"] == {}
    assert entities.updates == 0


def test_approve_and_remember_saves_the_rule_on_the_agent(
    client: TestClient, gateway: FakeGateway, entities: FakeEntities
) -> None:
    body = client.post(f"{BASE}/sandboxes/nhx-job/requests/c1/approve", json={"remember": True}).json()

    assert gateway.approved_with == ["token-c1"]
    assert entities.agent.network_policies == {"allow_example_com_443": EXAMPLE_RULE}
    assert _requests(body, "nhx-job")[0]["remembered"] is True
    assert yaml.safe_load(body["agent_yaml"]) == {
        "helix_registration": {"network_policies": {"allow_example_com_443": EXAMPLE_RULE}}
    }


def test_an_already_approved_request_can_still_be_remembered(
    client: TestClient, gateway: FakeGateway, entities: FakeEntities
) -> None:
    gateway.chunks["nhx-job"] = [_chunk("c1", "approved")]

    response = client.post(f"{BASE}/sandboxes/nhx-job/requests/c1/approve", json={"remember": True})

    assert response.status_code == 200
    assert gateway.approved_with == []
    assert entities.agent.network_policies == {"allow_example_com_443": EXAMPLE_RULE}


def test_same_destination_from_another_executable_is_remembered_beside_the_first(
    client: TestClient, gateway: FakeGateway, entities: FakeEntities
) -> None:
    # A job image and a deployment image run different interpreters, and the gateway names
    # both requests after the destination.
    entities.agent.network_policies = {"allow_example_com_443": EXAMPLE_RULE}
    gateway.chunks["nhx-dep"] = [_chunk("c2", binary="/opt/uv/python/cpython-3.13/bin/python3.13")]

    body = client.post(f"{BASE}/sandboxes/nhx-dep/requests/c2/approve", json={"remember": True}).json()

    assert sorted(body["rules"]) == ["allow_example_com_443", "allow_example_com_443_2"]
    assert body["rules"]["allow_example_com_443"] == EXAMPLE_RULE
    assert body["rules"]["allow_example_com_443_2"]["binaries"] == [
        {"path": "/opt/uv/python/cpython-3.13/bin/python3.13"}
    ]
    assert body["rules"]["allow_example_com_443_2"]["name"] == "allow_example_com_443_2"


def test_remembering_the_same_rule_twice_keeps_one(
    client: TestClient, gateway: FakeGateway, entities: FakeEntities
) -> None:
    entities.agent.network_policies = {"allow_example_com_443": EXAMPLE_RULE}

    body = client.post(f"{BASE}/sandboxes/nhx-job/requests/c1/approve", json={"remember": True}).json()

    assert body["rules"] == {"allow_example_com_443": EXAMPLE_RULE}


def test_reject_keeps_the_sandbox_blocked_and_can_forget_the_rule(
    client: TestClient, gateway: FakeGateway, entities: FakeEntities
) -> None:
    entities.agent.network_policies = {"allow_example_com_443": EXAMPLE_RULE}

    body = client.post(
        f"{BASE}/sandboxes/nhx-job/requests/c1/reject", json={"reason": "Not needed.", "forget": True}
    ).json()

    assert gateway.rejections == [("c1", "Not needed.")]
    assert _requests(body, "nhx-job")[0]["status"] == "rejected"
    assert body["rules"] == {}


def test_reject_leaves_the_agents_rules_alone_by_default(client: TestClient, entities: FakeEntities) -> None:
    entities.agent.network_policies = {"allow_example_com_443": EXAMPLE_RULE}

    body = client.post(f"{BASE}/sandboxes/nhx-job/requests/c1/reject", json={}).json()

    assert body["rules"] == {"allow_example_com_443": EXAMPLE_RULE}


def test_a_decided_request_cannot_be_rejected(client: TestClient, gateway: FakeGateway) -> None:
    gateway.chunks["nhx-job"] = [_chunk("c1", "approved")]

    assert client.post(f"{BASE}/sandboxes/nhx-job/requests/c1/reject", json={}).status_code == 409
    assert gateway.rejections == []


def test_another_agents_sandbox_cannot_be_decided(client: TestClient, gateway: FakeGateway) -> None:
    response = client.post(f"{BASE}/sandboxes/nhx-other/requests/c9/approve", json={"remember": True})

    assert response.status_code == 404
    assert gateway.approved_with == []


def test_unknown_request_is_not_found(client: TestClient) -> None:
    assert client.post(f"{BASE}/sandboxes/nhx-job/requests/nope/approve", json={}).status_code == 404


def test_forget_drops_a_remembered_rule(client: TestClient, entities: FakeEntities) -> None:
    entities.agent.network_policies = {"allow_example_com_443": EXAMPLE_RULE}

    body = client.delete(f"{BASE}/rules/allow_example_com_443").json()

    assert body["rules"] == {}
    assert entities.agent.network_policies == {}
    assert client.delete(f"{BASE}/rules/allow_example_com_443").status_code == 404
