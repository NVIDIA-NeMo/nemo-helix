# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.tenancy import caller_is_admin


@pytest.fixture(autouse=True)
def _restore_overrides():
    yield
    v1.dependency_overrides.pop(current_principal, None)
    v1.dependency_overrides[caller_is_admin] = lambda: True


def _act_as(owner_id: str, *, admin: bool) -> None:
    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id=owner_id, source="platform"
    )
    v1.dependency_overrides[caller_is_admin] = lambda: admin


def _create_body() -> dict:
    return {
        "bundle_name": "my-custom-agent",
        "agent_name": "custom-agent",
        "agent_version": "1.2.3",
        "image_ref": "artifactory.example/team/agent:1.2.3",
        "image_digest": "artifactory.example/team/agent@sha256:" + "a" * 64,
        "entrypoint": "bin/custom-agent",
        "source_lock_digest": "sha256:" + "b" * 64,
        "fingerprint": "sha256:" + "c" * 64,
    }


def test_bundle_lifecycle_is_owner_scoped_until_promoted() -> None:
    _act_as("alice", admin=False)
    created = client.post("/v1/agent-bundles", json=_create_body())
    assert created.status_code == 201, created.text
    bundle = created.json()
    assert (bundle["owner_id"], bundle["visibility"], bundle["qualification_status"]) == (
        "alice",
        "private",
        "registered",
    )
    duplicate = client.post("/v1/agent-bundles", json=_create_body())
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["error"]["code"] == "agent_bundle_exists"

    _act_as("bob", admin=False)
    assert client.get(f"/v1/agent-bundles/{bundle['id']}").status_code == 404
    assert client.get("/v1/agent-bundles").json() == {"data": [], "next_cursor": None}
    assert client.delete(f"/v1/agent-bundles/{bundle['id']}").status_code == 404

    _act_as("admin", admin=True)
    not_qualified = client.post(f"/v1/agent-bundles/{bundle['id']}/promote")
    assert not_qualified.status_code == 409
    assert not_qualified.json()["detail"]["error"]["code"] == "agent_bundle_not_qualified"
    qualified = client.post(
        f"/v1/agent-bundles/{bundle['id']}/qualification",
        json={"status": "qualified", "evidence": {"rhacs": "admitted"}},
    )
    assert qualified.status_code == 200, qualified.text
    assert qualified.json()["qualified_by"] == "admin"
    assert qualified.json()["qualified_at"] is not None
    promoted = client.post(f"/v1/agent-bundles/{bundle['id']}/promote")
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["visibility"] == "public"

    _act_as("bob", admin=False)
    assert [item["id"] for item in client.get("/v1/agent-bundles").json()["data"]] == [bundle["id"]]
    assert client.get("/v1/agent-bundles", params={"mine": True}).json()["data"] == []
    # Public bundles can only be deleted by an admin, even by their owner.
    _act_as("alice", admin=False)
    assert client.delete(f"/v1/agent-bundles/{bundle['id']}").status_code == 404

    _act_as("admin", admin=True)
    rejected = client.post(f"/v1/agent-bundles/{bundle['id']}/qualification", json={"status": "rejected"})
    assert rejected.json()["visibility"] == "private"
    _act_as("alice", admin=False)
    assert client.delete(f"/v1/agent-bundles/{bundle['id']}").json() == {"id": bundle["id"], "deleted": True}
    assert client.get(f"/v1/agent-bundles/{bundle['id']}").status_code == 404
    assert client.post("/v1/agent-bundles", json=_create_body()).status_code == 201


def test_bundle_list_pages_newest_first() -> None:
    ids = []
    for index in range(3):
        body = {**_create_body(), "bundle_name": f"agent-{index}"}
        ids.append(client.post("/v1/agent-bundles", json=body).json()["id"])
    first = client.get("/v1/agent-bundles", params={"limit": 2}).json()
    assert [item["id"] for item in first["data"]] == [ids[2], ids[1]]
    second = client.get("/v1/agent-bundles", params={"limit": 2, "cursor": first["next_cursor"]}).json()
    assert [item["id"] for item in second["data"]] == [ids[0]]
    assert second["next_cursor"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"image_digest": "artifactory.example/team/agent:latest"},
        {"image_ref": "artifactory.example/other/agent:1.2.3"},
        {"entrypoint": "../bin/agent"},
    ],
)
def test_create_rejects_invalid_images_and_entrypoints(overrides: dict) -> None:
    response = client.post("/v1/agent-bundles", json={**_create_body(), **overrides})
    assert response.status_code == 422
