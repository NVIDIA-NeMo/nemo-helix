# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import get_conn
from scaled_evals.api.settings import settings


def _row(**overrides) -> dict:
    row = {
        "id": "ab_test",
        "owner_id": "dev",
        "bundle_name": "my-custom-agent",
        "agent_name": "custom-agent",
        "agent_version": "1.2.3",
        "image_ref": "artifactory.example/team/agent:1.2.3",
        "image_digest": "artifactory.example/team/agent@sha256:" + "a" * 64,
        "entrypoint": "bin/custom-agent",
        "platform": "linux/amd64",
        "runtime_abi": "glibc",
        "bundle_layout_version": 1,
        "builder_profile": "node22-npm-v1",
        "source_lock_digest": "sha256:" + "b" * 64,
        "fingerprint": "sha256:" + "c" * 64,
        "metadata": {},
        "visibility": "private",
        "qualification_status": "registered",
        "qualification_evidence": {},
        "qualified_at": None,
        "qualified_by": None,
        "created_at": "2026-07-08T00:00:00Z",
        "updated_at": "2026-07-08T00:00:00Z",
    }
    row.update(overrides)
    return row


def _connection(*, fetchone=None, fetchall=None) -> MagicMock:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = fetchone
    cur.fetchall.return_value = [] if fetchall is None else fetchall
    return conn


def _override(conn: MagicMock) -> None:
    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen


@pytest.fixture(autouse=True)
def _clean_override():
    _override(_connection())
    yield
    v1.dependency_overrides.pop(get_conn, None)
    v1.dependency_overrides.pop(current_principal, None)


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


def test_list_returns_accessible_envelope() -> None:
    response = client.get("/v1/agent-bundles")
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}


def test_create_registers_private_owned_bundle() -> None:
    conn = _connection(fetchone=_row())
    _override(conn)
    response = client.post("/v1/agent-bundles", json=_create_body())
    assert response.status_code == 201, response.text
    assert response.json()["owner_id"] == "dev"
    assert response.json()["visibility"] == "private"
    assert response.json()["qualification_status"] == "registered"


def test_create_rejects_mutable_image_tag() -> None:
    body = _create_body()
    body["image_digest"] = "artifactory.example/team/agent:latest"
    response = client.post("/v1/agent-bundles", json=body)
    assert response.status_code == 422


def test_create_rejects_mismatched_runtime_tag_repository() -> None:
    body = _create_body()
    body["image_ref"] = "artifactory.example/other/agent:1.2.3"
    response = client.post("/v1/agent-bundles", json=body)
    assert response.status_code == 422


def test_create_rejects_parent_entrypoint() -> None:
    body = _create_body()
    body["entrypoint"] = "../bin/agent"
    response = client.post("/v1/agent-bundles", json=body)
    assert response.status_code == 422


def test_get_inaccessible_bundle_is_hidden() -> None:
    response = client.get("/v1/agent-bundles/ab_other")
    assert response.status_code == 404


def test_non_admin_cannot_qualify_or_promote() -> None:
    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id="hosted-user", source="starfleet_jwt"
    )
    qualify = client.post(
        "/v1/agent-bundles/ab_test/qualification",
        json={"status": "qualified", "evidence": {}},
    )
    promote = client.post("/v1/agent-bundles/ab_test/promote")
    assert qualify.status_code == 404
    assert promote.status_code == 404


def test_admin_can_qualify_and_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "dev")
    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id="dev", source="starfleet_jwt"
    )
    conn = _connection(
        fetchone=_row(
            visibility="public",
            qualification_status="qualified",
            qualification_evidence={"rhacs": "admitted"},
            qualified_by="dev",
            qualified_at="2026-07-08T00:01:00Z",
        )
    )
    _override(conn)
    qualify = client.post(
        "/v1/agent-bundles/ab_test/qualification",
        json={"status": "qualified", "evidence": {"rhacs": "admitted"}},
    )
    assert qualify.status_code == 200, qualify.text
    promote = client.post("/v1/agent-bundles/ab_test/promote")
    assert promote.status_code == 200, promote.text
    assert promote.json()["visibility"] == "public"


def test_owner_can_delete_private_bundle() -> None:
    _override(_connection(fetchone={"id": "ab_test"}))
    response = client.delete("/v1/agent-bundles/ab_test")
    assert response.status_code == 200
    assert response.json() == {"id": "ab_test", "deleted": True}
