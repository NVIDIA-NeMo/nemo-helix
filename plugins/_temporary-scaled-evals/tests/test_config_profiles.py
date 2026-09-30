# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the config-profiles router.

Uses an in-process FastAPI TestClient with a mocked psycopg connection
(via dependency_overrides on the mounted /v1 sub-app). End-to-end
coverage against a real Postgres lives in tests/integration/.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import get_conn
from scaled_evals.api.settings import settings

_VALID_GYM_CONFIG = {
    "command": "run_and_collect",
    "config_paths": ["/harness/gym-sandbox-opensandbox/configs/mini_swe_agent_opensandbox_smoke.yaml"],
    "agent_name": "mini_swe_agent_2",
}


def _empty_db() -> Iterator[MagicMock]:
    """Fake connection whose queries return nothing — empty list, no row."""
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    yield conn


@pytest.fixture(autouse=True)
def _override_db():
    v1.dependency_overrides[get_conn] = _empty_db
    yield
    v1.dependency_overrides.pop(get_conn, None)
    v1.dependency_overrides.pop(current_principal, None)


def _use_principal(owner_id: str) -> None:
    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id=owner_id, source="starfleet_jwt"
    )


def _profile_row(**overrides) -> dict:
    row = {
        "id": "cfg_1",
        "name": "profile",
        "type": "harbor",
        "config": {},
        "created_at": "2026-07-10T00:00:00Z",
        "updated_at": "2026-07-10T00:00:00Z",
    }
    row.update(overrides)
    return row


# ---------- list ----------------------------------------------------------


def test_list_returns_envelope_shape() -> None:
    response = client.get("/v1/config-profiles")
    assert response.status_code == 200
    assert response.json() == {"data": [], "next_cursor": None}


def test_list_rejects_unknown_type_filter() -> None:
    response = client.get("/v1/config-profiles", params={"type": "bogus"})
    assert response.status_code == 422


def test_list_type_filter_scopes_query() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.get("/v1/config-profiles", params={"type": "intake"})

    assert response.status_code == 200
    select = cur.execute.call_args_list[0]
    assert "type = %s" in select.args[0]
    assert "intake" in select.args[1]


# ---------- get -----------------------------------------------------------


def test_get_returns_404_when_missing() -> None:
    response = client.get("/v1/config-profiles/cfg_does_not_exist")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


# ---------- create: request-body validation (no DB hit) -------------------
#
# Pydantic validation runs before dependency resolution, so these don't
# need a mock that emulates an INSERT — they're guarded at the boundary.


def test_create_rejects_empty_name() -> None:
    response = client.post("/v1/config-profiles", json={"name": "", "type": "harbor"})
    assert response.status_code == 422


def test_create_rejects_unknown_type() -> None:
    response = client.post("/v1/config-profiles", json={"name": "x", "type": "bogus"})
    assert response.status_code == 422


def test_create_accepts_gym_type() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {
        "id": "cfg_gym",
        "name": "gym",
        "type": "gym",
        "config": {},
        "created_at": "2026-06-06T00:00:00Z",
        "updated_at": "2026-06-06T00:00:00Z",
    }

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen

    response = client.post(
        "/v1/config-profiles",
        json={"name": "gym", "type": "gym", "config": _VALID_GYM_CONFIG},
    )

    assert response.status_code == 201, response.text
    assert response.json()["type"] == "gym"


@pytest.mark.parametrize(
    ("profile_type", "config"),
    [
        ("harbor", {"harbor_config": 123}),
        ("gym", {}),
        ("intake", {"app": "missing-workspace"}),
    ],
)
def test_create_rejects_invalid_config_for_each_profile_type(profile_type: str, config: dict) -> None:
    response = client.post(
        "/v1/config-profiles",
        json={"name": "invalid", "type": profile_type, "config": config},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_config"


@pytest.mark.parametrize(
    "config",
    [
        {"workspace": "team-workspace", "app": "scaled-evals"},
        {"intake_workspace": "legacy-workspace", "intake_app": "scaled-evals"},
    ],
)
def test_create_accepts_canonical_and_legacy_intake_workspace(config: dict) -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = _profile_row(type="intake", config=config)

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.post(
        "/v1/config-profiles",
        json={"name": "intake", "type": "intake", "config": config},
    )

    assert response.status_code == 201, response.text


def test_create_rejects_invalid_gym_profile_before_persistence() -> None:
    response = client.post(
        "/v1/config-profiles",
        json={
            "name": "unsafe gym",
            "type": "gym",
            "config": {
                "command": "run_and_collect",
                "config_paths": ["/etc/passwd"],
                "agent_name": "mini_swe_agent_2",
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_config"


def test_create_rejects_retired_switchyard_type() -> None:
    response = client.post(
        "/v1/config-profiles",
        json={"name": "retired", "type": "switchyard", "config": {}},
    )
    assert response.status_code == 422


# ---------- patch / delete: 404 on missing --------------------------------


def test_patch_returns_404_when_missing() -> None:
    response = client.patch("/v1/config-profiles/cfg_missing", json={"name": "new"})
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    ("profile_type", "config"),
    [
        ("harbor", {"env": []}),
        ("gym", {}),
        ("intake", {"workspace": 42}),
    ],
)
def test_patch_validates_config_against_existing_profile_type(profile_type: str, config: dict) -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = _profile_row(type=profile_type)

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.patch("/v1/config-profiles/cfg_1", json={"config": config})

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_config"
    assert not any("UPDATE config_profiles" in call.args[0] for call in cur.execute.call_args_list)


def test_patch_rejects_attempt_to_change_immutable_type() -> None:
    response = client.patch("/v1/config-profiles/cfg_1", json={"type": "intake"})

    assert response.status_code == 422


def test_patch_config_returns_409_when_active_evaluation_references_profile() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [
        {
            "id": "cfg_in_use",
            "name": "profile",
            "type": "harbor",
            "config": {},
            "created_at": "2026-07-10T00:00:00Z",
            "updated_at": "2026-07-10T00:00:00Z",
        },
        {"id": "cfg_in_use"},
        {"id": "ev_active"},
    ]

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.patch("/v1/config-profiles/cfg_in_use", json={"config": {"x": 1}})

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "profile_in_use"
    calls = cur.execute.call_args_list
    assert "FOR UPDATE" in calls[1].args[0]
    assert "status NOT IN" in calls[2].args[0]


def test_delete_returns_404_when_missing() -> None:
    response = client.delete("/v1/config-profiles/cfg_missing")
    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"


def test_delete_returns_409_when_active_evaluation_references_profile() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = {"id": "ev_active"}

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen

    response = client.delete("/v1/config-profiles/cfg_in_use")

    assert response.status_code == 409
    assert response.json()["detail"]["error"]["code"] == "profile_in_use"
    query = cur.execute.call_args_list[0].args[0]
    assert "framework_profile_id = %s" in query


def test_delete_allows_profile_when_no_active_evaluation_references_it() -> None:
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [None, {"id": "cfg_done"}]

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen

    response = client.delete("/v1/config-profiles/cfg_done")

    assert response.status_code == 200
    assert response.json() == {"id": "cfg_done", "deleted": True}


# ---------- ownership: create stamps owner, writes are owner-scoped --------


def test_create_stamps_caller_as_owner() -> None:
    _use_principal("user-1")
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = _profile_row()

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.post("/v1/config-profiles", json={"name": "profile", "type": "harbor"})

    assert response.status_code == 201, response.text
    insert = next(call for call in cur.execute.call_args_list if "INSERT INTO config_profiles" in call.args[0])
    assert "owner_id" in insert.args[0]
    assert insert.args[1][-1] == "user-1"


def test_patch_by_non_owner_returns_404() -> None:
    _use_principal("user-2")
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    # get() finds the profile; the owner-scoped UPDATE matches no row.
    cur.fetchone.side_effect = [_profile_row(), None]

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.patch("/v1/config-profiles/cfg_1", json={"name": "hijacked"})

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "not_found"
    update = next(call for call in cur.execute.call_args_list if "UPDATE config_profiles" in call.args[0])
    assert "owner_id = %s" in update.args[0]
    assert "user-2" in update.args[1]


def test_delete_by_non_owner_returns_404() -> None:
    _use_principal("user-2")
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    # No active evaluation references it; the owner-scoped UPDATE matches no row.
    cur.fetchone.side_effect = [None, None]

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.delete("/v1/config-profiles/cfg_1")

    assert response.status_code == 404
    update = next(call for call in cur.execute.call_args_list if "UPDATE config_profiles" in call.args[0])
    assert "owner_id = %s" in update.args[0]
    assert "user-2" in update.args[1]


def test_admin_patch_bypasses_owner_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "control_plane_admin_subjects", "admin-1")
    _use_principal("admin-1")
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [_profile_row(), _profile_row(name="renamed")]

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.patch("/v1/config-profiles/cfg_1", json={"name": "renamed"})

    assert response.status_code == 200, response.text
    update = next(call for call in cur.execute.call_args_list if "UPDATE config_profiles" in call.args[0])
    assert "owner_id = %s" not in update.args[0]


def test_list_mine_scopes_to_caller() -> None:
    _use_principal("user-1")
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = []

    def _gen() -> Iterator[MagicMock]:
        yield conn

    v1.dependency_overrides[get_conn] = _gen
    response = client.get("/v1/config-profiles", params={"mine": "true"})

    assert response.status_code == 200
    select = cur.execute.call_args_list[0]
    assert "owner_id = %s" in select.args[0]
    assert "user-1" in select.args[1]
