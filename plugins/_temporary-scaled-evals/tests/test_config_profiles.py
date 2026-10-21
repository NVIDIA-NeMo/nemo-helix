# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the config-profiles router.

Profiles live in the conftest's in-memory Entity Store. The mocked psycopg
connection only answers whether an active evaluation references a profile.
"""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("scaled_evals")

from api_test_fixture import client, v1
from scaled_evals.api.auth import CurrentPrincipal, current_principal
from scaled_evals.api.db import get_conn
from scaled_evals.api.tenancy import caller_is_admin

_VALID_GYM_CONFIG = {
    "command": "run_and_collect",
    "config_paths": ["/harness/gym-sandbox-opensandbox/configs/mini_swe_agent_opensandbox_smoke.yaml"],
    "agent_name": "mini_swe_agent_2",
}


@pytest.fixture
def in_use() -> MagicMock:
    """Connection whose active-evaluation lookup finds nothing unless a test says so."""
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
    return conn


@pytest.fixture(autouse=True)
def _override_db(in_use: MagicMock):
    v1.dependency_overrides[get_conn] = lambda: in_use
    yield
    v1.dependency_overrides.pop(get_conn, None)
    v1.dependency_overrides.pop(current_principal, None)
    v1.dependency_overrides[caller_is_admin] = lambda: True


def _act_as(owner_id: str, *, admin: bool = False) -> None:
    v1.dependency_overrides[current_principal] = lambda: CurrentPrincipal(
        owner_type="USER", owner_id=owner_id, source="platform"
    )
    v1.dependency_overrides[caller_is_admin] = lambda: admin


def _create(name: str = "profile", type: str = "harbor", config: dict | None = None) -> dict:
    response = client.post("/v1/config-profiles", json={"name": name, "type": type, "config": config or {}})
    assert response.status_code == 201, response.text
    return response.json()


def test_profile_crud_is_owner_scoped_and_searchable() -> None:
    _act_as("user-1")
    harbor = _create("Nightly Harbor")
    gym = _create("gym", "gym", _VALID_GYM_CONFIG)
    assert set(harbor) == {"id", "name", "type", "config", "created_at", "updated_at"}
    assert client.get(f"/v1/config-profiles/{harbor['id']}").json() == harbor

    assert [item["id"] for item in client.get("/v1/config-profiles").json()["data"]] == [gym["id"], harbor["id"]]
    assert [item["id"] for item in client.get("/v1/config-profiles", params={"type": "gym"}).json()["data"]] == [
        gym["id"]
    ]
    assert [item["id"] for item in client.get("/v1/config-profiles", params={"q": "nightly"}).json()["data"]] == [
        harbor["id"]
    ]
    first = client.get("/v1/config-profiles", params={"limit": 1, "order": "asc"}).json()
    assert [item["id"] for item in first["data"]] == [harbor["id"]]
    second = client.get("/v1/config-profiles", params={"limit": 1, "order": "asc", "cursor": first["next_cursor"]})
    assert [item["id"] for item in second.json()["data"]] == [gym["id"]]

    # Profiles are org-readable, but only the owner (or an admin) writes them.
    _act_as("user-2")
    assert client.get(f"/v1/config-profiles/{harbor['id']}").status_code == 200
    assert client.get("/v1/config-profiles", params={"mine": True}).json()["data"] == []
    assert client.patch(f"/v1/config-profiles/{harbor['id']}", json={"name": "hijacked"}).status_code == 404
    assert client.delete(f"/v1/config-profiles/{harbor['id']}").status_code == 404

    _act_as("admin-1", admin=True)
    renamed = client.patch(f"/v1/config-profiles/{harbor['id']}", json={"name": "renamed"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "renamed"
    assert renamed.json()["updated_at"] > harbor["updated_at"]

    _act_as("user-1")
    assert client.get("/v1/config-profiles", params={"mine": True}).json()["data"][1]["name"] == "renamed"
    assert client.delete(f"/v1/config-profiles/{harbor['id']}").json() == {"id": harbor["id"], "deleted": True}
    assert client.get(f"/v1/config-profiles/{harbor['id']}").status_code == 404
    assert client.patch(f"/v1/config-profiles/{harbor['id']}", json={"name": "gone"}).status_code == 404


def test_profile_referenced_by_an_active_evaluation_cannot_change(in_use: MagicMock) -> None:
    _act_as("user-1")
    profile = _create()
    in_use.cursor.return_value.__enter__.return_value.fetchone.return_value = {"id": "ev_active"}

    # A non-owner gets 404 and does not learn the profile is in use.
    _act_as("user-2")
    assert client.delete(f"/v1/config-profiles/{profile['id']}").status_code == 404
    _act_as("user-1")
    for response in (
        client.patch(f"/v1/config-profiles/{profile['id']}", json={"config": {"env": {}}}),
        client.delete(f"/v1/config-profiles/{profile['id']}"),
    ):
        assert response.status_code == 409
        assert response.json()["detail"]["error"]["code"] == "profile_in_use"
    # Renaming does not change what a run executes, so it is allowed.
    assert client.patch(f"/v1/config-profiles/{profile['id']}", json={"name": "renamed"}).status_code == 200
    query = in_use.cursor.return_value.__enter__.return_value.execute.call_args.args[0]
    assert "intake_profile_id" in query and "status NOT IN" in query


def test_dispatch_loads_unsnapshotted_profiles_from_the_entity_store() -> None:
    from scaled_evals.dispatch.worker import Dispatcher

    gym = _create("gym", "gym", _VALID_GYM_CONFIG)
    intake = _create("intake", "intake", {"workspace": "team", "app": "evals"})
    row = {"framework": "gym", "framework_profile_id": gym["id"], "intake_profile_id": intake["id"]}

    assert Dispatcher._profile_config(None, row, snapshot=None, role="framework") == _VALID_GYM_CONFIG
    assert Dispatcher._profile_config(None, row, snapshot=None, role="intake")["app"] == "evals"
    with pytest.raises(RuntimeError, match="harbor profile not found"):
        Dispatcher._profile_config(None, {**row, "harbor_profile_id": gym["id"]}, snapshot=None, role="harbor")


@pytest.mark.parametrize(
    ("profile_type", "config"),
    [
        ("harbor", {"env": []}),
        ("gym", {}),
        ("intake", {"workspace": 42}),
    ],
)
def test_patch_validates_config_against_existing_profile_type(profile_type: str, config: dict) -> None:
    valid = {"harbor": {}, "gym": _VALID_GYM_CONFIG, "intake": {"workspace": "team", "app": "evals"}}
    profile = _create(type=profile_type, config=valid[profile_type])
    response = client.patch(f"/v1/config-profiles/{profile['id']}", json={"config": config})

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "invalid_config"
    assert client.get(f"/v1/config-profiles/{profile['id']}").json()["config"] == valid[profile_type]


def test_missing_profiles_return_404() -> None:
    for response in (
        client.get("/v1/config-profiles/cfg_missing"),
        client.patch("/v1/config-profiles/cfg_missing", json={"name": "new"}),
        client.delete("/v1/config-profiles/cfg_missing"),
    ):
        assert response.status_code == 404
        assert response.json()["detail"]["error"]["code"] == "not_found"


# ---------- request-body validation (no store hit) -------------------------


def test_list_rejects_unknown_type_filter() -> None:
    response = client.get("/v1/config-profiles", params={"type": "bogus"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"name": "", "type": "harbor"},
        {"name": "x", "type": "bogus"},
        {"name": "retired", "type": "switchyard", "config": {}},
    ],
)
def test_create_rejects_invalid_bodies(body: dict) -> None:
    assert client.post("/v1/config-profiles", json=body).status_code == 422


@pytest.mark.parametrize(
    ("profile_type", "config"),
    [
        ("harbor", {"harbor_config": 123}),
        ("gym", {}),
        ("gym", {"command": "run_and_collect", "config_paths": ["/etc/passwd"], "agent_name": "mini_swe_agent_2"}),
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
    assert _create("intake", "intake", config)["config"] == config


def test_patch_rejects_attempt_to_change_immutable_type() -> None:
    response = client.patch("/v1/config-profiles/cfg_1", json={"type": "intake"})

    assert response.status_code == 422
