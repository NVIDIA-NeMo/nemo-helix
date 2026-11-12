# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for /me, preferences, preference defaults, and profile resolution with auth enabled."""

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from nhx.common.config import get_service_config
from nhx.core.auth.api.v2.profiles.endpoints import get_config
from nhx.core.auth.config import AuthServiceConfig
from nhx.testing.client import TEST_ADMIN_EMAIL, TEST_USER_EMAIL, create_test_client

pytestmark = pytest.mark.xdist_group("auth_profiles")

ME = "/apis/auth/v2/me"
PREFS = "/apis/auth/v2/me/preferences"
DEFAULTS = "/apis/auth/v2/preferences/defaults"
CONFIG_DEFAULTS = {"studio.ui": {"theme": "dark", "density": "comfortable"}}


def _config_with_defaults() -> AuthServiceConfig:
    return get_service_config(AuthServiceConfig).model_copy(update={"preference_defaults": CONFIG_DEFAULTS})


@pytest.fixture(scope="module")
def http_client() -> Iterator[TestClient]:
    with create_test_client(
        client_type=TestClient,
        auth_enabled=True,
        dependency_overrides={get_config: _config_with_defaults},
    ) as client:
        yield client


def _user(principal: str | None = None) -> dict[str, str]:
    principal = principal or f"user-{uuid.uuid4().hex[:8]}@example.com"
    return {"X-NHX-Principal-Id": principal, "X-NHX-Principal-Email": principal}


def test_me_resolves_account_and_falls_back_to_email(http_client: TestClient) -> None:
    headers = _user()
    response = http_client.get(ME, headers=headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["account_id"].startswith("account-")
    assert body["principal_id"] == headers["X-NHX-Principal-Id"]
    assert body["auth_method"] == "trusted_header"
    assert body["caller_kind"] == "principal"
    assert body["can_edit_profile"] is True
    assert body["is_platform_admin"] is False
    assert body["profile"]["display_name"] == headers["X-NHX-Principal-Email"]
    assert http_client.get(ME, headers=headers).json()["account_id"] == body["account_id"]


def test_me_requires_authentication(http_client: TestClient) -> None:
    assert http_client.get(ME).status_code == 401


def test_platform_admin_flag(http_client: TestClient) -> None:
    assert http_client.get(ME, headers=_user(TEST_ADMIN_EMAIL)).json()["is_platform_admin"] is True


def test_profile_update_sets_and_clears_fields(http_client: TestClient) -> None:
    headers = _user()
    updated = http_client.patch(
        f"{ME}/profile", json={"display_name": " Alice A. ", "timezone": "Europe/Paris"}, headers=headers
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["profile"]["display_name"] == "Alice A."
    assert updated.json()["profile"]["display_name_override"] == "Alice A."
    assert updated.json()["profile"]["timezone"] == "Europe/Paris"

    cleared = http_client.patch(f"{ME}/profile", json={"display_name": None}, headers=headers)
    assert cleared.json()["profile"]["display_name"] == headers["X-NHX-Principal-Email"]
    assert cleared.json()["profile"]["timezone"] == "Europe/Paris"

    assert http_client.patch(f"{ME}/profile", json={"timezone": "Mars/Base"}, headers=headers).status_code == 422
    assert http_client.patch(f"{ME}/profile", json={"email": "x@example.com"}, headers=headers).status_code == 422


def test_preferences_round_trip_with_defaults(http_client: TestClient) -> None:
    headers = _user()
    empty = http_client.get(f"{PREFS}/studio.ui", headers=headers)
    assert empty.status_code == 200, empty.text
    assert empty.json()["value"] == {}
    assert empty.json()["db_version"] == 0
    assert empty.json()["effective"] == CONFIG_DEFAULTS["studio.ui"]

    put = http_client.put(f"{PREFS}/studio.ui", json={"theme": "light"}, headers={**headers, "If-Match": "0"})
    assert put.status_code == 200, put.text
    assert put.json()["effective"] == {"theme": "light", "density": "comfortable"}

    patched = http_client.patch(
        f"{PREFS}/studio.ui",
        json={"theme": None, "beta": True},
        headers={**headers, "Content-Type": "application/merge-patch+json"},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["value"] == {"beta": True}
    assert patched.json()["effective"] == {"theme": "dark", "density": "comfortable", "beta": True}

    stale = http_client.put(
        f"{PREFS}/studio.ui", json={}, headers={**headers, "If-Match": str(put.json()["db_version"])}
    )
    assert stale.status_code == 412
    assert stale.json()["code"] == "preference_version_conflict"

    listed = http_client.get(PREFS, params={"namespace": "studio"}, headers=headers)
    assert [doc["namespace"] for doc in listed.json()["data"]] == ["studio.ui"]

    assert http_client.delete(f"{PREFS}/studio.ui", headers=headers).status_code == 204
    assert http_client.get(f"{PREFS}/studio.ui", headers=headers).json()["db_version"] == 0


def test_preferences_are_private_to_each_account(http_client: TestClient) -> None:
    alice, bob = _user(), _user()
    http_client.put(f"{PREFS}/cli", json={"output": "json"}, headers=alice)
    assert http_client.get(f"{PREFS}/cli", headers=bob).json()["value"] == {}


def test_preference_validation(http_client: TestClient) -> None:
    headers = _user()
    assert http_client.put(f"{PREFS}/Bad Namespace", json={}, headers=headers).status_code == 422
    assert http_client.put(f"{PREFS}/cli", json=["not", "an", "object"], headers=headers).status_code == 422
    too_large = http_client.put(f"{PREFS}/cli", json={"blob": "x" * 70_000}, headers=headers)
    assert too_large.status_code == 413
    assert too_large.json()["code"] == "preference_too_large"


def test_workspace_preferences_require_access(http_client: TestClient) -> None:
    member = _user(TEST_USER_EMAIL)
    ok = http_client.put(f"{PREFS}/studio.onboarding", params={"workspace": "default"}, json={"x": 1}, headers=member)
    assert ok.status_code == 200, ok.text
    assert ok.json()["workspace"] == "default"
    assert http_client.get(f"{PREFS}/studio.onboarding", headers=member).json()["value"] == {}

    missing = http_client.get(f"{PREFS}/studio.onboarding", params={"workspace": "no-such-ws"}, headers=member)
    assert missing.status_code == 404


def test_service_accounts_have_no_preferences(http_client: TestClient) -> None:
    service = {"X-NHX-Principal-Id": "service:integration-test"}
    assert http_client.get(ME, headers=service).json()["caller_kind"] == "service_principal"
    assert http_client.get(f"{PREFS}/cli", headers=service).status_code == 403
    assert http_client.patch(f"{ME}/profile", json={"display_name": "x"}, headers=service).status_code == 403


def test_delegated_callers_read_but_cannot_write(http_client: TestClient) -> None:
    user = _user(TEST_USER_EMAIL)
    http_client.put(f"{PREFS}/cli", json={"output": "table"}, headers=user)
    delegated = {"X-NHX-Principal-Id": "service:integration-test", "X-NHX-Principal-On-Behalf-Of": TEST_USER_EMAIL}

    me = http_client.get(ME, headers=delegated)
    assert me.status_code == 200, me.text
    assert me.json()["principal_id"] == TEST_USER_EMAIL
    assert me.json()["can_edit_profile"] is False
    assert http_client.get(f"{PREFS}/cli", headers=delegated).json()["value"] == {"output": "table"}
    assert http_client.put(f"{PREFS}/cli", json={}, headers=delegated).status_code == 403
    assert http_client.patch(f"{ME}/profile", json={"display_name": "Mallory"}, headers=delegated).status_code == 403


def test_defaults_admin_writes_and_config_managed_namespaces(http_client: TestClient) -> None:
    admin, user = _user(TEST_ADMIN_EMAIL), _user()
    namespace = f"plugin.test-{uuid.uuid4().hex[:6]}"

    assert http_client.put(f"{DEFAULTS}/{namespace}", json={"a": 1}, headers=user).status_code == 403
    created = http_client.put(f"{DEFAULTS}/{namespace}", json={"a": 1, "b": 2}, headers=admin)
    assert created.status_code == 200, created.text
    assert created.json()["managed_by"] == "api"

    http_client.patch(f"{PREFS}/{namespace}", json={"b": 3}, headers=user)
    assert http_client.get(f"{PREFS}/{namespace}", headers=user).json()["effective"] == {"a": 1, "b": 3}

    config_managed = http_client.put(f"{DEFAULTS}/studio.ui", json={}, headers=admin)
    assert config_managed.status_code == 409
    assert config_managed.json()["managed_by"] == "config"

    listed = {item["namespace"]: item for item in http_client.get(DEFAULTS, headers=user).json()["data"]}
    assert listed["studio.ui"]["managed_by"] == "config"
    assert listed[namespace]["value"] == {"a": 1, "b": 2}

    assert http_client.delete(f"{DEFAULTS}/{namespace}", headers=admin).status_code == 204
    assert http_client.get(f"{PREFS}/{namespace}", headers=user).json()["effective"] == {"b": 3}


WORKSPACES = "/apis/entities/v2/workspaces"
RESOLVE = "/apis/auth/v2/profiles/resolve"


def _shared_workspace(http_client: TestClient, owner: dict[str, str], *members: dict[str, str]) -> str:
    """A workspace the owner creates (gaining Admin) and adds members to by email or group."""
    name = f"ws-{uuid.uuid4().hex[:8]}"
    created = http_client.post(WORKSPACES, json={"name": name}, headers=owner)
    assert created.status_code in (200, 201), created.text
    for member in members:
        principal = member.get("X-NHX-Principal-Groups") or member["X-NHX-Principal-Email"]
        added = http_client.post(
            f"{WORKSPACES}/{name}/members", json={"principal": principal, "roles": ["Viewer"]}, headers=owner
        )
        assert added.status_code in (200, 201), added.text
    return name


def _resolve(http_client: TestClient, caller: dict[str, str], *principals: str) -> list[dict]:
    response = http_client.post(RESOLVE, json={"principals": list(principals)}, headers=caller)
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_resolve_returns_public_fields_only(http_client: TestClient) -> None:
    alice, bob = _user(), _user()
    http_client.patch(f"{ME}/profile", json={"display_name": "Alice A."}, headers=alice)
    _shared_workspace(http_client, alice, bob)

    alice_entry, nobody, wildcard = _resolve(http_client, bob, alice["X-NHX-Principal-Id"], "nobody@example.com", "*")

    assert alice_entry["display_name"] == "Alice A."
    assert alice_entry["display_name_is_override"] is True
    assert set(alice_entry) == {
        "principal",
        "account_id",
        "display_name",
        "idp_display_name",
        "display_name_is_override",
        "avatar_url",
        "status",
        "ambiguous",
    }
    assert nobody["display_name"] is None
    assert wildcard["display_name"] is None

    too_many = http_client.post("/apis/auth/v2/profiles/resolve", json={"principals": ["x"] * 101}, headers=_user())
    assert too_many.status_code == 422


def test_resolve_hides_accounts_outside_the_callers_explicit_workspaces(http_client: TestClient) -> None:
    alice, bob, stranger = _user(), _user(), _user()
    group_member = {**_user(), "X-NHX-Principal-Groups": f"team-{uuid.uuid4().hex[:8]}"}
    for headers in (alice, bob, stranger):
        http_client.patch(f"{ME}/profile", json={"display_name": "Named"}, headers=headers)
    _shared_workspace(http_client, alice, bob, group_member)
    principals = [h["X-NHX-Principal-Id"] for h in (bob, stranger, alice)]

    # Everyone shares `default` and `system` through `*` bindings; those don't count.
    bob_entry, stranger_entry, self_entry = _resolve(http_client, alice, *principals)
    assert bob_entry["display_name"] == "Named"
    assert self_entry["display_name"] == "Named"
    assert stranger_entry == {
        "principal": stranger["X-NHX-Principal-Id"],
        "account_id": None,
        "display_name": None,
        "idp_display_name": None,
        "display_name_is_override": False,
        "avatar_url": None,
        "status": None,
        "ambiguous": False,
    }

    # Membership through a group binding counts for the caller.
    assert _resolve(http_client, group_member, alice["X-NHX-Principal-Id"])[0]["display_name"] == "Named"
    assert _resolve(http_client, stranger, alice["X-NHX-Principal-Id"])[0]["account_id"] is None
    admin_view = _resolve(http_client, _user(TEST_ADMIN_EMAIL), stranger["X-NHX-Principal-Id"])
    assert admin_view[0]["display_name"] == "Named"
