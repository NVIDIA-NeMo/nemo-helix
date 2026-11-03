# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit service entity grants through the production WASM runtime."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from nhx.core.auth.app.embedded_pdp import OPAPolicy

ASSETS = Path(__file__).parents[1] / "src/nhx/core/auth/assets"
ENTITY_PERMISSIONS = ("entities.create", "entities.read", "entities.update", "entities.delete")
SERVICE = "service:scaled-evals"


@pytest.fixture
def policy_data() -> dict[str, Any]:
    data = yaml.safe_load((ASSETS / "static-authz.yaml").read_text())
    data["authz"]["principals"] = {
        SERVICE: {"workspaces": {"private": ["ServiceEntityStore"]}},
        "human": {"workspaces": {"private": ["ServiceEntityStore"]}},
    }
    data["authz"]["roles"]["Combined"] = {"includes": ["Admin", "ServiceEntityStore"]}
    return data


@pytest.fixture
def policy() -> OPAPolicy:
    return OPAPolicy(str(ASSETS / "policy.wasm"))


def allows(
    policy: OPAPolicy,
    data: dict[str, Any],
    permission: str,
    *,
    principal: str = SERVICE,
    workspace: str = "private",
    on_behalf_of: str = "",
) -> bool:
    policy.set_data(data)
    result = policy.evaluate(
        {
            "principal_id": principal,
            "workspace": workspace,
            "permissions": [permission],
            "on_behalf_of_principal_id": on_behalf_of,
        },
        entrypoint=1,
    )
    return result[0]["result"]["allowed"]


def test_static_storage_role_has_only_entity_permissions(policy_data: dict[str, Any]) -> None:
    roles = policy_data["authz"]["roles"]
    assert set(roles["ServiceEntityStore"]["permissions"]) == set(ENTITY_PERMISSIONS)
    assert roles["ServiceEntityStore"]["service_only"] is True
    assert "includes" not in roles["ServiceEntityStore"]
    for action in ("create", "read", "update", "delete"):
        assert policy_data["authz"]["permissions"]["entities"][action].get("has_role", True) is True
    for role in ("Viewer", "Editor", "Admin"):
        assert not set(ENTITY_PERMISSIONS).intersection(roles[role]["permissions"])


@pytest.mark.parametrize("permission", ENTITY_PERMISSIONS)
@pytest.mark.parametrize("role,expected", [("ServiceEntityStore", True), ("Viewer", False), ("Admin", False)])
def test_explicit_entity_permissions(
    policy: OPAPolicy, policy_data: dict[str, Any], permission: str, role: str, expected: bool
) -> None:
    policy_data["authz"]["principals"][SERVICE]["workspaces"]["private"] = [role]
    assert allows(policy, policy_data, permission) is expected


@pytest.mark.parametrize("role,expected", [("Viewer", True), ("ServiceEntityStore", False), ("Admin", True)])
def test_workspace_read_is_independent(
    policy: OPAPolicy, policy_data: dict[str, Any], role: str, expected: bool
) -> None:
    policy_data["authz"]["principals"][SERVICE]["workspaces"]["private"] = [role]
    assert allows(policy, policy_data, "workspaces.read") is expected


@pytest.mark.parametrize("permission", (*ENTITY_PERMISSIONS, "workspaces.read"))
def test_admin_and_storage_roles_compose(policy: OPAPolicy, policy_data: dict[str, Any], permission: str) -> None:
    policy_data["authz"]["principals"][SERVICE]["workspaces"]["private"].append("Admin")
    assert allows(policy, policy_data, permission)


@pytest.mark.parametrize("role", ["ServiceEntityStore", "Combined"])
@pytest.mark.parametrize("permission", ENTITY_PERMISSIONS)
def test_humans_cannot_inherit_service_storage(
    policy: OPAPolicy, policy_data: dict[str, Any], role: str, permission: str
) -> None:
    policy_data["authz"]["principals"]["human"]["workspaces"]["private"] = [role]
    assert not allows(policy, policy_data, permission, principal="human")


def test_human_admin_does_not_gain_entity_access(policy: OPAPolicy, policy_data: dict[str, Any]) -> None:
    policy_data["authz"]["principals"]["human"]["workspaces"]["private"].append("Admin")
    assert allows(policy, policy_data, "workspaces.read", principal="human")
    assert not allows(policy, policy_data, "entities.read", principal="human")


@pytest.mark.parametrize("binding_principal", ["*", "team"])
def test_service_role_cannot_be_granted_through_a_group_or_wildcard(
    policy: OPAPolicy, policy_data: dict[str, Any], binding_principal: str
) -> None:
    policy_data["authz"]["principals"][binding_principal] = {"workspaces": {"private": ["ServiceEntityStore"]}}
    policy_data["authz"]["principals"]["human"] = {"workspaces": {"private": ["Viewer"]}}
    policy.set_data(policy_data)
    result = policy.evaluate(
        {
            "principal_id": "human",
            "principal_groups": ["team"],
            "workspace": "private",
            "permissions": ["entities.read"],
        },
        entrypoint=1,
    )
    assert result[0]["result"]["allowed"] is False


def test_delegation_does_not_use_the_actors_storage_role(policy: OPAPolicy, policy_data: dict[str, Any]) -> None:
    policy_data["authz"]["principals"]["human"]["workspaces"]["private"] = ["Viewer"]
    assert not allows(policy, policy_data, "entities.read", on_behalf_of="human")


def test_roles_in_different_workspaces_do_not_combine(policy: OPAPolicy, policy_data: dict[str, Any]) -> None:
    policy_data["authz"]["principals"][SERVICE]["workspaces"] = {
        "private": ["Admin"],
        "projection": ["ServiceEntityStore"],
    }
    assert not allows(policy, policy_data, "entities.read")
    assert not allows(policy, policy_data, "workspaces.read", workspace="projection")
    assert allows(policy, policy_data, "entities.read", workspace="projection")


@pytest.mark.parametrize("permission", ["iam.create", "iam.list", "workspaces.delete", "jobs.create", "secrets.read"])
def test_storage_role_does_not_grant_unrelated_operations(
    policy: OPAPolicy, policy_data: dict[str, Any], permission: str
) -> None:
    assert not allows(policy, policy_data, permission)


def test_unbound_workspace_retains_existing_service_fallback(policy: OPAPolicy, policy_data: dict[str, Any]) -> None:
    assert allows(policy, policy_data, "entities.read", workspace="unbound")


def test_explicit_management_grants_remain_usable(policy: OPAPolicy, policy_data: dict[str, Any]) -> None:
    policy_data["authz"]["principals"][SERVICE]["workspaces"]["private"].append("Admin")
    assert allows(policy, policy_data, "iam.list")
    assert allows(policy, policy_data, "iam.list", workspace="*")
