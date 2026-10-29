# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml
from nemo_helix_plugin.authz import AuthzContribution
from nemo_helix_plugin.authz_format import validate_static_authz_data
from nemo_helix_plugin.authz_merge import merge_authz_contributions


def test_merge_adds_endpoints_and_role_permissions() -> None:
    base = {
        "authz": {
            "permissions": {},
            "roles": {"Editor": {"permissions": ["jobs.read"]}, "Viewer": {"permissions": []}},
            "endpoints": {},
        }
    }
    overlay = [
        {
            "permissions": {
                "customization.automodel.jobs.create": "Create automodel jobs",
                "customization.automodel.jobs.read": "Read automodel jobs",
            },
            "endpoints": {
                "/apis/customization/v2/workspaces/{workspace}/automodel/jobs": {
                    "post": {
                        "permissions": ["customization.automodel.jobs.create"],
                        "scopes": ["customization:write", "platform:write"],
                    },
                },
            },
        }
    ]
    merged = merge_authz_contributions(base, overlay)
    validate_static_authz_data(merged)
    endpoints = merged["authz"]["endpoints"]
    assert "post" in endpoints["/apis/customization/v2/workspaces/{workspace}/automodel/jobs"]
    editor_perms = merged["authz"]["roles"]["Editor"]["permissions"]
    assert "customization.automodel.jobs.create" in editor_perms
    viewer_perms = merged["authz"]["roles"]["Viewer"]["permissions"]
    assert "customization.automodel.jobs.read" in viewer_perms


@pytest.mark.parametrize(
    ("permission_id", "expected_roles"),
    [
        ("scaled-evals.admin", []),
        ("example.items.admin", []),
        ("example.items.read", ["Viewer", "Editor"]),
        ("example.items.list", ["Viewer", "Editor"]),
        ("example.items.create", ["Editor"]),
        ("example.admin.read", ["Viewer", "Editor"]),
        ("example.admin.update", ["Editor"]),
        ("example.items.administrate", ["Editor"]),
    ],
)
def test_default_role_grants_require_an_exact_admin_action(permission_id: str, expected_roles: list[str]) -> None:
    contribution = AuthzContribution(permissions={permission_id: "Example permission"})
    merged = merge_authz_contributions({"authz": {}}, [contribution.to_dict()])

    granted_roles = {
        role_name for role_name, role in merged["authz"]["roles"].items() if permission_id in role["permissions"]
    }
    assert granted_roles == set(expected_roles)
    registry_leaf = merged["authz"]["permissions"]
    for segment in permission_id.split("."):
        registry_leaf = registry_leaf[segment]
    expected_leaf: dict[str, str | bool] = {"description": "Example permission"}
    if not expected_roles:
        expected_leaf["has_role"] = False
    assert registry_leaf == expected_leaf


def test_admin_permissions_preserve_explicit_grants_and_inputs() -> None:
    permission_id = "scaled-evals.admin"
    base = {"authz": {"roles": {"ExistingOperator": {"permissions": [permission_id]}}}}
    contribution = AuthzContribution(
        permissions={permission_id: "Operate Scaled Evals"},
        role_permissions={"ScaledEvalsOperator": [permission_id]},
    )
    contributions = [contribution.to_dict()]
    original_base = deepcopy(base)
    original_contributions = deepcopy(contributions)

    merged = merge_authz_contributions(base, contributions)

    assert merged["authz"]["roles"] == {
        "ExistingOperator": {"permissions": [permission_id]},
        "ScaledEvalsOperator": {"permissions": [permission_id]},
    }
    assert base == original_base
    assert contributions == original_contributions
    assert merge_authz_contributions(merged, contributions) == merged


def test_admin_registry_marker_survives_yaml_and_repeat_merges() -> None:
    contribution = AuthzContribution(permissions={"scaled-evals.admin": "Operate Scaled Evals"})
    contributions = [contribution.to_dict()]
    merged = merge_authz_contributions({"authz": {}}, contributions)

    restored = yaml.safe_load(yaml.safe_dump(merged))
    validate_static_authz_data(restored)

    assert restored["authz"]["permissions"]["scaled-evals"]["admin"] == {
        "description": "Operate Scaled Evals",
        "has_role": False,
    }
    assert restored["authz"]["roles"] == {}
    assert merge_authz_contributions(restored, contributions) == merged
