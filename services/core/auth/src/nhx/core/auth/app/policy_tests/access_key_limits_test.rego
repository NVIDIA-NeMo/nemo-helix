# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

access_key_limit_test_data := {
    "roles": {
        "Viewer": {"permissions": ["models.read", "models.list"]},
        "Admin": {"includes": ["Viewer"], "permissions": ["models.delete"]},
        "HelixAdmin": {"includes": ["Admin"], "permissions": ["iam.read"]}
    },
    "endpoints": {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {"permissions": ["models.list"], "scopes": ["models:read"]}
        },
        "/apis/models/v2/workspaces/{workspace}/models/{name}": {
            "get": {"permissions": ["models.read"], "scopes": ["models:read"]}
        },
        "/apis/files/v2/workspaces/{workspace}/filesets": {
            "get": {"permissions": []}
        },
        "/apis/entities/v2/workspaces": {
            "get": {"permissions": []}
        }
    },
    "scope_families": {"models": true},
    "workspaces": {"system": {}, "team-a": {}, "team-b": {}},
    "principals": {
        "service-account:ci": {"workspaces": {"team-a": ["Viewer"], "team-b": ["Viewer"]}},
        "service-account:root": {"workspaces": {"system": ["HelixAdmin"]}}
    }
}

limited_allow(request) := result if {
    result := authz.allow with input as request
        with data.authz.roles as access_key_limit_test_data.roles
        with data.authz.endpoints as access_key_limit_test_data.endpoints
        with data.authz.scope_families as access_key_limit_test_data.scope_families
        with data.authz.workspaces as access_key_limit_test_data.workspaces
        with data.authz.principals as access_key_limit_test_data.principals
}

limited_has_permissions(request) := result if {
    result := authz.has_permissions with input as request
        with data.authz.roles as access_key_limit_test_data.roles
        with data.authz.workspaces as access_key_limit_test_data.workspaces
        with data.authz.principals as access_key_limit_test_data.principals
}

limited_has_role(request) := result if {
    result := authz.has_role with input as request
        with data.authz.roles as access_key_limit_test_data.roles
        with data.authz.workspaces as access_key_limit_test_data.workspaces
        with data.authz.principals as access_key_limit_test_data.principals
}

test_limited_key_allowed_in_listed_workspace if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-a/models",
        "access_key_workspaces": ["team-a"]
    }).allowed == true
}

test_limited_key_denied_in_unlisted_workspace_despite_rbac if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-b/models",
        "access_key_workspaces": ["team-a"]
    }).allowed == false
}

test_no_limit_keeps_rbac_behavior if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-b/models",
        "access_key_workspaces": null
    }).allowed == true
}

test_limit_overrides_helix_admin_bypass if {
    limited_allow({
        "principal_id": "service-account:root",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-b/models",
        "access_key_workspaces": ["team-a"]
    }).allowed == false
}

test_empty_limit_denies_listed_rbac_workspace if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-a/models",
        "access_key_workspaces": []
    }).allowed == false
}

test_limited_key_denied_on_cross_workspace_list if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/-/models",
        "access_key_workspaces": ["team-a"]
    }).allowed == false
}

test_limited_key_denied_on_workspace_less_route if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces",
        "access_key_workspaces": ["team-a"]
    }).allowed == false
}

test_limited_key_with_matching_scope_allowed if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-a/models",
        "scopes": ["models:read", "models:write"],
        "access_key_workspaces": ["team-a"]
    }).allowed == true
}

test_limited_key_with_unknown_scope_family_denied if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-a/models",
        "scopes": ["models:read", "unknown:read"],
        "access_key_workspaces": ["team-a"]
    }).allowed == false
}

test_unlimited_key_with_unknown_scope_family_keeps_lenient_behavior if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-a/models",
        "scopes": ["models:read", "unknown:read"],
        "access_key_workspaces": null
    }).allowed == true
}

test_limited_scoped_key_denied_on_route_without_scopes if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/files/v2/workspaces/team-a/filesets",
        "scopes": ["models:read"],
        "access_key_workspaces": ["team-a"]
    }).allowed == false
}

test_limited_unscoped_key_allowed_on_route_without_scopes if {
    limited_allow({
        "principal_id": "service-account:ci",
        "method": "GET",
        "path": "/apis/files/v2/workspaces/team-a/filesets",
        "access_key_workspaces": ["team-a"]
    }).allowed == true
}

test_has_permissions_respects_limit if {
    limited_has_permissions({
        "principal_id": "service-account:ci",
        "workspace": "team-a",
        "permissions": ["models.read"],
        "access_key_workspaces": ["team-a"]
    }).allowed == true
    limited_has_permissions({
        "principal_id": "service-account:ci",
        "workspace": "team-b",
        "permissions": ["models.read"],
        "access_key_workspaces": ["team-a"]
    }).allowed == false
}

test_has_permissions_limit_overrides_helix_admin if {
    limited_has_permissions({
        "principal_id": "service-account:root",
        "workspace": "team-b",
        "permissions": ["models.read"],
        "access_key_workspaces": ["team-a"]
    }).allowed == false
}

test_has_role_respects_limit if {
    limited_has_role({
        "principal_id": "service-account:ci",
        "workspace": "team-a",
        "role": "Viewer",
        "access_key_workspaces": ["team-a"]
    }).has_role == true
    limited_has_role({
        "principal_id": "service-account:ci",
        "workspace": "team-b",
        "role": "Viewer",
        "access_key_workspaces": ["team-a"]
    }).has_role == false
}
