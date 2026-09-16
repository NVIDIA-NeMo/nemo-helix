# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Test data for platform admin scenarios
platform_admin_test_data := {
    "roles": {
        "Viewer": {
            "permissions": ["models.read", "models.list", "filesets.read", "filesets.list", "workspaces.read", "workspaces.list"]
        },
        "Editor": {
            "includes": ["Viewer"],
            "permissions": ["models.create", "models.update", "models.delete", "filesets.create", "filesets.update", "filesets.delete"]
        },
        "Admin": {
            "includes": ["Editor"],
            "permissions": ["workspaces.delete", "workspaces.members.list", "workspaces.members.create", "workspaces.members.update", "workspaces.members.delete", "workspaces.update"]
        },
        "PlatformAdmin": {
            "includes": ["Admin"],
            "permissions": ["iam.read", "iam.create", "iam.delete"]
        }
    },
    "endpoints": {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {"permissions": ["models.list"]}
        },
        "/apis/models/v2/workspaces/{workspace}/models/{name}": {
            "get": {"permissions": ["models.read"]},
            "delete": {"permissions": ["models.delete"]}
        },
        "/apis/files/v2/workspaces/{workspace}/filesets": {
            "post": {"permissions": ["filesets.create"]}
        },
        "/apis/auth/v2/iam/role-bindings": {
            "post": {"permissions": ["iam.create"]}
        },
        "/apis/entities/v2/workspaces/{name}": {
            "delete": {"permissions": ["workspaces.delete"]}
        }
    },
    "workspaces": {
        "system": {},
        "workspace1": {},
        "workspace2": {}
    },
    "principals": {
        "platform-admin@example.com": {
            "workspaces": {
                "system": ["PlatformAdmin"]
            }
        },
        "regular-admin@example.com": {
            "workspaces": {
                "system": ["Admin"],
                "workspace1": ["Admin"]
            }
        },
        "user@example.com": {
            "workspaces": {
                "workspace1": ["Editor"]
            }
        }
    }
}

# Test platform admin has access to everything
test_platform_admin_can_access_any_workspace if {
    # Platform admin can access workspace1
    result1 := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/workspace1/models/model1"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result1.allowed == true

    # Platform admin can access workspace2
    result2 := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "POST",
        "path": "/apis/files/v2/workspaces/workspace2/filesets"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result2.allowed == true

    # Platform admin can manage IAM
    result3 := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "POST",
        "path": "/apis/auth/v2/iam/role-bindings"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result3.allowed == true
}

# Test regular admin in system workspace doesn't have platform admin powers
test_regular_admin_limited_access if {
    # Regular admin can't access IAM endpoints
    result1 := authz.allow with input as {
        "principal_id": "regular-admin@example.com",
        "method": "POST",
        "path": "/apis/auth/v2/iam/role-bindings"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result1.allowed == false

    # Regular admin can access their own workspace
    result2 := authz.allow with input as {
        "principal_id": "regular-admin@example.com",
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/workspace1/models/model1"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result2.allowed == true

    # Regular admin can't delete from workspace they don't have access to
    result3 := authz.allow with input as {
        "principal_id": "regular-admin@example.com",
        "method": "DELETE",
        "path": "/apis/entities/v2/workspaces/workspace3"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result3.allowed == false
}

# Test platform admin is allowed
test_platform_admin_no_filters if {
    result := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/-/models"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result.allowed == true
}

# Test regular user is allowed
test_regular_user_gets_filters if {
    result := authz.allow with input as {
        "principal_id": "user@example.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/-/models"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result.allowed == true
}

# Test platform admin is DENIED direct access to secret values
test_platform_admin_denied_secret_value_access if {
    result := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "GET",
        "path": "/apis/secrets/v2/workspaces/workspace1/secrets/my-secret/access"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result.allowed == false
}

# Test platform admin is denied secret value access even with query params
test_platform_admin_denied_secret_value_access_with_query if {
    result := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "GET",
        "path": "/apis/secrets/v2/workspaces/workspace1/secrets/my-secret/access?version=1"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result.allowed == false
}

# Test platform admin can still list secrets (not blocked by deny rule)
test_platform_admin_can_list_secrets if {
    result := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "GET",
        "path": "/apis/secrets/v2/workspaces/workspace1/secrets"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result.allowed == true
}

# Test platform admin can still delete secrets (not blocked by deny rule)
test_platform_admin_can_delete_secrets if {
    result := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "DELETE",
        "path": "/apis/secrets/v2/workspaces/workspace1/secrets/my-secret"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result.allowed == true
}

# Test platform admin can still read secret metadata (not blocked by deny rule)
test_platform_admin_can_read_secret_metadata if {
    result := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "GET",
        "path": "/apis/secrets/v2/workspaces/workspace1/secrets/my-secret"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as platform_admin_test_data.endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result.allowed == true
}

# Endpoints that include a service-only route (callers: ["service_principal"]) for
# exercising the caller-kind deny against a PlatformAdmin caller.
service_only_endpoints := {
    "/apis/models/v2/workspaces/{workspace}/models/{name}": {
        "delete": {"permissions": ["models.delete"], "callers": ["service_principal"]}
    }
}

# Test platform admin is ALLOWED on a service-only route — the admin global bypass holds
# here (only non-admin humans are denied on service-only routes).
test_platform_admin_allowed_on_service_only_route if {
    result := authz.allow with input as {
        "principal_id": "platform-admin@example.com",
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/workspace1/models/model1"
    }
    with data.authz.roles as platform_admin_test_data.roles
    with data.authz.endpoints as service_only_endpoints
    with data.authz.workspaces as platform_admin_test_data.workspaces
    with data.authz.principals as platform_admin_test_data.principals

    result.allowed == true
}
