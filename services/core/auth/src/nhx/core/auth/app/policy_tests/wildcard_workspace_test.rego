# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if

# Test data for wildcard workspace scenarios (models API — avoids nested-entities deny for user principals)
wildcard_workspace_test_data := {
    "roles": {
        "Viewer": {
            "permissions": ["models.read", "models.list"]
        },
        "Editor": {
            "includes": ["Viewer"],
            "permissions": ["models.create", "models.update", "models.delete"]
        },
        "ServiceSystem": {
            "permissions": ["*"]
        }
    },
    "endpoints": {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {"permissions": ["models.list"]},
            "post": {"permissions": ["models.create"]}
        },
        "/apis/models/v2/workspaces/{workspace}/models/{name}": {
            "get": {"permissions": ["models.read"]},
            "patch": {"permissions": ["models.update"]},
            "delete": {"permissions": ["models.delete"]}
        }
    },
    "workspaces": {
        "test-workspace": {},
        "other-workspace": {}
    },
    "principals": {
        "viewer@test.com": {
            "workspaces": {"test-workspace": ["Viewer"]}
        },
        "multi-viewer@test.com": {
            "workspaces": {
                "test-workspace": ["Viewer"],
                "other-workspace": ["Viewer"]
            }
        },
        "no-access@test.com": {
            "workspaces": {}
        }
    }
}

# ============================================================================
# Test: Wildcard workspace "-" in path - should work for authenticated users
# ============================================================================

test_wildcard_workspace_get_authenticated if {
    # Authenticated user with roles should be allowed to use GET with wildcard workspace
    result := authz.allow
        with input as {
            "principal_id": "viewer@test.com",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == true
}

test_wildcard_workspace_head_authenticated if {
    # Authenticated user with roles should be allowed to use HEAD with wildcard workspace
    result := authz.allow
        with input as {
            "principal_id": "viewer@test.com",
            "method": "HEAD",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == true
}

test_wildcard_workspace_get_unauthenticated if {
    # Unauthenticated user should be denied even with wildcard workspace
    result := authz.allow
        with input as {
            "principal_id": "",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == false
}

test_wildcard_workspace_get_authenticated_no_roles if {
    # User with principal_id but no role bindings CAN still use wildcard workspace
    # This is by design - they'll get an empty result filtered by the service
    result := authz.allow
        with input as {
            "principal_id": "no-access@test.com",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == true
}

# ============================================================================
# Test: Wildcard workspace should NOT work for write operations
# ============================================================================

test_wildcard_workspace_post_denied if {
    # POST with wildcard workspace should be denied (only GET/HEAD allowed with wildcard)
    result := authz.allow
        with input as {
            "principal_id": "viewer@test.com",
            "method": "POST",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == false
}

test_wildcard_workspace_put_denied if {
    # PATCH with wildcard workspace should be denied for non-service principals
    result := authz.allow
        with input as {
            "principal_id": "viewer@test.com",
            "method": "PATCH",
            "path": "/apis/models/v2/workspaces/-/models/my-model"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == false
}

test_wildcard_workspace_delete_denied if {
    # DELETE with wildcard workspace should be denied
    result := authz.allow
        with input as {
            "principal_id": "viewer@test.com",
            "method": "DELETE",
            "path": "/apis/models/v2/workspaces/-/models/my-model"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == false
}

# ============================================================================
# Test: Normal workspace access should still work
# ============================================================================

test_normal_workspace_get_authorized if {
    # GET on a specific workspace should work if user has access
    result := authz.allow
        with input as {
            "principal_id": "viewer@test.com",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/test-workspace/models/my-model"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == true
}

test_normal_workspace_get_unauthorized if {
    # GET on a workspace user doesn't have access to should be denied
    result := authz.allow
        with input as {
            "principal_id": "viewer@test.com",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/other-workspace/models/my-model"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == false
}

test_normal_workspace_post_authorized if {
    # POST on a specific workspace should work if user has Editor role
    result := authz.allow
        with input as {
            "principal_id": "editor@test.com",
            "method": "POST",
            "path": "/apis/models/v2/workspaces/test-workspace/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as {
            "editor@test.com": {
                "workspaces": {"test-workspace": ["Editor"]}
            }
        }

    result.allowed == true
}

# ============================================================================
# Test: Service principal bypass
# ============================================================================

test_service_principal_wildcard_workspace if {
    # Service principal should be allowed regardless of workspace
    result := authz.allow
        with input as {
            "principal_id": "service:entity-store",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == true
}

test_service_principal_wildcard_workspace_post if {
    # Service principal should be allowed for write operations too
    result := authz.allow
        with input as {
            "principal_id": "service:entity-store",
            "method": "POST",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == true
}

# ============================================================================
# Test: Platform admin bypass
# ============================================================================

test_platform_admin_wildcard_workspace if {
    # Platform admin should be allowed regardless of workspace
    result := authz.allow
        with input as {
            "principal_id": "admin@test.com",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as {
            "admin@test.com": {
                "workspaces": {"system": ["PlatformAdmin"]}
            }
        }

    result.allowed == true
}

test_platform_admin_wildcard_workspace_post if {
    # Platform admin should be allowed for write operations too
    result := authz.allow
        with input as {
            "principal_id": "admin@test.com",
            "method": "POST",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as {
            "admin@test.com": {
                "workspaces": {"system": ["PlatformAdmin"]}
            }
        }

    result.allowed == true
}

# ============================================================================
# Test: Edge cases
# ============================================================================

test_wildcard_workspace_multiple_workspace_roles if {
    # User with roles in multiple workspaces should be allowed
    result := authz.allow
        with input as {
            "principal_id": "multi-viewer@test.com",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    result.allowed == true
}

test_wildcard_not_confused_with_real_workspace if {
    # Ensure "-" is treated as wildcard, not a failure to match workspace
    result := authz.allow
        with input as {
            "principal_id": "viewer@test.com",
            "method": "GET",
            "path": "/apis/models/v2/workspaces/-/models"
        }
        with data.authz.roles as wildcard_workspace_test_data.roles
        with data.authz.endpoints as wildcard_workspace_test_data.endpoints
        with data.authz.workspaces as wildcard_workspace_test_data.workspaces
        with data.authz.principals as wildcard_workspace_test_data.principals

    # Allowed via wildcard GET/HEAD rule for authenticated users
    result.allowed == true
}
