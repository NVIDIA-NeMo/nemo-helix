# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import data.common

# Test data with endpoints requiring multiple permissions
mock_multi_permission_data := {
    "roles": {
        "Viewer": {
            "permissions": ["models.read", "models.list"]
        },
        "PartialEditor": {
            "permissions": ["models.read", "models.update"]
        },
        "FullEditor": {
            "permissions": ["models.read", "models.update", "models.delete"]
        }
    },
    "endpoints": {
        "/apis/models/v2/workspaces/{workspace}/models/{name}": {
            "get": {"permissions": ["models.read"]},
            "patch": {"permissions": ["models.read", "models.update"]},
            "delete": {"permissions": ["models.read", "models.update", "models.delete"]}
        }
    },
    "workspaces": {
        "test-ns": {},
        "public-ns": {}
    },
    "principals": {
        "viewer@test.com": {
            "workspaces": {"test-ns": ["Viewer"]}
        },
        "partial-editor@test.com": {
            "workspaces": {"test-ns": ["PartialEditor"]}
        },
        "full-editor@test.com": {
            "workspaces": {"test-ns": ["FullEditor"]}
        },
        # Wildcard principal grants Viewer access to public-ns for all authenticated users
        "*": {
            "workspaces": {"public-ns": ["Viewer"]}
        }
    }
}

# Test: Single permission requirement (should still work)
test_single_permission_required if {
    result := authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    with data.authz.principals as mock_multi_permission_data.principals

    result.allowed == true
}

# Test: Multiple permissions - user has all required permissions
test_multiple_permissions_all_satisfied if {
    result := authz.allow with input as {
        "principal_id": "partial-editor@test.com",
        "method": "PATCH",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    with data.authz.principals as mock_multi_permission_data.principals

    result.allowed == true
}

# Test: Multiple permissions - user missing one permission (should fail)
test_multiple_permissions_missing_one if {
    result := authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "PATCH",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    with data.authz.principals as mock_multi_permission_data.principals

    result.allowed == false
}

# Test: Three permissions required - user has all
test_three_permissions_all_satisfied if {
    result := authz.allow with input as {
        "principal_id": "full-editor@test.com",
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    with data.authz.principals as mock_multi_permission_data.principals

    result.allowed == true
}

# Test: Three permissions required - user has only two (should fail)
test_three_permissions_missing_one if {
    result := authz.allow with input as {
        "principal_id": "partial-editor@test.com",
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    with data.authz.principals as mock_multi_permission_data.principals

    result.allowed == false
}

# Test: User missing a permission that they don't have in their role
test_missing_delete_permission if {
    result := authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    with data.authz.principals as mock_multi_permission_data.principals

    result.allowed == false
}

# Test: all_permissions_satisfied helper function directly
test_all_permissions_satisfied_helper if {
    # Test with all permissions present
    common.all_permissions_satisfied(
        ["models.read", "models.update"],
        {"models.read", "models.update", "models.delete"}
    )
}

# Test: all_permissions_satisfied helper function with missing permission
test_all_permissions_satisfied_helper_missing if {
    not common.all_permissions_satisfied(
        ["models.read", "models.export"],
        {"models.read", "models.update", "models.delete"}
    )
}

# Test: all_permissions_satisfied with empty required permissions
test_all_permissions_satisfied_empty if {
    common.all_permissions_satisfied(
        [],
        {"models.read", "models.update"}
    )
}

# Test: Wildcard principal with multiple permissions (should work if all are in Viewer role)
test_public_workspace_multiple_permissions if {
    result := authz.allow with input as {
        "principal_id": "anyone@test.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/public-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    # Include wildcard principal that grants Viewer access to public-ns
    with data.authz.principals as {
        "anyone@test.com": {"workspaces": {}},
        "*": {"workspaces": {"public-ns": ["Viewer"]}}
    }

    result.allowed == true
}

# Test for allow with multiple permissions
test_allow_multiple_permissions if {
    result := authz.allow with input as {
        "principal_id": "partial-editor@test.com",
        "method": "PATCH",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    with data.authz.principals as mock_multi_permission_data.principals

    result.allowed == true
}

# Test for allow with missing permissions
test_allow_missing_permissions if {
    result := authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "PATCH",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_multi_permission_data.roles
    with data.authz.endpoints as mock_multi_permission_data.endpoints
    with data.authz.workspaces as mock_multi_permission_data.workspaces
    with data.authz.principals as mock_multi_permission_data.principals

    result.allowed == false
}
