# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if

# Test data
multi_principal_mock_principals := {
    "user-id-123": {
        "workspaces": {
            "workspace-a": ["Viewer"]
        }
    },
    "user@example.com": {
        "workspaces": {
            "workspace-b": ["Editor"]
        }
    },
    "engineering-team": {
        "workspaces": {
            "workspace-c": ["Admin"]
        }
    }
}

multi_principal_mock_roles := {
    "Viewer": {
        "permissions": ["models.read", "models.list"]
    },
    "Editor": {
        "permissions": ["models.read", "models.list", "models.create", "models.update"]
    },
    "Admin": {
        "permissions": ["models.read", "models.list", "models.create", "models.update", "models.delete"]
    }
}

multi_principal_mock_endpoints := {
    "/apis/models/v2/workspaces/{workspace}/models/{name}": {
        "get": {
            "permissions": ["models.read"],
            "scopes": ["models:read"]
        },
        "delete": {
            "permissions": ["models.delete"],
            "scopes": ["models:write"]
        }
    }
}

# Test: User can access via principal_id
test_access_via_principal_id if {
    result := authz.allow with input as {
        "principal_id": "user-id-123",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/workspace-a/models/my-model"
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    with data.authz.endpoints as multi_principal_mock_endpoints
    
    result.allowed == true
}

# Test: User can access via email (different workspace)
test_access_via_email if {
    result := authz.allow with input as {
        "principal_id": "user-id-123",
        "principal_email": "user@example.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/workspace-b/models/my-model"
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    with data.authz.endpoints as multi_principal_mock_endpoints
    
    result.allowed == true
}

# Test: User can access via group membership
test_access_via_group if {
    result := authz.allow with input as {
        "principal_id": "user-id-123",
        "principal_email": "user@example.com",
        "principal_groups": ["engineering-team", "qa-team"],
        "method": "GET",
        "path": "/apis/models/v2/workspaces/workspace-c/models/my-model"
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    with data.authz.endpoints as multi_principal_mock_endpoints
    
    result.allowed == true
}

# Test: User denied when none of the principals have access
test_denied_when_no_principal_has_access if {
    result := authz.allow with input as {
        "principal_id": "user-id-123",
        "principal_email": "user@example.com",
        "principal_groups": ["engineering-team"],
        "method": "GET",
        "path": "/apis/models/v2/workspaces/workspace-d/models/my-model"  # workspace-d not accessible
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    with data.authz.endpoints as multi_principal_mock_endpoints
    
    result.allowed == false
}

# Test: Group has higher permissions than individual
test_group_higher_permissions if {
    # User via principal_id can only read (Viewer)
    # But via group membership has delete permission (Admin)
    result := authz.allow with input as {
        "principal_id": "user-id-123",
        "principal_groups": ["engineering-team"],
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/workspace-c/models/my-model"
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    with data.authz.endpoints as multi_principal_mock_endpoints
    
    result.allowed == true
}

# Test: Cannot delete in workspace where only Viewer
test_viewer_cannot_delete if {
    result := authz.allow with input as {
        "principal_id": "user-id-123",
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/workspace-a/models/my-model"
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    with data.authz.endpoints as multi_principal_mock_endpoints
    
    result.allowed == false
}

# Test: has_permissions API with multiple principals
test_has_permissions_via_email if {
    result := authz.has_permissions with input as {
        "principal_id": "user-id-123",
        "principal_email": "user@example.com",
        "workspace": "workspace-b",
        "permissions": ["models.create"]
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    
    result.allowed == true
}

# Test: has_permissions with group
test_has_permissions_via_group if {
    result := authz.has_permissions with input as {
        "principal_id": "user-id-123",
        "principal_groups": ["engineering-team"],
        "workspace": "workspace-c",
        "permissions": ["models.delete"]
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    
    result.allowed == true
}

# Test: Multi-principal list allowed
test_filters_combine_all_workspaces if {
    result := authz.allow with input as {
        "principal_id": "user-id-123",
        "principal_email": "user@example.com",
        "principal_groups": ["engineering-team"],
        "method": "GET",
        "path": "/apis/models/v2/workspaces/-/models"
    }
    with data.authz.principals as multi_principal_mock_principals
    with data.authz.roles as multi_principal_mock_roles
    with data.authz.endpoints as {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read"]
            }
        }
    }
    
    result.allowed == true
}
