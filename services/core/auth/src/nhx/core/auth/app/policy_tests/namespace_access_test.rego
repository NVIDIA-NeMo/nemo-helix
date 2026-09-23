# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if

# Test data matching the Python integration test scenarios
workspace_access_test_data := {
    "roles": {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        },
        "Editor": {
            "includes": ["Viewer"],
            "permissions": ["workspaces.update"]
        },
        "WorkspaceCreator": {
            "permissions": ["workspaces.create"]
        },
        "Admin": {
            "includes": ["Editor"],
            "permissions": [
                "workspaces.delete", 
                "workspaces.members.list",
                "workspaces.members.create",
                "workspaces.members.update",
                "workspaces.members.delete",
                "iam.create",
                "iam.read"
            ]
        },
        "ServiceSystem": {
            "permissions": ["*"]
        }
    },
    "endpoints": {
        "/apis/entities/v2/workspaces": {
            "get": {
                "permissions": ["workspaces.list"],
                "scopes": ["platform:read"]
            },
            "post": {
                "permissions": ["workspaces.create"],
                "scopes": ["entities:write", "platform:write"]
            }
        },
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["platform:read"]
            },
            "put": {
                "permissions": ["workspaces.update"],
                "scopes": ["platform:write"]
            },
            "delete": {
                "permissions": ["workspaces.delete"],
                "scopes": ["platform:write"]
            }
        }
    },
    "workspaces": {
        "test-workspace": {}
    },
    "principals": {
        "user123": {
            "workspaces": {
                # Admin of test-workspace, plus a system Viewer grant. Listing workspaces now
                # requires workspaces.list in the SYSTEM workspace, so a system-level
                # role is what lets this user list.
                "test-workspace": ["Admin"],
                "system": ["Viewer"]
            }
        }
    }
}

# ============================================================================
# Test 1: List workspaces with no access
# ============================================================================

test_list_workspaces_without_system_permission_denied if {
    # Listing workspaces now requires workspaces.list in the SYSTEM workspace. A user
    # with no such grant is denied — previously this rule allowed any authenticated user.
    result := authz.allow
        with input as {
            "principal_id": "test-user",
            "principal_email": "test@example.com",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces",
            "scopes": ["platform:read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as workspace_access_test_data.workspaces
        with data.authz.principals as {
            "test-user": {"workspaces": {}}  # No system workspaces.list grant
        }

    result.allowed == false
}

# ============================================================================
# Test 2: Workspace creation
# ============================================================================

test_create_workspace_allowed if {
    # Workspace creation is authorized through a system-scoped role binding.
    result := authz.allow 
        with input as {
            "principal_id": "user123",
            "principal_email": "creator@example.com",
            "method": "POST",
            "path": "/apis/entities/v2/workspaces",
            "scopes": ["entities:write", "platform:write"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as {}  # Empty initially
        with data.authz.principals as {
            "*": {"workspaces": {"system": ["WorkspaceCreator"]}}
        }
    
    result.allowed == true
}

# ============================================================================
# Test 3: Service principal checks
# ============================================================================

test_service_principal_workspace_read if {
    # Service principal should be allowed to read workspace
    result := authz.has_permissions 
        with input as {
            "principal_id": "service:entity-store",
            "workspace": "test-workspace",
            "permissions": ["workspaces.read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.principals as workspace_access_test_data.principals
    
    result.allowed == true
}

test_service_principal_iam_create if {
    # Service principal should be allowed to create IAM resources
    result := authz.has_permissions 
        with input as {
            "principal_id": "service:entity-store",
            "workspace": "test-workspace",
            "permissions": ["iam.create"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.principals as workspace_access_test_data.principals
    
    result.allowed == true
}

# ============================================================================
# Test 4: Creator permission check after admin role granted
# ============================================================================

test_creator_workspace_update_permission if {
    # Creator with Admin role should have update permission
    result := authz.has_permissions 
        with input as {
            "principal_id": "user123",
            "workspace": "test-workspace",
            "permissions": ["workspaces.update"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.principals as workspace_access_test_data.principals
    
    result.allowed == true
}

# ============================================================================
# Test 5: Creator can access workspace
# ============================================================================

test_creator_get_workspace_middleware_allowed if {
    # Creator should be allowed at middleware level
    result := authz.allow 
        with input as {
            "principal_id": "user123",
            "principal_email": "creator@example.com",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces/test-workspace",
            "scopes": ["platform:read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as workspace_access_test_data.workspaces
        with data.authz.principals as workspace_access_test_data.principals
    
    result.allowed == true
}

test_creator_get_workspace_storage_permission if {
    # Creator should pass storage-layer permission check
    result := authz.has_permissions 
        with input as {
            "principal_id": "user123",
            "workspace": "test-workspace",
            "permissions": ["workspaces.read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.principals as workspace_access_test_data.principals
    
    result.allowed == true
}

# ============================================================================
# Test 6: Other user cannot access workspace
# ============================================================================

test_other_user_get_workspace_denied if {
    # User without access should be denied
    result := authz.allow 
        with input as {
            "principal_id": "user456",
            "principal_email": "other@example.com",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces/test-workspace",
            "scopes": ["platform:read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as workspace_access_test_data.workspaces
        with data.authz.principals as {
            "user456": {"workspaces": {}}  # No access
        }
    
    result.allowed == false
}

# ============================================================================
# Test 7: List workspaces with access
# ============================================================================

test_creator_list_workspaces_allowed if {
    # Creator should be allowed to list workspaces
    result := authz.allow 
        with input as {
            "principal_id": "user123",
            "principal_email": "creator@example.com",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces",
            "scopes": ["platform:read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as workspace_access_test_data.workspaces
        with data.authz.principals as workspace_access_test_data.principals
    
    result.allowed == true
}

# ============================================================================
# Test 8: Other user list workspaces
# ============================================================================

test_other_user_list_workspaces_denied if {
    # A user with only workspace-scoped (or no) roles lacks workspaces.list in the system
    # workspace, so they can no longer list workspaces.
    result := authz.allow
        with input as {
            "principal_id": "user456",
            "principal_email": "other@example.com",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces",
            "scopes": ["platform:read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as workspace_access_test_data.workspaces
        with data.authz.principals as {
            "user456": {"workspaces": {}}
        }

    result.allowed == false
}

# ============================================================================
# Additional tests: Missing scopes
# ============================================================================

test_list_workspaces_missing_scope if {
    # Request without proper scope should be denied
    result := authz.allow 
        with input as {
            "principal_id": "user123",
            "principal_email": "creator@example.com",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces",
            "scopes": ["models:read"]  # Wrong scope, needs platform:read
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as workspace_access_test_data.workspaces
        with data.authz.principals as workspace_access_test_data.principals
    
    result.allowed == false
}

# ============================================================================
# Additional tests: Multiple workspaces
# ============================================================================

test_list_multiple_workspaces if {
    # User with access to multiple workspaces should be allowed
    result := authz.allow 
        with input as {
            "principal_id": "multi-user",
            "principal_email": "multi@example.com",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces",
            "scopes": ["platform:read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as {
            "ns1": {},
            "ns2": {},
            "ns3": {}
        }
        with data.authz.principals as {
            "multi-user": {
                "workspaces": {
                    "ns1": ["Viewer"],
                    "ns2": ["Editor"],
                    "ns3": ["Admin"],
                    # system Viewer carries workspaces.list, the system-workspace grant that
                    # listing now requires; the multi-workspace setup is otherwise unchanged.
                    "system": ["Viewer"]
                }
            }
        }

    result.allowed == true
}

# ============================================================================
# Additional tests: Unauthenticated requests
# ============================================================================

test_list_workspaces_unauthenticated if {
    # Unauthenticated request (empty principal_id) should be denied
    result := authz.allow 
        with input as {
            "principal_id": "",
            "method": "GET",
            "path": "/apis/entities/v2/workspaces",
            "scopes": ["platform:read"]
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as workspace_access_test_data.workspaces
        with data.authz.principals as workspace_access_test_data.principals
    
    result.allowed == false
}

test_create_workspace_unauthenticated if {
    # Unauthenticated request should be denied
    result := authz.allow 
        with input as {
            "principal_id": "",
            "method": "POST",
            "path": "/apis/entities/v2/workspaces",
            "scopes": []
        }
        with data.authz.roles as workspace_access_test_data.roles
        with data.authz.endpoints as workspace_access_test_data.endpoints
        with data.authz.workspaces as {}
        with data.authz.principals as {}
    
    result.allowed == false
}
