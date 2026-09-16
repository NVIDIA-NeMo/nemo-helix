# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if

# Test has_permissions with single permission - user has it
test_has_permissions_single_allowed if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-1",
            "permissions": ["models.read"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.allowed == true
}

# Test has_permissions with single permission - user doesn't have it
test_has_permissions_single_denied if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-1",
            "permissions": ["models.create"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.allowed == false
}

# Test has_permissions with multiple permissions - user has all
test_has_permissions_multiple_allowed if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user2@example.com",
            "workspace": "workspace-1",
            "permissions": ["models.read", "models.create", "models.update"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            },
            "Editor": {
                "permissions": ["models.create", "models.update", "models.delete"],
                "includes": ["Viewer"]
            }
        }
        with data.authz.principals as {
            "user2@example.com": {
                "workspaces": {
                    "workspace-1": ["Editor"]
                }
            }
        }
    
    result.allowed == true
}

# Test has_permissions with multiple permissions - user missing one
test_has_permissions_multiple_denied if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-1",
            "permissions": ["models.read", "models.create"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.allowed == false
}

# Test has_permissions in different workspace
test_has_permissions_different_workspace if {
    # User has Editor role in workspace-2 (includes Viewer permissions)
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-2",
            "permissions": ["models.create", "models.read"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            },
            "Editor": {
                "permissions": ["models.create", "models.update", "models.delete"],
                "includes": ["Viewer"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-2": ["Editor"]
                }
            }
        }
    
    result.allowed == true
}

# Test has_permissions with no permissions in workspace
test_has_permissions_no_access if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-3",
            "permissions": ["models.read"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.allowed == false
}

# Test has_permissions with unknown principal
test_has_permissions_unknown_principal if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "unknown@example.com",
            "workspace": "workspace-1",
            "permissions": ["models.read"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.allowed == false
}

# Test has_permissions with empty permissions list
test_has_permissions_empty_permissions if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-1",
            "permissions": []
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    # Empty permissions should be allowed (all zero permissions are satisfied)
    result.allowed == true
}

# Test has_permissions with service principal - ServiceSystem role (wildcard permission)
test_has_permissions_service_principal if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "service:entity-store",
            "workspace": "workspace-1",
            "permissions": ["models.read", "models.create", "models.delete"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read"]
            },
            "ServiceSystem": {
                "permissions": ["*"]
            }
        }
        with data.authz.principals as {
            # Service principal not in principals list — defaults to ServiceSystem
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.allowed == true
}

# Test has_permissions with service principal - ServiceSystem must exist in role definitions
test_has_permissions_service_principal_no_data if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "service:my-service",
            "workspace": "any-workspace",
            "permissions": ["any.permission"]
        }
        with data.authz.roles as {
            "ServiceSystem": {"permissions": ["*"]}
        }
        with data.authz.principals as {}
    
    result.allowed == true
}

# Test has_permissions with wildcard principal - user has no explicit role but should be allowed via "*" principal
test_has_permissions_public_workspace_viewer_permissions if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "public-workspace",
            "permissions": ["models.read"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "other-workspace": ["Viewer"]
                }
            },
            # Wildcard principal grants Viewer to all authenticated users
            "*": {
                "workspaces": {
                    "public-workspace": ["Viewer"]
                }
            }
        }
        with data.authz.workspaces as {
            "public-workspace": {}
        }
    
    # Wildcard principal with viewer permissions should be allowed
    result.allowed == true
}

# Test has_permissions with wildcard principal - multiple viewer permissions
test_has_permissions_public_workspace_multiple_viewer_permissions if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "public-workspace",
            "permissions": ["models.read", "models.list"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list", "evaluations.read"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "other-workspace": ["Editor"]
                }
            },
            # Wildcard principal grants Viewer to all authenticated users
            "*": {
                "workspaces": {
                    "public-workspace": ["Viewer"]
                }
            }
        }
        with data.authz.workspaces as {
            "public-workspace": {}
        }
    
    # Wildcard principal with multiple viewer permissions should be allowed
    result.allowed == true
}

# Test has_permissions with wildcard principal - denied for write permissions
test_has_permissions_public_workspace_denied_write if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "public-workspace",
            "permissions": ["models.create"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            },
            "Editor": {
                "permissions": ["models.create", "models.update", "models.delete"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "other-workspace": ["Editor"]
                }
            },
            # Wildcard principal only has Viewer, not Editor
            "*": {
                "workspaces": {
                    "public-workspace": ["Viewer"]
                }
            }
        }
        with data.authz.workspaces as {
            "public-workspace": {}
        }
    
    # Wildcard principal should deny write permissions (only has Viewer)
    result.allowed == false
}

# Test has_permissions with public workspace - denied for private workspace
test_has_permissions_private_workspace_no_access if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "private-workspace",
            "permissions": ["models.read"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "other-workspace": ["Viewer"]
                }
            }
        }
        with data.authz.workspaces as {
            "private-workspace": {}
        }
    
    # Private workspace should deny access without explicit role
    result.allowed == false
}

# Test has_permissions with public workspace - unauthenticated principal denied
test_has_permissions_public_workspace_unauthenticated if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "",
            "workspace": "public-workspace",
            "permissions": ["models.read"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            }
        }
        with data.authz.principals as {}
        with data.authz.workspaces as {
            "public-workspace": {}
        }
    
    # Public workspace still requires authentication
    result.allowed == false
}

# Test has_permissions with public workspace - explicit role takes precedence
test_has_permissions_public_workspace_explicit_role if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "public-workspace",
            "permissions": ["models.create"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read", "models.list"]
            },
            "Editor": {
                "permissions": ["models.create", "models.update", "models.delete"],
                "includes": ["Viewer"]
            }
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "public-workspace": ["Editor"]
                }
            }
        }
        with data.authz.workspaces as {
            "public-workspace": {}
        }
    
    # User with explicit Editor role in public workspace should have write access
    result.allowed == true
}

# Test has_permissions with platform admin - should always be allowed regardless of permissions
test_has_permissions_platform_admin_allowed if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "admin@example.com",
            "workspace": "any-workspace",
            "permissions": ["models.create", "models.delete", "any.permission"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read"]
            }
        }
        with data.authz.principals as {
            "admin@example.com": {
                "workspaces": {
                    "system": ["PlatformAdmin"],
                    "other-workspace": ["Viewer"]
                }
            }
        }
    
    # Platform admin should have access to everything
    result.allowed == true
}

# Test has_permissions with platform admin - works in any workspace
test_has_permissions_platform_admin_any_workspace if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "admin@example.com",
            "workspace": "random-workspace",
            "permissions": ["write.permission", "delete.permission"]
        }
        with data.authz.roles as {}
        with data.authz.principals as {
            "admin@example.com": {
                "workspaces": {
                    "system": ["PlatformAdmin"]
                }
            }
        }
    
    # Platform admin bypasses all permission checks
    result.allowed == true
}

# Test has_permissions with platform admin - empty permissions
test_has_permissions_platform_admin_empty_permissions if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "admin@example.com",
            "workspace": "any-workspace",
            "permissions": []
        }
        with data.authz.principals as {
            "admin@example.com": {
                "workspaces": {
                    "system": ["PlatformAdmin"]
                }
            }
        }
    
    # Platform admin should be allowed even with empty permissions
    result.allowed == true
}

# Test has_permissions with regular user - not platform admin
test_has_permissions_not_platform_admin if {
    result := authz.has_permissions 
        with input as {
            "principal_id": "user@example.com",
            "workspace": "workspace-1",
            "permissions": ["models.create"]
        }
        with data.authz.roles as {
            "Viewer": {
                "permissions": ["models.read"]
            }
        }
        with data.authz.principals as {
            "user@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    # Regular user without required permissions should be denied
    result.allowed == false
}
