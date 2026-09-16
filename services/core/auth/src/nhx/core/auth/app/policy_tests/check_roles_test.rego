# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if

# Test check_role with single role - user has it
test_check_role_single_allowed if {
    result := authz.has_role 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-1",
            "role": "Viewer"
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.has_role == true
}

# Test check_role with single role - user doesn't have it
test_check_role_single_denied if {
    result := authz.has_role 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-1",
            "role": "Editor"
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.has_role == false
}

# Test check_role - user has multiple roles, checking for one of them
test_check_role_user_has_multiple if {
    result := authz.has_role 
        with input as {
            "principal_id": "user2@example.com",
            "workspace": "workspace-1",
            "role": "Editor"
        }
        with data.authz.principals as {
            "user2@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer", "Editor", "Admin"]
                }
            }
        }
    
    result.has_role == true
}

# Test check_role - user has multiple roles, checking for one they don't have
test_check_role_user_missing_role if {
    result := authz.has_role 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-1",
            "role": "Admin"
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer", "Editor"]
                }
            }
        }
    
    result.has_role == false
}

# Test check_role in different workspace
test_check_role_different_workspace if {
    # User has Editor role in workspace-2
    result := authz.has_role 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-2",
            "role": "Editor"
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-2": ["Editor"]
                }
            }
        }
    
    result.has_role == true
}

# Test check_role with no roles in workspace
test_check_role_no_access if {
    result := authz.has_role 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-3",
            "role": "Viewer"
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.has_role == false
}

# Test check_role with unknown principal
test_check_role_unknown_principal if {
    result := authz.has_role 
        with input as {
            "principal_id": "unknown@example.com",
            "workspace": "workspace-1",
            "role": "Viewer"
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.has_role == false
}

# Test check_role with principal having role in wrong workspace
test_check_role_wrong_workspace if {
    result := authz.has_role 
        with input as {
            "principal_id": "user1@example.com",
            "workspace": "workspace-2",
            "role": "Viewer"
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer", "Editor"]
                }
            }
        }
    
    result.has_role == false
}

# Test check_role using principal_email
test_check_role_with_email if {
    result := authz.has_role 
        with input as {
            "principal_id": "user-id-123",
            "principal_email": "user1@example.com",
            "workspace": "workspace-1",
            "role": "Viewer"
        }
        with data.authz.principals as {
            "user1@example.com": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            }
        }
    
    result.has_role == true
}

# Test check_role using principal_groups
test_check_role_with_groups if {
    result := authz.has_role 
        with input as {
            "principal_id": "user-id-123",
            "principal_groups": ["group1", "group2"],
            "workspace": "workspace-1",
            "role": "Editor"
        }
        with data.authz.principals as {
            "group2": {
                "workspaces": {
                    "workspace-1": ["Editor", "Viewer"]
                }
            }
        }
    
    result.has_role == true
}

# Test check_role with all three - principal_id, principal_email, principal_groups
test_check_role_with_all_principals if {
    result := authz.has_role 
        with input as {
            "principal_id": "user-id-456",
            "principal_email": "user2@example.com",
            "principal_groups": ["group3"],
            "workspace": "workspace-1",
            "role": "Admin"
        }
        with data.authz.principals as {
            "user-id-456": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            },
            "user2@example.com": {
                "workspaces": {
                    "workspace-1": ["Editor"]
                }
            },
            "group3": {
                "workspaces": {
                    "workspace-1": ["Admin"]
                }
            }
        }
    
    # Should succeed because group3 has Admin role
    result.has_role == true
}

# Test check_role - user doesn't have role via any principal
test_check_role_no_principal_has_role if {
    result := authz.has_role 
        with input as {
            "principal_id": "user-id-789",
            "principal_email": "user3@example.com",
            "principal_groups": ["group4"],
            "workspace": "workspace-1",
            "role": "PlatformAdmin"
        }
        with data.authz.principals as {
            "user-id-789": {
                "workspaces": {
                    "workspace-1": ["Viewer"]
                }
            },
            "user3@example.com": {
                "workspaces": {
                    "workspace-1": ["Editor"]
                }
            },
            "group4": {
                "workspaces": {
                    "workspace-1": ["Admin"]
                }
            }
        }
    
    # None of the principals have PlatformAdmin role
    result.has_role == false
}


