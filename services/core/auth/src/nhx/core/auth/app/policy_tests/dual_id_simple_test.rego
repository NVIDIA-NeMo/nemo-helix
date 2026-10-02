# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Simple tests to verify dual-identifier (principal ID + email) authorization works

mock_roles := {
    "Editor": {
        "permissions": ["models.read", "models.create"]
    }
}

mock_endpoints := {
    "/apis/models/v2/workspaces/{workspace}/models/{name}": {
        "get": {"permissions": ["models.read"], "scopes": []}
    }
}

mock_workspaces := {
    "team-alpha": {}
}

# Test 1: Access granted when permissions are on email and principal ID differs
test_dual_id_email_permissions if {
    result := authz.allow with input as {
        "principal_id": "user-uuid-456",
        "principal_email": "alice@company.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-alpha/models/model-1"
    }
    with data.authz.roles as mock_roles
    with data.authz.endpoints as mock_endpoints
    with data.authz.workspaces as mock_workspaces
    with data.authz.principals as {
        "alice@company.com": {
            "workspaces": {
                "team-alpha": ["Editor"]
            }
        }
    }

    result.allowed == true
}

# Test 2: Access denied when neither principal ID nor email has permissions
test_dual_id_no_permissions if {
    result := authz.allow with input as {
        "principal_id": "user-uuid-789",
        "principal_email": "bob@company.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-alpha/models/model-1"
    }
    with data.authz.roles as mock_roles
    with data.authz.endpoints as mock_endpoints
    with data.authz.workspaces as mock_workspaces
    with data.authz.principals as {
        "alice@company.com": {
            "workspaces": {
                "team-alpha": ["Editor"]
            }
        }
    }

    result.allowed == false
}
