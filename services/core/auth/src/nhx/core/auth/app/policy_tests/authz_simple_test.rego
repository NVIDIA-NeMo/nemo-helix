# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import data.common

# Simple test data with minimal setup
mock_authz_data := {
    "roles": {
        "Viewer": {
            "permissions": ["models.read", "models.list"]
        },
        "Editor": {
            "includes": ["Viewer"],
            "permissions": ["models.create", "models.update"]
        }
    },
    "endpoints": {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {"permissions": ["models.list"]},
            "post": {"permissions": ["models.create"]}
        },
        "/apis/models/v2/workspaces/{workspace}/models/{name}": {
            "get": {"permissions": ["models.read"]},
            "patch": {"permissions": ["models.update"]}
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
        "editor@test.com": {
            "workspaces": {"test-ns": ["Editor"]}
        }
    }
}

# Test 1: Basic viewer permissions
test_viewer_can_read_models if {
    authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_authz_data.roles
    with data.authz.endpoints as mock_authz_data.endpoints
    with data.authz.workspaces as mock_authz_data.workspaces
    with data.authz.principals as mock_authz_data.principals
}

# Test 2: Health endpoint (simpler test)
test_health_endpoint if {
    authz.allow with input as {
        "principal_id": "",
        "method": "GET",
        "path": "/health"
    }
}

# Debug test - check if direct role permissions work (no inheritance)
test_direct_role_permissions if {
    role := mock_authz_data.roles["Viewer"]
    perms := role.permissions
    "models.read" in perms
}

# Debug test - check if principal has role
test_principal_has_role if {
    roles := mock_authz_data.principals["viewer@test.com"].workspaces["test-ns"]
    "Viewer" in roles
}

# Debug test - check endpoint normalization
test_endpoint_normalization if {
    normalized := common.normalize_endpoint("/apis/models/v2/workspaces/test-ns/models/model-123")
        with data.authz.endpoints as mock_authz_data.endpoints
    normalized == "/apis/models/v2/workspaces/{workspace}/models/{name}"
}

# Debug test - check if required permissions are found
test_required_permissions if {
    perms := common.get_required_permissions("/apis/models/v2/workspaces/-/models", "GET")
        with data.authz.endpoints as mock_authz_data.endpoints
    "models.list" in perms
}

# Minimal authorization test - check the core logic
test_minimal_auth if {
    # Check if has_permission works with simple data
    common.has_permissions("viewer@test.com", "test-ns", ["models.read"])
        with input as {"principal_id": "viewer@test.com"}
        with data.authz.roles as mock_authz_data.roles
        with data.authz.principals as mock_authz_data.principals
}

# Test allow for LIST operations
test_allow_list_operation if {
    result := authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/-/models"
    }
    with data.authz.roles as mock_authz_data.roles
    with data.authz.endpoints as mock_authz_data.endpoints
    with data.authz.workspaces as mock_authz_data.workspaces
    with data.authz.principals as mock_authz_data.principals

    result.allowed == true
}

# Debug test for list operation permissions
test_debug_list_permissions if {
    # Check if required permissions are found for list
    perms := common.get_required_permissions("/apis/models/v2/workspaces/-/models", "GET")
        with data.authz.endpoints as mock_authz_data.endpoints
    perms == ["models.list"]

    # Check if viewer has models.list permission
    viewer_perms := common.get_role_permissions("Viewer")
        with data.authz.roles as mock_authz_data.roles
    "models.list" in viewer_perms

    # Check if has_permission works for list
    common.has_permissions("viewer@test.com", "test-ns", ["models.list"])
        with input as {"principal_id": "viewer@test.com"}
        with data.authz.roles as mock_authz_data.roles
        with data.authz.principals as mock_authz_data.principals
}

# Test allow for non-LIST operations
test_allow_single_resource if {
    result := authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as mock_authz_data.roles
    with data.authz.endpoints as mock_authz_data.endpoints
    with data.authz.workspaces as mock_authz_data.workspaces
    with data.authz.principals as mock_authz_data.principals

    result.allowed == true
}

# Test allow for denied access
test_allow_denied if {
    result := authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "POST",
        "path": "/apis/models/v2/workspaces/test-ns/models"
    }
    with data.authz.roles as mock_authz_data.roles
    with data.authz.endpoints as mock_authz_data.endpoints
    with data.authz.workspaces as mock_authz_data.workspaces
    with data.authz.principals as mock_authz_data.principals

    # Should be denied because viewer doesn't have models.create permission
    result.allowed == false
    result.headers["X-NHX-Authorized"] == "false"
}

# Test allow with public workspace access
test_allow_public_workspace if {
    result := authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/-/models"
    }
    with data.authz.roles as mock_authz_data.roles
    with data.authz.endpoints as mock_authz_data.endpoints
    with data.authz.workspaces as mock_authz_data.workspaces
    with data.authz.principals as mock_authz_data.principals

    result.allowed == true
}
