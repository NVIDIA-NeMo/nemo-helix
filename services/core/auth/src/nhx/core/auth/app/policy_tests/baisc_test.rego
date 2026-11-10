# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Simple test data with minimal setup
basic_mock_authz_data := {
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
            "get": {"direct": {"actor_perms": ["models.list"]}, "delegated": {"actor_perms": [], "subject_perms": ["models.list"]}},
            "post": {"direct": {"actor_perms": ["models.create"]}, "delegated": {"actor_perms": [], "subject_perms": ["models.create"]}}
        },
        "/apis/models/v2/workspaces/{workspace}/models/{name}": {
            "get": {"direct": {"actor_perms": ["models.read"]}, "delegated": {"actor_perms": [], "subject_perms": ["models.read"]}},
            "patch": {"direct": {"actor_perms": ["models.update"]}, "delegated": {"actor_perms": [], "subject_perms": ["models.update"]}}
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
test_basic_viewer_can_read_models if {
    result := data.authz.allow with input as {
        "principal_id": "viewer@test.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/test-ns/models/123"
    }
    with data.authz.roles as basic_mock_authz_data.roles
    with data.authz.endpoints as basic_mock_authz_data.endpoints
    with data.authz.workspaces as basic_mock_authz_data.workspaces
    with data.authz.principals as basic_mock_authz_data.principals

    result.allowed == true
}
