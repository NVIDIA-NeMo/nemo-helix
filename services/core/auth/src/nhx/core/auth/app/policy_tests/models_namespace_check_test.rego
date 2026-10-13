# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Test data for models endpoints (aligned with static-authz)
models_test_data := {
    "roles": {
        "Viewer": {
            "permissions": ["models.read", "models.list"]
        },
        "Editor": {
            "includes": ["Viewer"],
            "permissions": ["models.create", "models.delete"]
        }
    },
    "endpoints": {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read"]
            },
            "post": {
                "permissions": ["models.create"],
                "scopes": ["models:write"]
            }
        },
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
    },
    "workspaces": {
        "team-ml": {},
        "public-models": {}
    },
    "principals": {
        "user@example.com": {
            "workspaces": {
                "team-ml": ["Editor"]
            }
        },
        # Wildcard principal grants Viewer access to public-models for all authenticated users
        "*": {
            "workspaces": {
                "public-models": ["Viewer"]
            }
        }
    }
}

# Test 1: GET model with workspace in path
test_model_get_with_workspace_no_check_header if {
    result := authz.allow with input as {
        "principal_id": "user@example.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-ml/models/model-123",
        "scopes": ["models:read"]
    }
    with data.authz.roles as models_test_data.roles
    with data.authz.endpoints as models_test_data.endpoints
    with data.authz.workspaces as models_test_data.workspaces
    with data.authz.principals as models_test_data.principals

    result.allowed == true
    result.headers["X-NHX-Authorized"] == "true"
}

# Test 2: POST create model (collection)
test_model_put_with_workspace_no_check_header if {
    result := authz.allow with input as {
        "principal_id": "user@example.com",
        "method": "POST",
        "path": "/apis/models/v2/workspaces/team-ml/models",
        "scopes": ["models:write"]
    }
    with data.authz.roles as models_test_data.roles
    with data.authz.endpoints as models_test_data.endpoints
    with data.authz.workspaces as models_test_data.workspaces
    with data.authz.principals as models_test_data.principals

    result.allowed == true
    result.headers["X-NHX-Authorized"] == "true"
}

# Test 3: DELETE model with workspace in path
test_model_delete_with_workspace_no_check_header if {
    result := authz.allow with input as {
        "principal_id": "user@example.com",
        "method": "DELETE",
        "path": "/apis/models/v2/workspaces/team-ml/models/model-123",
        "scopes": ["models:write"]
    }
    with data.authz.roles as models_test_data.roles
    with data.authz.endpoints as models_test_data.endpoints
    with data.authz.workspaces as models_test_data.workspaces
    with data.authz.principals as models_test_data.principals

    result.allowed == true
    result.headers["X-NHX-Authorized"] == "true"
}

# Test 4: GET model list for a workspace
test_model_list_without_workspace_has_check_header if {
    result := authz.allow with input as {
        "principal_id": "user@example.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/team-ml/models",
        "scopes": ["models:read"]
    }
    with data.authz.roles as models_test_data.roles
    with data.authz.endpoints as models_test_data.endpoints
    with data.authz.workspaces as models_test_data.workspaces
    with data.authz.principals as models_test_data.principals

    result.allowed == true
}

# Test 5: Verify workspace extraction from model path
test_workspace_extraction_from_model_path if {
    workspace := authz.extract_workspace_from_path("/apis/models/v2/workspaces/team-ml/models/model-123") with data.authz.endpoints as models_test_data.endpoints

    workspace == "team-ml"
}

# Test 6: Access denied to model in workspace without permission
test_model_access_denied_without_permission if {
    result := authz.allow with input as {
        "principal_id": "user@example.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/other-team/models/model-123",
        "scopes": ["models:read"]
    }
    with data.authz.roles as models_test_data.roles
    with data.authz.endpoints as models_test_data.endpoints
    with data.authz.workspaces as {
        "team-ml": {},
        "other-team": {}
    }
    with data.authz.principals as models_test_data.principals

    result.allowed == false
    result.headers["X-NHX-Authorized"] == "false"
}

# Test 7: Access to public workspace model
test_model_access_public_workspace if {
    result := authz.allow with input as {
        "principal_id": "user@example.com",
        "method": "GET",
        "path": "/apis/models/v2/workspaces/public-models/models/model-456",
        "scopes": ["models:read"]
    }
    with data.authz.roles as models_test_data.roles
    with data.authz.endpoints as models_test_data.endpoints
    with data.authz.workspaces as models_test_data.workspaces
    with data.authz.principals as models_test_data.principals

    result.allowed == true
    result.headers["X-NHX-Authorized"] == "true"
}
