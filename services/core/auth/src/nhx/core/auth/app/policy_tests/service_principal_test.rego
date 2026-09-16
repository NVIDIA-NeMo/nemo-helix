# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import data.authz.allow
import future.keywords.if
import future.keywords.in

# Mock data for non-service principal test
mock_roles := {
    "Viewer": {
        "permissions": ["filesets.read"]
    }
}

mock_endpoints := {
    "/apis/files/v2/workspaces/{workspace}/filesets/{name}": {
        "get": {"permissions": ["filesets.read"]}
    }
}

mock_workspaces := {
    "e2ens": {}
}

mock_principals := {}

# Test service principal bypass for fileset access
# This test reproduces the exact failure from production logs
test_service_customizer_dataset_access if {
    result := allow with input as {
        "method": "GET",
        "path": "/apis/files/v2/workspaces/e2ens/filesets/e2eds",
        "principal_id": "service:customizer"
    }
    
    result.allowed == true
}

# Test service principal bypass - should allow all operations
test_service_principal_bypass_read if {
    result := allow with input as {
        "method": "GET",
        "path": "/apis/models/v2/workspaces/test-ns/models/test-model",
        "principal_id": "service:customizer"
    }
    
    result.allowed == true
}

# Test service principal bypass - should allow writes too
test_service_principal_bypass_write if {
    result := allow with input as {
        "method": "POST",
        "path": "/apis/models/v2/workspaces/test-ns/models",
        "principal_id": "service:entity-store"
    }
    
    result.allowed == true
}

# Test service principal bypass - different service
test_service_jobs_controller if {
    result := allow with input as {
        "method": "GET",
        "path": "/apis/jobs/v2/workspaces/-/jobs/run-1/steps/step-1",
        "principal_id": "service:jobs-controller"
    }
    
    result.allowed == true
}

# Test that non-service principals are denied without permissions
test_non_service_principal_denied if {
    result := allow 
        with input as {
            "method": "GET",
            "path": "/apis/files/v2/workspaces/e2ens/filesets/e2eds",
            "principal_id": "user-without-access"
        }
        with data.authz.roles as mock_roles
        with data.authz.endpoints as mock_endpoints
        with data.authz.workspaces as mock_workspaces
        with data.authz.principals as mock_principals
    
    result.allowed == false
}

# Test service principal with empty credentials
test_service_principal_no_workspace if {
    result := allow with input as {
        "method": "GET",
        "path": "/apis/files/v2/workspaces/ws/filesets",
        "principal_id": "service:customizer"
    }
    
    result.allowed == true
}

# Test service principal can access secret values (delegation pattern)
test_service_principal_can_access_secret_values if {
    result := allow with input as {
        "method": "GET",
        "path": "/apis/secrets/v2/workspaces/workspace1/secrets/my-secret/access",
        "principal_id": "service:secrets"
    }
    
    result.allowed == true
}

# Test service principal can access secret values with different service name
test_service_principal_entity_store_can_access_secret_values if {
    result := allow with input as {
        "method": "GET",
        "path": "/apis/secrets/v2/workspaces/workspace1/secrets/api-key/access",
        "principal_id": "service:entity-store"
    }
    
    result.allowed == true
}
