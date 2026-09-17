# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import future.keywords.if

# Test get_required_scopes

test_get_required_scopes_for_defined_endpoint if {
    scopes := get_required_scopes("/apis/models/v2/workspaces/-/models", "GET") with data.authz.endpoints as {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read", "platform:read"]
            }
        }
    }
    scopes == ["models:read", "platform:read"]
}

test_get_required_scopes_for_undefined_endpoint if {
    scopes := get_required_scopes("/apis/not-configured/v1/undefined", "GET") with data.authz.endpoints as {}
    scopes == []
}

# Test has_required_scopes

# Test when no scopes are provided (scope check is optional)
test_has_required_scopes_no_scopes_provided if {
    has_required_scopes("/apis/models/v2/workspaces/-/models", "GET", null)
}

# Test when scopes are provided but empty and no scopes required
test_has_required_scopes_empty_array_no_required if {
    has_required_scopes("/apis/not-configured/v1/undefined", "GET", [])
}

# Test when scopes are provided but empty and scopes are required
test_has_required_scopes_empty_array_with_required if {
    not has_required_scopes("/apis/models/v2/workspaces/-/models", "GET", []) with data.authz.endpoints as {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read", "platform:read"]
            }
        }
    }
}

# Test when no scopes required for endpoint
test_has_required_scopes_no_scopes_required if {
    has_required_scopes("/apis/entities/v2/workspaces", "POST", ["models:read"]) with data.authz.endpoints as {
        "/apis/entities/v2/workspaces": {
            "post": {"permissions": [], "scopes": []}
        }
    }
}

# Test when user has one of the required scopes
test_has_required_scopes_has_one_scope if {
    has_required_scopes("/apis/models/v2/workspaces/-/models", "GET", ["models:read"]) with data.authz.endpoints as {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read", "platform:read"]
            }
        }
    }
}

# Test when user has one of the required scopes (second option)
test_has_required_scopes_has_second_scope if {
    has_required_scopes("/apis/models/v2/workspaces/-/models", "GET", ["platform:read"]) with data.authz.endpoints as {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read", "platform:read"]
            }
        }
    }
}

# Test when user has all required scopes
test_has_required_scopes_has_all_scopes if {
    has_required_scopes("/apis/models/v2/workspaces/-/models", "GET", ["models:read", "platform:read"]) with data.authz.endpoints as {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read", "platform:read"]
            }
        }
    }
}

# Test when user has extra scopes
test_has_required_scopes_has_extra_scopes if {
    has_required_scopes("/apis/models/v2/workspaces/-/models", "GET", ["models:read", "platform:read", "audit:read"]) with data.authz.endpoints as {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read", "platform:read"]
            }
        }
    }
}

# Test when user doesn't have required scopes
test_has_required_scopes_missing_scopes if {
    not has_required_scopes("/apis/models/v2/workspaces/-/models", "GET", ["audit:read"]) with data.authz.endpoints as {
        "/apis/models/v2/workspaces/{workspace}/models": {
            "get": {
                "permissions": ["models.list"],
                "scopes": ["models:read", "platform:read"]
            }
        }
    }
}

# Test authorization with scopes

test_allow_with_valid_scopes if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": ["entities:read"]
    }
    with data.authz.principals as {"user1": {"workspaces": {"ns1": ["Viewer"]}}}
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }

    result.allowed == true
}

test_allow_fails_with_invalid_scopes if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": ["models:read"]
    }
    with data.authz.principals as {"user1": {"workspaces": {"ns1": ["Viewer"]}}}
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }

    result.allowed == false
}

test_allow_with_no_scopes_provided if {
    # Test that when no scopes are provided, authorization works based on permissions alone
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1"
    }
    with data.authz.principals as {"user1": {"workspaces": {"ns1": ["Viewer"]}}}
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }

    result.allowed == true
}

test_allow_with_empty_scopes if {
    # Empty scopes array now skips scope checking (same as no scopes provided)
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": []
    }
    with data.authz.principals as {"user1": {"workspaces": {"ns1": ["Viewer"]}}}
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }

    result.allowed == true
}

# Test LIST operations with scopes

test_list_allow_with_valid_scopes if {
    # Valid scopes AND workspaces.list in the system workspace together allow listing.
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces",
        "scopes": ["entities:read"]
    }
    with data.authz.principals as {"user1": {"workspaces": {"system": ["Viewer"]}}}
    with data.authz.roles as {"Viewer": {"permissions": ["workspaces.list"]}}
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces": {
            "get": {
                "permissions": ["workspaces.list"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }

    result.allowed == true
}

test_list_allow_fails_with_invalid_scopes if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces",
        "scopes": ["models:read"]
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces": {
            "get": {
                "permissions": ["workspaces.list"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }

    result.allowed == false
}

# Test wildcard principal with scopes

test_public_workspace_with_valid_scopes if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/public-ns",
        "scopes": ["entities:read"]
    }
    with data.authz.workspaces as {
        "public-ns": {}
    }
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }
    with data.authz.principals as {
        # Wildcard principal grants Viewer access to public-ns
        "*": {
            "workspaces": {
                "public-ns": ["Viewer"]
            }
        }
    }

    result.allowed == true
}

test_public_workspace_fails_with_invalid_scopes if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/public-ns",
        "scopes": ["models:read"]
    }
    with data.authz.workspaces as {
        "public-ns": {}
    }
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }
    with data.authz.principals as {
        "*": {
            "workspaces": {
                "public-ns": ["Viewer"]
            }
        }
    }

    result.allowed == false
}

# Test platform admin with scopes (platform admin should have access regardless of scopes)

test_platform_admin_bypasses_scope_check if {
    result := allow with input as {
        "principal_id": "admin",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": []
    }
    with data.authz.principals as {"admin": {"workspaces": {"system": ["PlatformAdmin"]}}}

    result.allowed == true
}

# Test OIDC scope handling

# Test that OIDC scopes without ":" are ignored (treated as no platform scopes)
test_oidc_scopes_ignored if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": ["openid", "profile", "email", "offline_access"]
    }
    with data.authz.principals as {"user1": {"workspaces": {"ns1": ["Viewer"]}}}
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }

    result.allowed == true
}

# Test that mixing OIDC scopes with platform scopes works
test_mixed_oidc_and_platform_scopes if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": ["openid", "profile", "entities:read", "email"]
    }
    with data.authz.principals as {"user1": {"workspaces": {"ns1": ["Viewer"]}}}
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read", "models:read"]
            }
        }
    }

    result.allowed == true
}

# Test that platform scopes are still enforced when mixed with OIDC scopes
test_mixed_scopes_platform_scope_enforced if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": ["openid", "profile", "wrong:scope", "email"]
    }
    with data.authz.principals as {"user1": {"workspaces": {"ns1": ["Viewer"]}}}
    with data.authz.roles as {
        "Viewer": {
            "permissions": ["workspaces.read", "workspaces.list"]
        }
    }
    with data.authz.endpoints as {
        "/apis/entities/v2/workspaces/{name}": {
            "get": {
                "permissions": ["workspaces.read"],
                "scopes": ["entities:read", "platform:read"]
            }
        }
    }

    result.allowed == false
}

# Test scoped Access Key self-escalation via the access-keys endpoints.
# Uses the real bundled static-authz.yaml config, so a future edit that drops the
# `auth`/`platform` scopes from these endpoints fails this suite too, not just a mocked one.

test_scoped_access_key_cannot_create_access_keys if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "POST",
        "path": "/apis/auth/v2/access-keys",
        "scopes": ["intake:read", "intake:write"]
    }

    result.allowed == false
}

test_scoped_access_key_cannot_list_access_keys if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "GET",
        "path": "/apis/auth/v2/access-keys",
        "scopes": ["intake:read"]
    }

    result.allowed == false
}

test_scoped_access_key_cannot_revoke_access_keys if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "DELETE",
        "path": "/apis/auth/v2/access-keys/ak_00000000000000000000000000000000",
        "scopes": ["intake:read", "intake:write"]
    }

    result.allowed == false
}

test_scoped_access_key_cannot_rotate_access_keys if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "POST",
        "path": "/apis/auth/v2/access-keys/ak_00000000000000000000000000000000/rotate",
        "scopes": ["intake:read", "intake:write"]
    }

    result.allowed == false
}

test_access_key_scoped_to_auth_can_create_access_keys if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "POST",
        "path": "/apis/auth/v2/access-keys",
        "scopes": ["auth:write"]
    }

    result.allowed == true
}

test_access_key_scoped_to_platform_can_create_access_keys if {
    result := allow with input as {
        "principal_id": "user1",
        "method": "POST",
        "path": "/apis/auth/v2/access-keys",
        "scopes": ["platform:write"]
    }

    result.allowed == true
}

test_unscoped_access_key_can_create_access_keys if {
    # No `scope` claim at all skips scope checking; authorized on permissions/ownership alone.
    result := allow with input as {
        "principal_id": "user1",
        "method": "POST",
        "path": "/apis/auth/v2/access-keys"
    }

    result.allowed == true
}

# Test that a PlatformAdmin's own Scoped Access Key scope restriction is honored.
# Uses the real bundled static-authz.yaml config, so a future edit that drops
# `scope_check_passed` from the bypass rule fails this suite too, not just a mocked one.

test_platform_admin_scoped_access_key_denied_outside_its_scope if {
    result := allow with input as {
        "principal_id": "admin1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": ["intake:read", "intake:write"]
    }
    with data.authz.principals as {"admin1": {"workspaces": {"system": ["PlatformAdmin"]}}}

    result.allowed == false
}

test_platform_admin_scoped_access_key_allowed_within_its_scope if {
    result := allow with input as {
        "principal_id": "admin1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1",
        "scopes": ["entities:read"]
    }
    with data.authz.principals as {"admin1": {"workspaces": {"system": ["PlatformAdmin"]}}}

    result.allowed == true
}

test_platform_admin_without_platform_scope_claims_keeps_full_bypass if {
    # No `scope` claim at all: the platform admin bypass still applies unrestricted.
    result := allow with input as {
        "principal_id": "admin1",
        "method": "GET",
        "path": "/apis/entities/v2/workspaces/ns1"
    }
    with data.authz.principals as {"admin1": {"workspaces": {"system": ["PlatformAdmin"]}}}

    result.allowed == true
}
