# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import data.common

# Verify that unknown endpoints (not in static-authz.yaml) are denied by default.
# Previously, get_required_permissions returned [] for unknown endpoints,
# which triggered the "no permissions required" allow rule — granting access
# to any authenticated user (fail-open). After the fix, unknown endpoints are
# denied because normalize_endpoint is undefined for unrecognized paths,
# causing the allow rules to fail.

unknown_endpoint_test_data := {
	"roles": {
		"Viewer": {
			"permissions": ["entities.read", "workspaces.list"],
		},
		"Editor": {
			"includes": ["Viewer"],
			"permissions": ["entities.create", "entities.update", "workspaces.update"],
		},
		"WorkspaceCreator": {
			"permissions": ["workspaces.create"],
		},
	},
	"endpoints": {
		"/apis/entities/v2/workspaces": {
			"get": {"permissions": ["workspaces.list"]},
			"post": {
				"permissions": ["workspaces.create"],
				"scopes": ["entities:write", "platform:write"],
			},
		},
		"/apis/entities/v2/workspaces/{name}": {
			"get": {"permissions": ["workspaces.read"]},
			"put": {"permissions": ["workspaces.update"]},
			"delete": {"permissions": ["workspaces.delete"]},
		},
		"/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}": {
			"get": {"permissions": ["entities.read"]},
			"post": {"permissions": ["entities.create"]},
		},
		"/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}": {
			"get": {"permissions": ["entities.read"]},
			"put": {"permissions": ["entities.update"]},
			"delete": {"permissions": ["entities.delete"]},
		},
	},
	"workspaces": {
		"test-ns": {},
	},
	"principals": {
		"user@test.com": {
			"workspaces": {"test-ns": ["Viewer"]},
		},
		"editor@test.com": {
			"workspaces": {"test-ns": ["Editor"]},
		},
		"admin@test.com": {
			"workspaces": {"system": ["PlatformAdmin"]},
		},
	},
}

# --- Fail-closed: unknown endpoints are denied ---

# An authenticated user hitting an endpoint that does not exist in the config
# must be denied. This is the core regression test for the fail-open bug.
test_unknown_endpoint_denied_for_authenticated_user if {
	result := authz.allow with input as {
		"principal_id": "user@test.com",
		"method": "GET",
		"path": "/apis/entities/v2/workspaces/test-ns/unknown-resource",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as unknown_endpoint_test_data.principals

	result.allowed == false
}

# POST to an unknown endpoint must also be denied.
test_unknown_endpoint_denied_post if {
	result := authz.allow with input as {
		"principal_id": "editor@test.com",
		"method": "POST",
		"path": "/apis/entities/v2/workspaces/test-ns/unknown-resource",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as unknown_endpoint_test_data.principals

	result.allowed == false
}

# A path that *looks like* a real API but belongs to an unconfigured service must be denied.
test_unknown_api_group_denied if {
	result := authz.allow with input as {
		"principal_id": "user@test.com",
		"method": "GET",
		"path": "/apis/unknown-service/v2/workspaces/test-ns/things",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as unknown_endpoint_test_data.principals

	result.allowed == false
}

# GET to an unknown cross-workspace-style path (no workspace placeholder) must be denied.
# Previously this would match the cross-workspace LIST rule because
# extract_workspace_from_path was undefined (same as "no workspace placeholder").
test_unknown_endpoint_denied_cross_workspace_list if {
	result := authz.allow with input as {
		"principal_id": "user@test.com",
		"method": "GET",
		"path": "/apis/entities/v2/unknown-collection",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as unknown_endpoint_test_data.principals

	result.allowed == false
}

# HEAD must match GET: no bypass when the path is not a known endpoint pattern.
test_unknown_endpoint_denied_cross_workspace_list_head if {
	result := authz.allow with input as {
		"principal_id": "user@test.com",
		"method": "HEAD",
		"path": "/apis/entities/v2/unknown-collection",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as unknown_endpoint_test_data.principals

	result.allowed == false
}

# --- Existing behavior preserved: configured endpoints still work ---

# Workspace creation still works when the caller has a system-scoped creator binding.
test_workspace_creation_allowed_with_creator_binding if {
	result := authz.allow with input as {
		"principal_id": "user@test.com",
		"method": "POST",
		"path": "/apis/entities/v2/workspaces",
		"scopes": ["entities:write", "platform:write"],
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as {
			"*": {"workspaces": {"system": ["WorkspaceCreator"]}},
		}

	result.allowed == true
}

# Workspace creation must still enforce the endpoint's write scopes.
test_workspace_creation_denied_with_wrong_scope if {
	result := authz.allow with input as {
		"principal_id": "user@test.com",
		"method": "POST",
		"path": "/apis/entities/v2/workspaces",
		"scopes": ["platform:read"],
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as {
			"*": {"workspaces": {"system": ["WorkspaceCreator"]}},
		}

	result.allowed == false
}

# --- Bypass rules are unaffected ---

# Service principals can still access any endpoint, including unknown ones.
test_service_principal_bypasses_unknown_endpoint if {
	result := authz.allow with input as {
		"principal_id": "service:entity-store",
		"method": "POST",
		"path": "/apis/entities/v2/workspaces/test-ns/unknown-resource",
	}

	result.allowed == true
}

# Platform admins can still access any endpoint, including unknown ones.
test_platform_admin_bypasses_unknown_endpoint if {
	result := authz.allow with input as {
		"principal_id": "admin@test.com",
		"method": "DELETE",
		"path": "/apis/entities/v2/workspaces/test-ns/unknown-resource",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as unknown_endpoint_test_data.principals

	result.allowed == true
}

# --- Known endpoints still work as before ---

# A known endpoint with required permissions still works for a user with the right role.
# Use workspace object PUT (not nested .../entities/...) so nested-entities service-only deny does not apply.
test_known_endpoint_with_permissions_allowed if {
	result := authz.allow with input as {
		"principal_id": "editor@test.com",
		"method": "PUT",
		"path": "/apis/entities/v2/workspaces/test-ns",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as unknown_endpoint_test_data.principals

	result.allowed == true
}

# A known endpoint denies a user who lacks the required permission.
test_known_endpoint_denied_without_permission if {
	result := authz.allow with input as {
		"principal_id": "user@test.com",
		"method": "PUT",
		"path": "/apis/entities/v2/workspaces/test-ns/entities/models/my-model",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as unknown_endpoint_test_data.principals

	result.allowed == false
}

# A known cross-workspace LIST endpoint still works for authenticated users.
test_known_cross_workspace_list_still_allowed if {
	# Control for the fail-closed unknown-endpoint tests above: a KNOWN no-workspace list
	# endpoint is still allowed for a user holding the permission. lister@test.com has Viewer
	# (→ workspaces.list) in the system workspace, which listing now requires.
	result := authz.allow with input as {
		"principal_id": "lister@test.com",
		"method": "GET",
		"path": "/apis/entities/v2/workspaces",
	}
		with data.authz.roles as unknown_endpoint_test_data.roles
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
		with data.authz.workspaces as unknown_endpoint_test_data.workspaces
		with data.authz.principals as {"lister@test.com": {"workspaces": {"system": ["Viewer"]}}}

	result.allowed == true
}

# --- permission enforcement on permission-stamped no-workspace GETs ---

# A permissionless no-workspace GET stays open to any authenticated user (the cross-workspace
# list path is unchanged for endpoints declaring no permissions).
test_permissionless_no_workspace_get_allows_any_authenticated_user if {
	result := authz.allow with input as {
		"principal_id": "anyone@test.com",
		"method": "GET",
		"path": "/apis/example/v1/ping",
	}
		with data.authz.endpoints as {"/apis/example/v1/ping": {"get": {"permissions": []}}}
		with data.authz.principals as {"anyone@test.com": {"workspaces": {}}}
		with data.authz.roles as {}

	result.allowed == true
}

# A permission-stamped no-workspace GET is DENIED without the declared permission in the system
# workspace — the stamped permission is enforced rather than decorative.
test_permissioned_no_workspace_get_denied_without_permission if {
	result := authz.allow with input as {
		"principal_id": "nobody@test.com",
		"method": "GET",
		"path": "/apis/example/v1/hello/world",
	}
		with data.authz.endpoints as {"/apis/example/v1/hello/{name}": {"get": {"permissions": ["example.hello.read"]}}}
		with data.authz.principals as {"nobody@test.com": {"workspaces": {}}}
		with data.authz.roles as {}

	result.allowed == false
}

# ...and ALLOWED when the caller holds that permission in the system workspace.
test_permissioned_no_workspace_get_allowed_with_system_permission if {
	result := authz.allow with input as {
		"principal_id": "reader@test.com",
		"method": "GET",
		"path": "/apis/example/v1/hello/world",
	}
		with data.authz.endpoints as {"/apis/example/v1/hello/{name}": {"get": {"permissions": ["example.hello.read"]}}}
		with data.authz.principals as {"reader@test.com": {"workspaces": {"system": ["HelloReader"]}}}
		with data.authz.roles as {"HelloReader": {"permissions": ["example.hello.read"]}}

	result.allowed == true
}

# get_required_permissions is undefined for unknown endpoints (no longer returns []).
test_get_required_permissions_undefined_for_unknown if {
	not common.get_required_permissions("/apis/entities/v2/workspaces/test-ns/unknown", "GET")
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
}

# get_required_permissions still returns the configured permission for known endpoints.
test_get_required_permissions_for_workspace_creation if {
	perms := common.get_required_permissions("/apis/entities/v2/workspaces", "POST")
		with data.authz.endpoints as unknown_endpoint_test_data.endpoints
	perms == ["workspaces.create"]
}
