# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Test data with entities permissions in endpoints but NOT in roles
entities_deny_test_data := {
	"roles": {
		"Viewer": {
			"permissions": ["models.read", "workspaces.read", "workspaces.list"],
		},
		"Editor": {
			"includes": ["Viewer"],
			"permissions": ["models.create"],
		},
		"PlatformAdmin": {
			"includes": ["Editor"],
			"permissions": [],
		},
		"ServiceSystem": {
			"permissions": ["*"],
		},
	},
	"endpoints": {
		"/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}": {
			"get": {"permissions": ["entities.read"], "scopes": ["entities:read", "platform:read"]},
			"post": {"permissions": ["entities.create"], "scopes": ["entities:write", "platform:write"]},
		},
		"/apis/entities/v2/workspaces/{workspace}/entities/{entity_type}/{name}": {
			"get": {"permissions": ["entities.read"], "scopes": ["entities:read", "platform:read"]},
			"put": {"permissions": ["entities.update"], "scopes": ["entities:write", "platform:write"]},
			"delete": {"permissions": ["entities.delete"], "scopes": ["entities:write", "platform:write"]},
		},
		"/apis/entities/v2/entities/{id}": {
			"get": {"permissions": ["entities.read"], "scopes": ["entities:read", "platform:read"]},
		},
		"/apis/entities/v2/workspaces/{workspace}/members": {
			"get": {"permissions": ["workspaces.members.list"], "scopes": ["entities:read", "platform:read"]},
		},
		"/apis/entities/v2/workspaces": {
			"get": {"permissions": ["workspaces.list"], "scopes": ["entities:read", "platform:read"]},
		},
	},
	"workspaces": {
		"test-ws": {},
	},
	"principals": {
		"viewer@test.com": {
			"workspaces": {"test-ws": ["Viewer"]},
		},
		"editor@test.com": {
			"workspaces": {"test-ws": ["Editor"]},
		},
		"admin@test.com": {
			"workspaces": {"system": ["PlatformAdmin"]},
		},
	},
}

# Viewer cannot read entities (entities.read not in Viewer role)
test_viewer_denied_read_entities if {
	result := authz.allow
		with input as {
			"principal_id": "viewer@test.com",
			"method": "GET",
			"path": "/apis/entities/v2/workspaces/test-ws/entities/evaluation_config",
		}
		with data.authz.roles as entities_deny_test_data.roles
		with data.authz.endpoints as entities_deny_test_data.endpoints
		with data.authz.workspaces as entities_deny_test_data.workspaces
		with data.authz.principals as entities_deny_test_data.principals

	result.allowed == false
}

# Editor cannot create entities (entities.create not in Editor role)
test_editor_denied_create_entities if {
	result := authz.allow
		with input as {
			"principal_id": "editor@test.com",
			"method": "POST",
			"path": "/apis/entities/v2/workspaces/test-ws/entities/evaluation_config",
		}
		with data.authz.roles as entities_deny_test_data.roles
		with data.authz.endpoints as entities_deny_test_data.endpoints
		with data.authz.workspaces as entities_deny_test_data.workspaces
		with data.authz.principals as entities_deny_test_data.principals

	result.allowed == false
}

# Editor cannot read entities by name
test_editor_denied_get_entity_by_name if {
	result := authz.allow
		with input as {
			"principal_id": "editor@test.com",
			"method": "GET",
			"path": "/apis/entities/v2/workspaces/test-ws/entities/guardrail_config/my-config",
		}
		with data.authz.roles as entities_deny_test_data.roles
		with data.authz.endpoints as entities_deny_test_data.endpoints
		with data.authz.workspaces as entities_deny_test_data.workspaces
		with data.authz.principals as entities_deny_test_data.principals

	result.allowed == false
}

# Editor cannot update entities
test_editor_denied_update_entity if {
	result := authz.allow
		with input as {
			"principal_id": "editor@test.com",
			"method": "PUT",
			"path": "/apis/entities/v2/workspaces/test-ws/entities/role_binding/rb-1",
		}
		with data.authz.roles as entities_deny_test_data.roles
		with data.authz.endpoints as entities_deny_test_data.endpoints
		with data.authz.workspaces as entities_deny_test_data.workspaces
		with data.authz.principals as entities_deny_test_data.principals

	result.allowed == false
}

# Editor cannot delete entities
test_editor_denied_delete_entity if {
	result := authz.allow
		with input as {
			"principal_id": "editor@test.com",
			"method": "DELETE",
			"path": "/apis/entities/v2/workspaces/test-ws/entities/evaluation_config/eval-1",
		}
		with data.authz.roles as entities_deny_test_data.roles
		with data.authz.endpoints as entities_deny_test_data.endpoints
		with data.authz.workspaces as entities_deny_test_data.workspaces
		with data.authz.principals as entities_deny_test_data.principals

	result.allowed == false
}

# NOTE: GET /apis/entities/v2/entities/{id} is NOT tested here because the
# cross-workspace catch-all rule allows any authenticated user to access it.
# This is tracked in #3992 (OPA fail-open default for unknown endpoints).

# PlatformAdmin CAN access entities (admin bypass)
test_admin_allowed_read_entities if {
	result := authz.allow
		with input as {
			"principal_id": "admin@test.com",
			"method": "GET",
			"path": "/apis/entities/v2/workspaces/test-ws/entities/evaluation_config",
		}
		with data.authz.roles as entities_deny_test_data.roles
		with data.authz.endpoints as entities_deny_test_data.endpoints
		with data.authz.workspaces as entities_deny_test_data.workspaces
		with data.authz.principals as entities_deny_test_data.principals

	result.allowed == true
}

# Service principal CAN access entities (default ServiceSystem role = all permissions)
test_service_principal_allowed_entities if {
	result := authz.allow
		with input as {
			"principal_id": "service:evaluator",
			"method": "POST",
			"path": "/apis/entities/v2/workspaces/test-ws/entities/evaluation_config",
		}
		with data.authz.roles as entities_deny_test_data.roles
		with data.authz.endpoints as entities_deny_test_data.endpoints
		with data.authz.workspaces as entities_deny_test_data.workspaces
		with data.authz.principals as entities_deny_test_data.principals

	result.allowed == true
}

# Viewer CAN still list workspaces (non-entity endpoint unaffected)
test_workspace_viewer_cannot_list_workspaces if {
	# viewer@test.com is a Viewer of test-ws only, so it lacks workspaces.list in the system
	# workspace — listing workspaces now requires that system-level grant.
	result := authz.allow
		with input as {
			"principal_id": "viewer@test.com",
			"method": "GET",
			"path": "/apis/entities/v2/workspaces",
		}
		with data.authz.roles as entities_deny_test_data.roles
		with data.authz.endpoints as entities_deny_test_data.endpoints
		with data.authz.workspaces as entities_deny_test_data.workspaces
		with data.authz.principals as entities_deny_test_data.principals

	result.allowed == false
}
