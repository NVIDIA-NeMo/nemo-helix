# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if

# ASTD-526: the "default" workspace is the installation-wide GLOBAL workspace. Reads whose
# permissions are all marked global_read are satisfied for a principal holding them in any
# workspace, without a binding in "default".
global_read_test_data := {
	"roles": {
		"Viewer": {"permissions": ["models.read", "models.list", "secrets.read"]},
		"Editor": {
			"includes": ["Viewer"],
			"permissions": ["models.create", "models.update", "models.delete"],
		},
	},
	"endpoints": {
		"/apis/models/v2/workspaces/{workspace}/models": {
			"get": {"permissions": ["models.list"]},
			"head": {"permissions": ["models.list"]},
			"post": {"permissions": ["models.create"]},
		},
		"/apis/models/v2/workspaces/{workspace}/models/{name}": {
			"get": {"permissions": ["models.read"]},
			"patch": {"permissions": ["models.update"]},
			"delete": {"permissions": ["models.delete"]},
		},
		"/apis/secrets/v2/workspaces/{workspace}/secrets": {"get": {"permissions": ["secrets.read"]}},
	},
	"global_read_permissions": ["models.list", "models.read"],
	"workspaces": {"team-a": {}, "default": {}},
	"principals": {
		"viewer@test.com": {"workspaces": {"team-a": ["Viewer"]}},
		"editor@test.com": {"workspaces": {"team-a": ["Editor"]}},
		"no-access@test.com": {"workspaces": {}},
	},
}

global_read_result(principal, method, path) := result if {
	result := authz.allow with input as {
		"principal_id": principal,
		"method": method,
		"path": path,
	}
		with data.authz.roles as global_read_test_data.roles
		with data.authz.endpoints as global_read_test_data.endpoints
		with data.authz.workspaces as global_read_test_data.workspaces
		with data.authz.principals as global_read_test_data.principals
		with data.authz.global_read_permissions as global_read_test_data.global_read_permissions
}

test_global_read_allows_non_member_get if {
	result := global_read_result("viewer@test.com", "GET", "/apis/models/v2/workspaces/default/models/shared-llm")
	result.allowed == true
}

test_global_read_allows_non_member_list if {
	result := global_read_result("viewer@test.com", "GET", "/apis/models/v2/workspaces/default/models")
	result.allowed == true
}

# Endpoints rarely declare head; this one does, so the rule's HEAD arm is exercised.
test_global_read_allows_head if {
	result := global_read_result("viewer@test.com", "HEAD", "/apis/models/v2/workspaces/default/models")
	result.allowed == true
}

# The caller must already hold the read right somewhere; global is not public.
test_global_read_denies_principal_with_no_bindings if {
	result := global_read_result("no-access@test.com", "GET", "/apis/models/v2/workspaces/default/models/shared-llm")
	result.allowed == false
}

# Reads widen; writes do not. An Editor in team-a still cannot mutate global.
test_global_read_denies_post if {
	result := global_read_result("editor@test.com", "POST", "/apis/models/v2/workspaces/default/models")
	result.allowed == false
}

test_global_read_denies_patch if {
	result := global_read_result("editor@test.com", "PATCH", "/apis/models/v2/workspaces/default/models/shared-llm")
	result.allowed == false
}

test_global_read_denies_delete if {
	result := global_read_result("editor@test.com", "DELETE", "/apis/models/v2/workspaces/default/models/shared-llm")
	result.allowed == false
}

# A permission absent from the allowlist gets no global reach, even for a GET the caller
# can perform in its own workspace. This is what keeps secrets and role bindings in
# "default" unreadable to non-members.
test_global_read_denies_unlisted_permission if {
	result := global_read_result("viewer@test.com", "GET", "/apis/secrets/v2/workspaces/default/secrets")
	result.allowed == false
}

# The rule is specific to "default"; it must not widen reads of any other workspace.
test_global_read_does_not_widen_other_workspaces if {
	result := global_read_result("viewer@test.com", "GET", "/apis/models/v2/workspaces/other-team/models")
	result.allowed == false
}

test_member_of_own_workspace_still_allowed if {
	result := global_read_result("viewer@test.com", "GET", "/apis/models/v2/workspaces/team-a/models")
	result.allowed == true
}
