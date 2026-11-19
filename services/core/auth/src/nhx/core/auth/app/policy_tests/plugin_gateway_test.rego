# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package plugin_gateway_test

import data.authz
import data.common
import future.keywords.if

bindings := {
	"get": {"permissions": ["plugins.gateway.read"], "scopes": ["plugins:read", "platform:read"]},
	"post": {"permissions": ["plugins.gateway.write"], "scopes": ["plugins:write", "platform:write"]},
}

endpoints := {
	"/apis/plugins/{name}/{plugin_endpoint:path}": bindings,
	"/apis/plugins/{name}/v2/workspaces/{workspace}/{plugin_endpoint:path}": bindings,
}

check(path, method, scopes) := result if {
	result := authz.allow with input as {
		"principal_id": "gateway-viewer",
		"method": method,
		"path": path,
		"scopes": scopes,
	}
		with data.authz.principals as {"gateway-viewer": {"workspaces": {"alpha": ["Viewer"]}}}
		with data.authz.roles as {"Viewer": {"permissions": ["plugins.gateway.read"]}}
		with data.authz.endpoints as endpoints
		with data.authz.scope_families as {"platform": true, "plugins": true, "models": true}
}

test_nested_gateway_wildcard if {
	common.path_matches_pattern("/apis/plugins/phoenix/v1/traces/id/spans", "/apis/plugins/{name}/{plugin_endpoint:path}")
}

test_gateway_wildcard_requires_prefix if {
	not common.path_matches_pattern("/apis/models/phoenix/v1/traces", "/apis/plugins/{name}/{plugin_endpoint:path}")
}

test_gateway_wildcard_requires_suffix if {
	not common.path_matches_pattern("/apis/plugins/phoenix", "/apis/plugins/{name}/{plugin_endpoint:path}")
}

test_workspace_extracted_from_gateway_path if {
	workspace := authz.extract_workspace_from_path("/apis/plugins/phoenix/v2/workspaces/alpha/traces/id") with data.authz.endpoints as endpoints
	workspace == "alpha"
}

test_viewer_can_read_own_workspace if {
	result := check("/apis/plugins/phoenix/v2/workspaces/alpha/traces/id", "GET", ["plugins:read"])
	result.allowed == true
}

test_viewer_cannot_read_other_workspace if {
	result := check("/apis/plugins/phoenix/v2/workspaces/beta/traces/id", "GET", ["plugins:read"])
	result.allowed == false
}

test_viewer_cannot_write if {
	result := check("/apis/plugins/phoenix/v2/workspaces/alpha/traces/id", "POST", ["plugins:write"])
	result.allowed == false
}

test_scope_required_for_gateway_read if {
	result := check("/apis/plugins/phoenix/v2/workspaces/alpha/traces/id", "GET", ["models:read"])
	result.allowed == false
}

test_workspace_membership_does_not_grant_platform_route if {
	result := check("/apis/plugins/phoenix/v1/traces/id", "GET", ["plugins:read"])
	result.allowed == false
}

test_regular_placeholder_stays_single_segment if {
	not common.path_matches_pattern("/apis/plugins/phoenix/v1/traces/id", "/apis/plugins/{name}/{endpoint}")
}
