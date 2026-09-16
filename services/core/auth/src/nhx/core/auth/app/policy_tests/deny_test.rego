# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

# Tests for the explicit endpoint deny marker (data.authz.endpoints[...].deny == true),
# the fail-closed signal emitted for unruled / invalid plugin routes. A denied route must
# reject EVERY caller, overriding the empty-permissions allow, the ServiceSystem "*"
# wildcard, and the PlatformAdmin bypass.

deny_test_data := {
	"roles": {
		"Viewer": {"permissions": ["models.read"]},
		"PlatformAdmin": {"includes": ["Viewer"], "permissions": ["iam.read"]},
		# Default role for service:* principals: wildcard permission.
		"ServiceSystem": {"permissions": ["*"]},
	},
	"endpoints": {
		# Explicit deny marker (note permissions == [] would otherwise ALLOW any
		# authenticated caller via the empty-permissions rule — deny must override that).
		"/apis/agents/v2/workspaces/{workspace}/internal": {
			"get": {"permissions": [], "deny": true},
		},
		# Negative control: an ordinary route with no deny marker.
		"/apis/models/v2/workspaces/{workspace}/models/{name}": {
			"get": {"permissions": ["models.read"]},
		},
	},
	"workspaces": {"system": {}, "ws1": {}},
	"principals": {
		"platform-admin@example.com": {"workspaces": {"system": ["PlatformAdmin"]}},
		"user@example.com": {"workspaces": {"ws1": ["Viewer"]}},
	},
}

test_deny_marker_denies_human_principal if {
	result := authz.allow
		with input as {"principal_id": "user@example.com", "method": "GET", "path": "/apis/agents/v2/workspaces/ws1/internal"}
		with data.authz.roles as deny_test_data.roles
		with data.authz.endpoints as deny_test_data.endpoints
		with data.authz.workspaces as deny_test_data.workspaces
		with data.authz.principals as deny_test_data.principals

	result.allowed == false
}

# Overrides the ServiceSystem "*" wildcard that a service:* principal defaults to.
test_deny_marker_denies_service_principal if {
	result := authz.allow
		with input as {"principal_id": "service:agents", "method": "GET", "path": "/apis/agents/v2/workspaces/ws1/internal"}
		with data.authz.roles as deny_test_data.roles
		with data.authz.endpoints as deny_test_data.endpoints
		with data.authz.workspaces as deny_test_data.workspaces
		with data.authz.principals as deny_test_data.principals

	result.allowed == false
}

# Overrides the PlatformAdmin allow-bypass.
test_deny_marker_denies_platform_admin if {
	result := authz.allow
		with input as {"principal_id": "platform-admin@example.com", "method": "GET", "path": "/apis/agents/v2/workspaces/ws1/internal"}
		with data.authz.roles as deny_test_data.roles
		with data.authz.endpoints as deny_test_data.endpoints
		with data.authz.workspaces as deny_test_data.workspaces
		with data.authz.principals as deny_test_data.principals

	result.allowed == false
}

# A route without the deny marker is unaffected — normal permission check applies.
test_no_deny_marker_allows_with_permission if {
	result := authz.allow
		with input as {"principal_id": "user@example.com", "method": "GET", "path": "/apis/models/v2/workspaces/ws1/models/m1"}
		with data.authz.roles as deny_test_data.roles
		with data.authz.endpoints as deny_test_data.endpoints
		with data.authz.workspaces as deny_test_data.workspaces
		with data.authz.principals as deny_test_data.principals

	result.allowed == true
}

# Namespace fence: a path NOT in the endpoints map but under a degraded plugin's denied prefix
# must be denied — directly closing the service: no-match bypass for an unenumerable plugin.
test_denied_plugin_prefix_denies_service_principal if {
	result := authz.allow
		with input as {"principal_id": "service:x", "method": "POST", "path": "/apis/badplugin/v2/workspaces/ws1/anything/deep/path"}
		with data.authz.roles as deny_test_data.roles
		with data.authz.endpoints as deny_test_data.endpoints
		with data.authz.workspaces as deny_test_data.workspaces
		with data.authz.principals as deny_test_data.principals
		with data.authz.config as {"denied_plugin_prefixes": ["/apis/badplugin"]}

	result.allowed == false
}

# Control: WITHOUT the fence, that same unknown path is allowed for a service principal via the
# no-match bypass — proving the fence is what closes the hole.
test_unknown_path_without_fence_hits_service_bypass if {
	result := authz.allow
		with input as {"principal_id": "service:x", "method": "POST", "path": "/apis/badplugin/v2/workspaces/ws1/anything/deep/path"}
		with data.authz.roles as deny_test_data.roles
		with data.authz.endpoints as deny_test_data.endpoints
		with data.authz.workspaces as deny_test_data.workspaces
		with data.authz.principals as deny_test_data.principals

	result.allowed == true
}

# Sibling safety: a prefix-sharing namespace (/apis/badplugin-extra) is NOT collaterally fenced by
# /apis/badplugin. The trailing slash in the fence rule (sprintf("%s/", [prefix])) is the only thing
# keeping the sibling allowed; pin it so a refactor can't silently widen the fence onto a neighbour.
test_sibling_prefix_not_collaterally_fenced if {
	result := authz.allow
		with input as {"principal_id": "service:x", "method": "POST", "path": "/apis/badplugin-extra/v2/workspaces/ws1/anything"}
		with data.authz.roles as deny_test_data.roles
		with data.authz.endpoints as deny_test_data.endpoints
		with data.authz.workspaces as deny_test_data.workspaces
		with data.authz.principals as deny_test_data.principals
		with data.authz.config as {"denied_plugin_prefixes": ["/apis/badplugin"]}

	result.allowed == true
}

# the bare /apis/<plugin> route (the prefix with no trailing segment) must be fenced too.
# The old trailing-slash-only rule (startswith(path, "<prefix>/")) missed this exact-match case,
# leaving a degraded plugin's root path open.
test_bare_prefix_path_is_fenced if {
	result := authz.allow
		with input as {"principal_id": "service:x", "method": "GET", "path": "/apis/badplugin"}
		with data.authz.roles as deny_test_data.roles
		with data.authz.endpoints as deny_test_data.endpoints
		with data.authz.workspaces as deny_test_data.workspaces
		with data.authz.principals as deny_test_data.principals
		with data.authz.config as {"denied_plugin_prefixes": ["/apis/badplugin"]}

	result.allowed == false
}
