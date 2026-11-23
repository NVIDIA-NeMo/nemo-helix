# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import data.common
import future.keywords.if
import future.keywords.in

hf_read_requests := [
	{"method": "GET", "path": "/apis/files/v2/hf/api/models/default/model"},
	{"method": "GET", "path": "/apis/files/v2/hf/api/models/default/model/revision/main"},
	{"method": "GET", "path": "/apis/files/v2/hf/api/models/default/model/tree/main"},
	{"method": "POST", "path": "/apis/files/v2/hf/api/models/default/model/paths-info/main"},
	{"method": "HEAD", "path": "/apis/files/v2/hf/default/model/resolve/main/weights/model.safetensors"},
	{"method": "GET", "path": "/apis/files/v2/hf/default/model/resolve/main/weights/model.safetensors"},
]

hf_principals := {
	"creator": {"workspaces": {"default": ["Viewer"]}},
	"system:serviceaccount:cluster:default": {"workspaces": {"default": ["Admin"]}},
}

hf_allowed(request, subject, scopes) := allowed if {
	result := authz.allow with input as object.union(request, {
		"principal_id": "system:serviceaccount:cluster:default",
		"on_behalf_of_principal_id": subject,
		"scopes": scopes,
	})
		with data.authz.principals as hf_principals
		with data.authz.scope_families as {"files": true, "platform": true}
	allowed := result.allowed
}

test_hf_reader_can_use_every_download_route if {
	allowed := {request | some request in hf_read_requests; hf_allowed(request, "creator", ["platform:read"])}
	count(allowed) == count(hf_read_requests)
}

test_hf_actor_admin_cannot_bypass_creator_permissions if {
	allowed := {request | some request in hf_read_requests; hf_allowed(request, "outsider", ["platform:read"])}
	count(allowed) == 0
}

test_hf_workload_actor_needs_no_fileset_role if {
	result := authz.allow with input as {
		"principal_id": "system:serviceaccount:cluster:default",
		"on_behalf_of_principal_id": "creator",
		"method": "GET",
		"path": "/apis/files/v2/hf/api/models/default/model/revision/main",
		"scopes": [],
	}
		with data.authz.principals as {"creator": {"workspaces": {"default": ["Viewer"]}}}
	result.allowed
}

test_hf_write_scope_does_not_grant_read_access if {
	allowed := {request | some request in hf_read_requests; hf_allowed(request, "creator", ["files:write"])}
	count(allowed) == 0
}

test_hf_read_permission_does_not_cross_workspaces if {
	request := {"method": "GET", "path": "/apis/files/v2/hf/api/models/other/model/revision/main"}
	not hf_allowed(request, "creator", ["platform:read"])
}

# GET metadata shadows the catch-all download route. HEAD skips GET-only
# metadata routes, and longer download paths still address the api workspace.
hf_overlapping_requests := [
	{"method": "GET", "suffix": "/api/models/resolve/incident", "workspace": "resolve"},
	{"method": "GET", "suffix": "/api/models/resolve/incident/revision/main", "workspace": "resolve"},
	{"method": "GET", "suffix": "/api/models/resolve/incident/tree/main", "workspace": "resolve"},
	{"method": "POST", "suffix": "/api/models/resolve/incident/paths-info/main", "workspace": "resolve"},
	{"method": "HEAD", "suffix": "/api/models/resolve/incident/revision/main", "workspace": "api"},
	{"method": "GET", "suffix": "/api/models/resolve/incident/weights/model.bin", "workspace": "api"},
	{"method": "HEAD", "suffix": "/api/models/resolve/incident/weights/model.bin", "workspace": "api"},
	{"method": "GET", "suffix": "/resolve/model/resolve/main/config.json", "workspace": "resolve"},
]

test_hf_overlapping_routes_authorize_only_the_selected_workspace if {
	every request in hf_overlapping_requests {
		every granted_workspace in ["api", "resolve"] {
			result := authz.allow with input as {
				"principal_id": "reader",
				"method": request.method,
				"path": concat("", ["/apis/files/v2/hf", request.suffix]),
				"scopes": ["platform:read"],
			}
				with data.authz.principals as {"reader": {"workspaces": {granted_workspace: ["Viewer"]}}}
			expected := request.workspace == granted_workspace
			result.allowed == expected
		}
	}
}

test_hf_head_uses_download_contract_and_workspace if {
	request := {"method": "HEAD", "path": "/apis/files/v2/hf/api/models/resolve/incident/revision/main"}
	endpoint := common.endpoint_scan with input as request
	workspace := authz.workspace_scan with input as request
	endpoint == "/apis/files/v2/hf/{workspace}/{name}/resolve/{revision}/{path:path}"
	workspace == "api"
}

test_hf_unsupported_methods_cannot_use_admin_bypass if {
	every method in ["DELETE", "OPTIONS", "PATCH"] {
		request := {"method": method, "path": "/apis/files/v2/hf/api/models/resolve/incident/revision/main"}
		result := authz.allow with input as object.union(request, {"principal_id": "service:admin"})
			with data.authz.principals as {"service:admin": {"workspaces": {"system": ["HelixAdmin"]}}}
		not result.allowed
		endpoint := common.endpoint_scan with input as request
		endpoint == "/apis/files/v2/hf/api/models/{workspace}/{name}/revision/{revision}"
	}
}
