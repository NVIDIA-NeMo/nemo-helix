# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
package authz_test

import data.authz

secret_path := "/apis/secrets/v2/workspaces/ws/secrets/key/access"
secret_contracts := {
	"/apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access": {"get": {"direct": {"actor_perms": ["secrets.access"]}, "delegated": {"actor_perms": ["secrets.access"], "subject_perms": ["secrets.read"]}, "scopes": ["secrets:read"]}},
	"/apis/other/v2/workspaces/{workspace}/secrets/{name}/access": {"get": {"direct": {"actor_perms": ["secrets.access"]}, "delegated": {"actor_perms": ["secrets.access"], "subject_perms": ["secrets.read"]}}},
}

secret_roles := {"DelegatedService": {"permissions": ["secrets.access"]}, "Reader": {"permissions": ["secrets.read"]}, "Restricted": {"permissions": ["jobs.read"]}, "HelixAdmin": {"permissions": ["*"]}}
secret_principals := {"readers": {"workspaces": {"ws": ["Reader"]}}, "service:restricted": {"workspaces": {"ws": ["Restricted"]}}, "creator-admin": {"workspaces": {"system": ["HelixAdmin"]}}}
secret_input := {"principal_id": "docker-runtime", "on_behalf_of_principal_id": "creator", "principal_groups": ["readers"], "method": "GET", "path": secret_path, "verified_workload_delegation": true, "scopes": ["secrets:read"]}

test_verified_secret_authority_matrix if {
	cases := [
		{"change": {}, "allowed": true},
		{"change": {"verified_workload_delegation": false}, "allowed": false},
		{"change": {"verified_workload_delegation": "true"}, "allowed": false},
		{"change": {"principal_groups": []}, "allowed": false},
		{"change": {"on_behalf_of_principal_id": ""}, "allowed": false},
		{"change": {"principal_id": "service:restricted"}, "allowed": false},
		{"change": {"path": "/apis/secrets/v2/workspaces/other/secrets/key/access"}, "allowed": false},
		{"change": {"path": "/apis/other/v2/workspaces/ws/secrets/key/access"}, "allowed": false},
		{"change": {"method": "POST"}, "allowed": false},
		{"change": {"scopes": ["jobs:read"]}, "allowed": false},
		{"change": {"verified_workload_delegation": false, "on_behalf_of_principal_id": "creator-admin"}, "allowed": false},
	]
	every case in cases {
		request := object.union(secret_input, case.change)
		result := authz.allow with input as request
			with data.authz.scope_families as {"secrets": true, "jobs": true, "platform": true}
			with data.authz.roles as secret_roles
			with data.authz.principals as secret_principals
			with data.authz.workspaces as {"ws": {}, "other": {}, "system": {}}
			with data.authz.endpoints as secret_contracts
		result.allowed == case.allowed
	}
}

test_verified_secret_requires_actor_capability if {
	result := authz.allow with input as secret_input
		with data.authz.roles as object.remove(secret_roles, ["DelegatedService"])
		with data.authz.principals as secret_principals
		with data.authz.workspaces as {"ws": {}}
		with data.authz.endpoints as secret_contracts
	result.allowed == false
}

test_verified_fact_keeps_global_caller_kind if {
	kind := data.common.request_caller_kind with input as secret_input
	kind == "principal"
}

# A true delegation fact changes neither unrelated route decisions nor generic
# permission queries, even if their request input also contains a Secret path.
test_verified_fact_does_not_expand_other_operations if {
	routes := {
		"/apis/audit/v2/workspaces/{workspace}/human": {"get": {"direct": {"actor_perms": ["secrets.read"]}, "delegated": {"actor_perms": [], "subject_perms": ["secrets.read"]}, "callers": ["principal"]}},
		"/apis/audit/v2/workspaces/{workspace}/mixed": {"get": {"direct": {"actor_perms": ["secrets.read"]}, "delegated": {"actor_perms": [], "subject_perms": ["secrets.read"]}, "callers": ["principal", "service_principal"]}},
		"/apis/audit/v2/workspaces/{workspace}/internal": {"get": {"direct": {"actor_perms": ["secrets.read"]}, "delegated": {"actor_perms": [], "subject_perms": ["secrets.read"]}, "callers": ["service_principal"]}},
	}
	cases := [
		{"path": "/apis/audit/v2/workspaces/ws/human", "allowed": true},
		{"path": "/apis/audit/v2/workspaces/ws/mixed", "allowed": true},
		{"path": "/apis/audit/v2/workspaces/ws/internal", "allowed": false},
		{"path": "/apis/entities/v2/workspaces/ws/entities/key", "allowed": false},
		{"path": "/apis/auth/v2/iam/roles", "allowed": false},
		{"path": "/apis/entities/v2/workspaces/ws/limits", "method": "PATCH", "allowed": false},
		{"path": "/apis/files/v2/workspaces/ws/filesets/key", "allowed": false},
		{"path": "/apis/auth/v2/access-keys", "method": "POST", "allowed": true},
	]
	every case in cases {
		request := object.union(secret_input, case)
		every flag in [false, true] {
			result := authz.allow with input as object.union(request, {"verified_workload_delegation": flag})
				with data.authz.roles as secret_roles
				with data.authz.principals as secret_principals
				with data.authz.workspaces as {"ws": {}, "system": {}}
				with data.authz.endpoints as object.union(data.authz.endpoints, routes)
			result.allowed == case.allowed
		}
	}
	every permission in ["secrets.access", "iam.create", "workspaces.delete"] {
		result := authz.has_permissions with input as object.union(secret_input, {"workspace": "ws", "permissions": [permission]})
			with data.authz.roles as secret_roles
			with data.authz.principals as secret_principals
		result.allowed == false
	}
}
