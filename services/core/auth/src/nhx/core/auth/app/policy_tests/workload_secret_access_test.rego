# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

workload_secret_roles := {
	"JobRunner": data.authz.roles.JobRunner,
	"Reader": {"permissions": ["secrets.read"]},
}

workload_secret_principals := {
	"system:serviceaccount:qa:jobs": {"workspaces": {"team": ["JobRunner"]}},
	"creator": {"workspaces": {"team": ["Reader"]}},
}

workload_secret_endpoints := {"/apis/secrets/v2/workspaces/{workspace}/secrets/{name}/access": {"get": {"delegated": {"actor_perms": ["secrets.access"], "subject_perms": ["secrets.read"]}, "callers": ["service_principal"]}}}

workload_secret_allowed(kind, creator) := result.allowed if {
	result := authz.allow with input as {
		"principal_id": "system:serviceaccount:qa:jobs", "caller_kind": kind,
		"on_behalf_of_principal_id": creator, "method": "GET",
		"path": "/apis/secrets/v2/workspaces/team/secrets/key/access",
	}
		with data.authz.roles as workload_secret_roles
		with data.authz.principals as workload_secret_principals
		with data.authz.endpoints as workload_secret_endpoints
		with data.authz.workspaces as {"team": {}}
}

test_verified_workload_can_use_its_own_jobrunner_secret_grant if {
	workload_secret_allowed("service_principal", "creator")
}

test_unverified_actor_cannot_read_secret_with_jobrunner_role if {
	not workload_secret_allowed("principal", "creator")
}

test_actor_grant_does_not_replace_creator_secret_read if {
	not workload_secret_allowed("service_principal", "foreign-user")
}
