# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz
import future.keywords.if

secret_actor := "system:serviceaccount:helix-builds:nhx-build-push"

workload_secret_allowed(actor, subject, workspace) := allowed if {
	result := authz.allow with input as {
		"method": "GET",
		"path": sprintf("/apis/secrets/v2/workspaces/%s/secrets/builder-registry-username/access", [workspace]),
		"principal_id": actor,
		"on_behalf_of_principal_id": subject,
		"nhx_workload_job": "builder-example-1",
		"nhx_workload_workspace": "system",
		"scopes": ["platform:read"],
	}
		with data.authz.principals as {
			secret_actor: {"workspaces": {"system": ["SecretAccessor"]}},
			"log-runner": {"workspaces": {"system": ["JobRunner"]}},
			"submitter": {"workspaces": {"system": ["Viewer"], "other": ["Viewer"]}},
			"non-reader": {"workspaces": {"system": ["JobRunner"]}},
		}
		with data.authz.scope_families as {"secrets": true, "platform": true}
	allowed := result.allowed
}

test_workload_secret_accessor_needs_actor_and_subject_grants if {
	workload_secret_allowed(secret_actor, "submitter", "system")
	not workload_secret_allowed("unbound", "submitter", "system")
	not workload_secret_allowed("log-runner", "submitter", "system")
	not workload_secret_allowed(secret_actor, "non-reader", "system")
	not workload_secret_allowed(secret_actor, "unknown", "system")
	not workload_secret_allowed(secret_actor, "submitter", "other")
}

test_secret_accessor_does_not_include_secret_management if {
	data.authz.roles.SecretAccessor.permissions == ["secrets.access"]
}

test_secret_accessor_rejects_unbound_and_direct_tokens if {
	not workload_secret_allowed(secret_actor, "", "system")
	result := authz.allow with input as {
		"method": "GET",
		"path": "/apis/secrets/v2/workspaces/system/secrets/builder-registry-username/access",
		"principal_id": secret_actor,
		"on_behalf_of_principal_id": "submitter",
		"scopes": ["platform:read"],
	}
		with data.authz.principals as {
			secret_actor: {"workspaces": {"system": ["SecretAccessor"]}},
			"submitter": {"workspaces": {"system": ["Viewer"]}},
		}
		with data.authz.scope_families as {"secrets": true, "platform": true}
	not result.allowed
}
