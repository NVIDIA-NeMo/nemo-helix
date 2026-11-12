# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz_test

import data.authz

batch_principals := {
	"admin": {"workspaces": {"system": ["Admin"], "private": ["Viewer"]}},
	"service:powerful": {"workspaces": {"system": ["HelixAdmin"]}},
}

test_batch_subject_permissions_do_not_borrow_actor_grants if {
	result := authz.has_permissions with input as {
		"principal_id": "service:powerful", "on_behalf_of_principal_id": "admin",
		"workspaces": ["system", "private", "unknown"], "subject_permissions": ["iam.list"],
	}
		with data.authz.principals as batch_principals
	result == {"allowed_workspaces": ["system"]}
}

test_batch_actor_and_subject_are_independent if {
	result := authz.has_permissions with input as {
		"principal_id": "admin", "on_behalf_of_principal_id": "service:powerful",
		"workspaces": ["system", "private"], "actor_permissions": ["iam.list"], "subject_permissions": ["iam.list"],
	}
		with data.authz.principals as batch_principals
	result == {"allowed_workspaces": ["system"]}
}

test_batch_requires_subject_for_explicit_empty_subject_permissions if {
	result := authz.has_permissions with input as {"principal_id": "service:powerful", "workspaces": ["system", "private"], "subject_permissions": []}
		with data.authz.principals as batch_principals
	result == {"allowed_workspaces": []}
}

test_invalid_workspace_batch_fails_closed if {
	every invalid in [null, {}, "system", ["system", 1]] {
		result := authz.has_permissions with input as {"principal_id": "service:powerful", "workspaces": invalid, "actor_permissions": ["iam.list"]}
			with data.authz.principals as batch_principals
		result == {"allowed_workspaces": []}
	}
}
