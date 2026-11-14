# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

package authz

import future.keywords.if
import future.keywords.in

import data.common

# Explicit lists select authority; legacy permissions means subject under OBO,
# otherwise actor. No admin shortcut can combine the two identity sets.
has_permissions := {"allowed_workspaces": permitted_query_workspaces} if {
	"workspaces" in object.keys(input)
} else := {"allowed": true} if permission_query_allowed(input.workspace)

else := {"allowed": false}

# Reuse scalar permission checks without per-workspace requests or input copies.
permitted_query_workspaces := [workspace |
	workspace := input.workspaces[_]
	permission_query_allowed(workspace)
] if {
	is_array(input.workspaces)
	every workspace in input.workspaces { is_string(workspace) }
} else := []

explicit_permission_query if "actor_permissions" in object.keys(input)
explicit_permission_query if "subject_permissions" in object.keys(input)

permission_query_allowed(workspace) if {
	explicit_permission_query
	count(common.get_actor_principals) > 0
	actor_query_allowed(workspace)
	subject_query_allowed(workspace)
} else if {
	not explicit_permission_query
	extract_on_behalf_of_principal_id == ""
	common.actor_permissions_satisfied(workspace, input.permissions)
} else if {
	not explicit_permission_query
	extract_on_behalf_of_principal_id != ""
	common.subject_permissions_satisfied(workspace, input.permissions)
}

actor_query_allowed(workspace) if {
	not "actor_permissions" in object.keys(input)
} else if {
	common.actor_permissions_satisfied(workspace, input.actor_permissions)
}

subject_query_allowed(workspace) if {
	not "subject_permissions" in object.keys(input)
} else if {
	common.subject_permissions_satisfied(workspace, input.subject_permissions)
}
