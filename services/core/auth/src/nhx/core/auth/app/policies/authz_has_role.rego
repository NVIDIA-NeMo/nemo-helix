# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check Roles API
# Separate endpoint for checking if a principal has a specific role in a workspace
#
# Primary use case: Polling for role membership changes to propagate to OPA
# When role memberships are modified (e.g., granting a user Admin role), there is a
# propagation delay before OPA receives the updated bundle. Services can poll this
# endpoint to wait for the changes to take effect, providing better UX by not requiring
# users to manually wait or refresh.
#
# Other use cases: Role-based feature flags, administrative operations requiring specific roles
package authz

import future.keywords.if
import future.keywords.in

import data.common

# API to check if principal has a specific role in a workspace (called by services)
# Input format:
# {
#   "principal_id": "user@example.com",
#   "principal_email": "user@example.com", (optional)
#   "principal_groups": ["group1", "group2"], (optional)
#   "workspace": "my-workspace",
#   "role": "Editor"
# }
# Output format:
# {
#   "has_role": true
# }
has_role := result if {
	workspace := input.workspace
	required_role := input.role

	# Get all applicable principals
	applicable_principals := common.get_applicable_principals
	count(applicable_principals) > 0

	# Check if any principal has the required role in the workspace (including ServiceSystem for service:*)
	some principal in applicable_principals
	some r in common.effective_roles(principal, workspace)
	required_role == r

	result := {"has_role": true}
} else := result if {
	# Default: does not have role
	result := {"has_role": false}
}
