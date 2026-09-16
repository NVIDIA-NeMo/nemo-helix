# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check Permissions API
# Separate endpoint for checking if a principal has specific permissions in a workspace
# This is called by services that need to verify permissions outside of the request flow
package authz

import future.keywords.if
import future.keywords.in

import data.common

# API to check if principal has specific permissions in a workspace (called by services)
# Input format:
# {
#   "principal_id": "user@example.com",
#   "principal_email": "user@example.com", (optional)
#   "principal_groups": ["group1", "group2"], (optional)
#   "workspace": "my-workspace",
#   "permissions": ["models.create", "models.update"]
# }
# Output format:
# {
#   "allowed": true
# }
has_permissions := result if {
	# Platform admin bypass - has access to everything
	applicable_principals := common.get_applicable_principals
	count(applicable_principals) > 0

	# Check if any principal is a platform admin
	some principal in applicable_principals
	"PlatformAdmin" in data.authz.principals[principal].workspaces.system

	result := {"allowed": true}
} else := result if {
	workspace := input.workspace
	required_permissions := input.permissions

	# Get all applicable principals
	applicable_principals := common.get_applicable_principals
	count(applicable_principals) > 0

	# Check if any principal has all required permissions in the workspace
	# Note: has_permissions also checks wildcard principal "*" as fallback
	some principal in applicable_principals
	common.has_permissions(principal, workspace, required_permissions)

	result := {"allowed": true}
} else := result if {
	# Default deny
	result := {"allowed": false}
}
