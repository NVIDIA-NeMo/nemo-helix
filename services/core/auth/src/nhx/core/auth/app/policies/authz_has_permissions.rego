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
has_permissions := {"allowed": true} if {
    common.principal_set_permissions(common.get_applicable_principals, input.workspace, input.permissions, false, [])
} else := {"allowed": false}
